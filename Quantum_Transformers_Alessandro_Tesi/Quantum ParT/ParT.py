import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InteractionEncoder(nn.Module):
    """
    ParT interaction-feature encoder.

    Args
    ----
    n_heads per mhsa: output channels d′
    hidden_channels : list[int] for intermediate 1×1 conv layers
    eps             : numerical guard for log
    """

    def __init__(self,
                 n_heads: int = 8,
                 hidden_channels: list[int] = (64, 64, 64),
                 eps: float = 1e-8):
        super().__init__()
        self.eps = eps

        layers: list[nn.Module] = []
        in_ch = 4                               # lnΔ, ln kT, ln z, ln m²
        for h in hidden_channels:
            layers += [
                nn.Conv2d(in_ch, h, 1, bias=False),
                nn.BatchNorm2d(h),
                nn.GELU()
            ]
            in_ch = h
        layers.append(nn.Conv2d(in_ch, n_heads, 1, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 4, N)  where the 4 dims are (E, px, py, pz)
        returns
        ------
        U : (B, n_heads, N, N)  interaction embedding
        """
        B, four, N = x.shape
        assert four == 4, "input must have 4 features: E, px, py, pz"

        # Split components
        E, px, py, pz = x.unbind(dim=1)         # each (B, N)

        # Basic kinematics ------------------------------------------------
        pT = torch.sqrt(px**2 + py**2) + self.eps
        phi = torch.atan2(py, px)               # (−π, π]
        num = (E + pz).clamp(min=self.eps)  #need to avoid negative numbers
        den = (E - pz).clamp(min=self.eps)
        y   = 0.5 * torch.log(num / den)

        # Expand to (B, N, N)
        y_a, y_b = y.unsqueeze(2), y.unsqueeze(1)          # (B,N,1),(B,1,N)
        phi_a, phi_b = phi.unsqueeze(2), phi.unsqueeze(1)
        pT_a, pT_b = pT.unsqueeze(2), pT.unsqueeze(1)
        E_a, E_b = E.unsqueeze(2), E.unsqueeze(1)
        px_a, px_b = px.unsqueeze(2), px.unsqueeze(1)
        py_a, py_b = py.unsqueeze(2), py.unsqueeze(1)
        pz_a, pz_b = pz.unsqueeze(2), pz.unsqueeze(1)

        # ΔR, kT, z
        delta = torch.sqrt((y_a - y_b) ** 2 + (phi_a - phi_b) ** 2) + self.eps
        kT = torch.minimum(pT_a, pT_b) * delta
        z = torch.minimum(pT_a, pT_b) / (pT_a + pT_b + self.eps)

        # m² of pair
        E_sum = E_a + E_b
        px_sum = px_a + px_b
        py_sum = py_a + py_b
        pz_sum = pz_a + pz_b
        m2 = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2) + self.eps
        m2 = torch.clamp(m2, min=self.eps)      # avoid negatives

        # Stack → (B, 4, N, N)
        feats = torch.stack([
            torch.log(delta),
            torch.log(kT),
            torch.log(z),
            torch.log(m2)
        ], dim=1)

        # conv
        U = self.net(feats)                     # (B, n_heads, N, N)
        return U


class ParticleTokenizer(nn.Module):
    def __init__(self, in_dim=4, out_dim=6):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        """
        x: tensor of shape (B, n_particles, in_dim)
        returns: (B, n_particles, out_dim)
        """
        x = x.transpose(1, 2)  # Input shape: (B, n_particles, in_dim) → (B, in_dim, n_particles)
        return self.proj(x)

class MLP(nn.Module):
    def __init__(self, dim, expansion=1, dropout=0.):
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class ParticleMHA(nn.Module):
    """
    Multi-head self-attention with additive interaction bias U.

    Input
    -----
    x        : (B, N, d)        token / particle embeddings
    U        : (broadcast → B, H, N, N) or None

    Returns
    -------
    out      : (B, N, d)        attention output
    attn_map : (B, H, N, N)     attention weights (returned if
                                 return_attn=True)
    """
    def __init__(self, d: int, heads: int = 8,
                 dropout: float = 0.1, return_attn: bool = False):
        super().__init__()
        assert d % heads == 0, "`d` must be divisible by `heads`"

        self.d          = d
        self.h          = heads
        self.d_head     = d // heads
        self.scale      = 1 / math.sqrt(self.d_head)
        self.return_attn = return_attn

        # Projections
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

        self.drop = nn.Dropout(dropout)

    def _split(self, t: torch.Tensor):
        # (B, N, d) -> (B, H, N, d_head)
        B, N, _ = t.shape
        return (
            t.view(B, N, self.h, self.d_head)       # (B, N, H, d_head)
             .transpose(1, 2)                       # (B, H, N, d_head)
        )

    def forward(self, x: torch.Tensor,
                U: torch.Tensor | None = None):
        B, N, _ = x.shape

        Q = self._split(self.q(x))
        K = self._split(self.k(x))
        V = self._split(self.v(x))

        logits = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        if U is not None:
            logits = logits + U

        attn = F.softmax(logits, dim=-1)
        attn = self.drop(attn)

        context = attn @ V                               # (B, H, N, d_h)

        context = (
            context.transpose(1, 2)                      # (B, N, H, d_h)
                   .contiguous()
                   .view(B, N, self.d)                   # (B, N, d)
        )
        out = self.o(context)

        if self.return_attn:
            return out, attn        # (B, N, d), (B, H, N, N)
        else:
            return out

class MHA(nn.Module):
    """
    Multi-head attention (batch_first) implemented explicitly.

    Args
    ----
    d_model : int          embedding dim
    n_heads : int
    dropout: float
    bias   : bool          use bias in projections
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0., bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "`d_model` must be divisible by `n_heads`"
        self.d_model  = d_model
        self.h        = n_heads
        self.d_head   = d_model // n_heads
        self.scale    = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        self.drop = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor):
        # (B, L, d_model) -> (B, h, L, d_head)
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor):
        # (B, h, L, d_head) -> (B, L, d_model)
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def forward(
        self,
        q: torch.Tensor,          # (B, Lq, d_model)
        k: torch.Tensor,          # (B, Lk, d_model)
        v: torch.Tensor,          # (B, Lk, d_model)
        need_weights: bool = False
    ):
        B, Lq, _ = q.shape
        _, Lk, _ = k.shape

        Q = self._split_heads(self.q_proj(q))    # (B,h,Lq,d_h)
        K = self._split_heads(self.k_proj(k))    # (B,h,Lk,d_h)
        V = self._split_heads(self.v_proj(v))    # (B,h,Lk,d_h)

        logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale   # (B,h,Lq,Lk)

        attn = F.softmax(logits, dim=-1)
        attn = self.drop(attn)

        context = torch.matmul(attn, V)          # (B,h,Lq,d_h)

        # merge heads + output proj
        out = self.o_proj(self._merge_heads(context))  # (B,Lq,d_model)

        if need_weights:
            avg_weights = attn.mean(dim=1)            # (B,Lq,Lk)
            return out, avg_weights
        return out, None


# Particle attention block  (NormFormer style + U-bias)
class ParticleAttentionBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = ParticleMHA(dim, heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)
    def forward(self, x, U):
        x = x + self.attn(self.ln1(x), U)    # bias-aware MHSA
        x = x + self.mlp(self.ln2(x))        # feed-forward
        return x

# Class attention block  (CaiT style, no U)
class ClassAttentionBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, dropout=0.):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MHA(dim, heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)
    def forward(self, tokens, cls):          # tokens: (B,N,d), cls: (B,1,d)
        z   = torch.cat([cls, tokens], dim=1)   # (B,1+N,d)
        q   = self.ln1(cls)
        kv  = self.ln1(z)
        cls = cls + self.attn(q, kv, kv, need_weights=False)[0]
        cls = cls + self.mlp(self.ln2(cls))
        return cls                             # (B,1,d)

# Complete Particle Transformer
class ParT(nn.Module):
    def __init__(self,
                 in_dim=4,          # (E,px,py,pz)
                 embed_dim=10,
                 n_heads=2,
                 depth=2,           # particle blocks
                 class_depth=2,     # class-attention blocks
                 mlp_ratio=4,
                 num_classes=10,
                 dropout=0.1):
        super().__init__()

        self.tokenizer = ParticleTokenizer(in_dim, embed_dim)
        self.U_encoder = InteractionEncoder(n_heads=n_heads)

        self.blocks = nn.ModuleList([
            ParticleAttentionBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_blocks = nn.ModuleList([
            ClassAttentionBlock(embed_dim, n_heads, mlp_ratio, 0.0)
            for _ in range(class_depth)
        ])

        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight,  std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):               # x: (B,4,N)
        B, _, N = x.shape

        tokens = self.tokenizer(x)                  # (B,N,d)
        U      = self.U_encoder(x)                  # (B,H,N,N)

        for blk in self.blocks:
            tokens = blk(tokens, U)                # (B,N,d)

        cls = self.class_token.expand(B, -1, -1)    # (B,1,d)
        for blk in self.cls_blocks:
            cls = blk(tokens, cls)                 # (B,1,d)

        logits = self.head(cls.squeeze(1))          # (B,10)
        return logits