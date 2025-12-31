import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np
np.ComplexWarning = Warning

from typing import Callable

# --- monekypatch for tc---
import jax
try:
    _ = jax.tree_map  # JAX < 0.6 has this
except AttributeError:
    # JAX ≥ 0.6 moved it here
    from jax import tree_util as _jtu
    jax.tree_map = _jtu.tree_map

import tensorcircuit as tc

import tensorcircuit as tc
import jax.numpy as jnp
import flax.linen

import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorcircuit as tc

K = tc.set_backend("jax")


def angle_embedding(c: tc.Circuit, inputs):
    num_qubits = inputs.shape[-1]

    for j in range(num_qubits):
        c.rx(j, theta=inputs[j])


def basic_vqc(c: tc.Circuit, inputs, weights):
    num_qubits = inputs.shape[-1]
    num_qlayers = weights.shape[-2]

    for i in range(num_qlayers):
        for j in range(num_qubits):
            c.rx(j, theta=weights[i, j])
        if num_qubits == 2:
            c.cnot(0, 1)
        elif num_qubits > 2:
            for j in range(num_qubits):
                c.cnot(j, (j + 1) % num_qubits)


def get_quantum_layer_circuit(inputs, weights,
                              embedding: Callable = angle_embedding, vqc: Callable = basic_vqc):
    """
    Equivalent to the following PennyLane circuit:
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(num_qubits))
            qml.templates.BasicEntanglerLayers(weights, wires=range(num_qubits))
    """

    num_qubits = inputs.shape[-1]

    c = tc.Circuit(num_qubits)
    embedding(c, inputs)
    vqc(c, inputs, weights)

    return c


def get_circuit(embedding: Callable = angle_embedding, vqc: Callable = basic_vqc,
                torch_interface: bool = False):
    def qpred(inputs, weights):
        c = get_quantum_layer_circuit(inputs, weights, embedding, vqc)
        return K.real(jnp.array([c.expectation_ps(z=[i]) for i in range(weights.shape[1])]))

    qpred_batch = K.vmap(qpred, vectorized_argnums=0)
    if torch_interface:
        qpred_batch = tc.interfaces.torch_interface(qpred_batch, jit=True)

    return qpred_batch


class QuantumLayer(flax.linen.Module):
    circuit: Callable
    num_qubits: int
    w_shape: tuple = (1,)

    @flax.linen.compact
    def __call__(self, x):
        shape = x.shape
        x = jnp.reshape(x, (-1, shape[-1]))
        w = self.param('w', flax.linen.initializers.xavier_normal(), self.w_shape + (self.num_qubits,))
        x = self.circuit(x, w)
        x = jnp.concatenate(x, axis=-1)
        x = jnp.reshape(x, tuple(shape))
        return x

NUM_QUBITS     = 8
NUM_Q_LAYERS   = 1
torch_layer_fn = get_circuit(torch_interface=True)


class TCTorchLayer(nn.Module):
    """
    A thin PyTorch wrapper around the TensorCircuit/TC quantum layer.
    Stores the circuit's trainable parameters as an nn.Parameter so
    they appear in .parameters() and get updated by any torch optimizer.
    """
    def __init__(self, num_qubits=NUM_QUBITS, num_qlayers=NUM_Q_LAYERS):
        super().__init__()
        init_w = 0.01 * torch.randn(num_qlayers, num_qubits)
        self.w = nn.Parameter(init_w)
        self.num_qubits = num_qubits

    def forward(self, x):
        """
        x: (batch, num_qubits) – already pre-scaled into rotation angles.
        Returns expectation values ⟨Z_i⟩ for every qubit i, shape identical
        to the input (batch, num_qubits).
        """
        return torch_layer_fn(x, self.w)


class QuantumLinear(nn.Module):
    """
    Linear -> angle map -> TCTorchLayer -> Linear
    Works on tensors shaped (..., din) and returns (..., dout).
    """
    def __init__(self, din, dout, num_qubits):
        super().__init__()
        self.din  = din
        self.dout = dout
        self.nq   = num_qubits

        #self.to_q   = nn.Linear(din,  self.nq, bias=False)
        #self.from_q = nn.Linear(self.nq, dout, bias=False)
        self.q = TCTorchLayer(self.nq)

    @staticmethod
    def _to_angles(x):
        return torch.tanh(x) * math.pi

    def forward(self, x):
        # x: (..., din)
        *prefix, _ = x.shape
        x = x.reshape(-1, self.din)

        #x = self.to_q(x)
        x = self._to_angles(x)
        x = self.q(x).float()
        #x = self.from_q(x)

        x = x.reshape(*prefix, self.dout)
        return x

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
    """
    Same interface as your tiny MLP, but nn.Linear -> QuantumLinear.
    Works for inputs shaped (..., dim).

    Args:
        dim         : feature size
        dropout     : dropout prob
        num_qubits  : qubits per QuantumLinear block (defaults to dim)
    """
    def __init__(self, dim, dropout=0., num_qubits=None):
        super().__init__()
        nq = num_qubits if num_qubits is not None else dim

        self.fc1 = QuantumLinear(dim, dim, nq)
        self.fc2 = QuantumLinear(dim, dim, nq)

        self.act  = nn.GELU()
        self.do1  = nn.Dropout(dropout)
        self.do2  = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.do1(x)

        x = self.fc2(x)
        x = self.do2(x)
        return x

class ParticleMHA(nn.Module):
    """
    Multi-head self-attention with quantum projections (q, k, v, o).

    Args
    ----
    d            : embedding dim
    heads        : number of attention heads
    dropout      : dropout prob on attn weights
    return_attn  : return attention maps?
    num_qubits   : qubits per quantum block (defaults to d)
    """
    def __init__(self, d: int, heads: int = 8,
                 dropout: float = 0.1, return_attn: bool = False,
                 num_qubits: int | None = None):
        super().__init__()
        assert d % heads == 0, "`d` must be divisible by `heads`"

        self.d           = d
        self.h           = heads
        self.d_head      = d // heads
        self.scale       = 1 / math.sqrt(self.d_head)
        self.return_attn = return_attn

        nq = num_qubits if num_qubits is not None else d

        # quantum projections
        self.q_proj = QuantumLinear(d, d, nq)
        self.k_proj = QuantumLinear(d, d, nq)
        self.v_proj = QuantumLinear(d, d, nq)
        self.o_proj = QuantumLinear(d, d, nq)

        self.drop = nn.Dropout(dropout)

    def _split(self, t: torch.Tensor):
        # (B, N, d) -> (B, H, N, d_head)
        B, N, _ = t.shape
        return t.view(B, N, self.h, self.d_head).transpose(1, 2)

    def forward(self, x: torch.Tensor, U: torch.Tensor | None = None):
        B, N, _ = x.shape

        Q = self._split(self.q_proj(x))
        K = self._split(self.k_proj(x))
        V = self._split(self.v_proj(x))

        logits = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        if U is not None:
            logits = logits + U

        attn = F.softmax(logits, dim=-1)
        attn = self.drop(attn)

        context = attn @ V  # (B, H, N, d_head)

        context = (
            context.transpose(1, 2)   # (B, N, H, d_head)
                   .contiguous()
                   .view(B, N, self.d)
        )
        out = self.o_proj(context)

        if self.return_attn:
            return out, attn
        else:
            return out

class MHA(nn.Module):
    """
    Multi-head attention (batch_first) with QuantumLinear projections.

    Args
    ----
    d_model : int          embedding dim
    n_heads : int
    dropout: float
    bias   : bool          (ignored here, QuantumLinear has no bias)
    num_qubits : int|None  qubits per quantum block (defaults to d_model)
    """
    def __init__(self, d_model: int, n_heads: int,
                 dropout: float = 0., bias: bool = False,
                 num_qubits: int | None = None):
        super().__init__()
        assert d_model % n_heads == 0, "`d_model` must be divisible by `n_heads`"
        self.d_model = d_model
        self.h       = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5

        nq = num_qubits if num_qubits is not None else d_model

        # Quantum projections replace nn.Linear
        self.q_proj = QuantumLinear(d_model, d_model, nq)
        self.k_proj = QuantumLinear(d_model, d_model, nq)
        self.v_proj = QuantumLinear(d_model, d_model, nq)
        self.o_proj = QuantumLinear(d_model, d_model, nq)

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
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False
    ):
        B, Lq, _ = q.shape
        _, Lk, _ = k.shape

        Q = self._split_heads(self.q_proj(q))  # (B,h,Lq,d_h)
        K = self._split_heads(self.k_proj(k))  # (B,h,Lk,d_h)
        V = self._split_heads(self.v_proj(v))  # (B,h,Lk,d_h)

        logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B,h,Lq,Lk)

        attn = F.softmax(logits, dim=-1)
        attn = self.drop(attn)

        context = torch.matmul(attn, V)        # (B,h,Lq,d_h)

        out = self.o_proj(self._merge_heads(context))  # (B,Lq,d_model)

        if need_weights:
            return out, attn.mean(dim=1)  # (B,Lq,Lk)
        return out, None

# Particle attention block  (NormFormer style + U-bias)
class ParticleAttentionBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = ParticleMHA(dim, heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dropout)
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
        self.mlp = MLP(dim, dropout)
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

        # weight init
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