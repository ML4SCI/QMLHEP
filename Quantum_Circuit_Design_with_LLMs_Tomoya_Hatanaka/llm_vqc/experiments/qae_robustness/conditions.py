"""Condition definitions and the deterministic circuit space / data / trainer
generalisations used by the QAE robustness study.

Everything here reduces EXACTLY to the reference implementation
(`llm_vqc.experiments.qae_tfim.pilot` + `neutral_v3/v4/v5`) when evaluated
at the reference condition n=4, TFIM, B=8. `tests/test_qae_robustness_*.py`
proves that reduction bit-for-bit; do not "improve" any numeric detail here
without breaking those tests on purpose.

Width rule (pre-registered, identical for every method): 3 trainable
single-qubit rotations + 1 CNOT per qubit, i.e. 3n rotations and n CNOTs
in a 4n-gate ordered sequence. At n=4 this is the reference 12R + 4CX.

Latent/trash split (pre-registered): the first n/2 qubits are latent, the
last n/2 are trash. At n=4 this is the reference latent q0,q1 / trash q2,q3.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import torch

from llm_vqc.experiments.qae_tfim.pilot import (
    EPOCHS,
    I2,
    LEARNING_RATE,
    EvalResult,
    X,
    Y,
    Z,
    architecture_key,
)

# --------------------------------------------------------------- factors ---

REFERENCE_N_QUBITS = 4
REFERENCE_FAMILY = "TFIM"
REFERENCE_BUDGET = 8
REFERENCE_MODEL = "gpt-5.4-mini-2026-03-17"
ALTERNATIVE_MODEL = "gpt-4.1-mini-2025-04-14"

VERIFY_SEEDS = tuple(range(12))
FAMILIES = ("TFIM", "XXZ")
AXES = ("X", "Y", "Z")
PARAM_LOW, PARAM_HIGH = 0.2, 2.0


@dataclass(frozen=True)
class Condition:
    """One cell of the one-factor-at-a-time matrix."""

    key: str
    factor: str  # "reference" | "budget" | "qubits" | "hamiltonian" | "model"
    n_qubits: int = REFERENCE_N_QUBITS
    family: str = REFERENCE_FAMILY
    budget: int = REFERENCE_BUDGET
    model: str = REFERENCE_MODEL
    label: str = ""

    def __post_init__(self) -> None:
        if self.n_qubits % 2 != 0 or self.n_qubits < 2:
            raise ValueError("n_qubits must be even and >= 2 (half latent / half trash)")
        if self.family not in FAMILIES:
            raise ValueError(f"unknown Hamiltonian family {self.family}")
        if self.budget % 2 != 0 or self.budget < 2:
            raise ValueError("budget must be even (50/50 exploration/refinement split)")

    @property
    def n_warm(self) -> int:
        """Exploration evaluations of each adaptive method (50/50 split)."""
        return self.budget // 2

    @property
    def factors(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "family": self.family,
            "budget": self.budget,
            "model": self.model,
        }


REFERENCE = Condition(
    key="reference", factor="reference",
    label="Previous QAE baseline (4 qubits, TFIM, B=8, reference model)",
)

CONDITIONS: tuple[Condition, ...] = (
    REFERENCE,
    Condition("budget_b4", "budget", budget=4, label="B = 4"),
    Condition("budget_b16", "budget", budget=16, label="B = 16"),
    Condition("qubits_n6", "qubits", n_qubits=6, label="6 qubits"),
    Condition("qubits_n8", "qubits", n_qubits=8, label="8 qubits"),
    Condition("hamiltonian_xxz", "hamiltonian", family="XXZ", label="XXZ chain"),
    Condition("model_alt", "model", model=ALTERNATIVE_MODEL, label="Alternative model"),
)

CONDITIONS_BY_KEY = {c.key: c for c in CONDITIONS}


def changed_factors(
    condition: Condition, anchor: Condition | None = None
) -> list[str]:
    """Names of the factors that differ from `anchor` (default: the study
    reference condition).

    An explicit anchor is what makes a *second* protocol expressible without
    weakening the original one-factor rule: a budget probe anchored at the
    XXZ cell is one factor away from XXZ even though it is two factors away
    from the TFIM reference. The default argument keeps every existing
    caller and every existing test on the reference anchor.
    """
    base = (anchor or REFERENCE).factors
    return sorted(k for k, v in condition.factors.items() if base[k] != v)


# ----------------------------------------------------------------- space ---


@dataclass(frozen=True)
class Space:
    """The neutral architecture space at a given qubit count."""

    n: int
    wires: tuple[int, ...] = field(init=False)
    directed_pairs: tuple[tuple[int, int], ...] = field(init=False)

    def __post_init__(self) -> None:
        wires = tuple(range(self.n))
        object.__setattr__(self, "wires", wires)
        object.__setattr__(
            self, "directed_pairs",
            tuple((c, t) for c in wires for t in wires if c != t),
        )

    @property
    def n_rotations(self) -> int:
        return 3 * self.n

    @property
    def n_cnots(self) -> int:
        return self.n

    @property
    def n_gates(self) -> int:
        return self.n_rotations + self.n_cnots

    @property
    def latent(self) -> tuple[int, ...]:
        return tuple(range(self.n // 2))

    @property
    def trash(self) -> tuple[int, ...]:
        return tuple(range(self.n // 2, self.n))


def space_for(condition: Condition) -> Space:
    return Space(condition.n_qubits)


def verify_capacity(architecture: list[dict], space: Space) -> None:
    if len(architecture) != space.n_gates:
        raise ValueError(f"expected {space.n_gates} gates, got {len(architecture)}")
    rotations = sum(op["type"] == "R" for op in architecture)
    cnots = sum(op["type"] == "CX" for op in architecture)
    if rotations != space.n_rotations or cnots != space.n_cnots:
        raise ValueError(f"capacity mismatch: rotations={rotations}, CNOTs={cnots}")
    for op in architecture:
        if op["type"] == "R":
            if op["axis"] not in AXES or op["q"] not in space.wires:
                raise ValueError(f"invalid rotation {op}")
        else:
            if (op["c"], op["t"]) not in space.directed_pairs:
                raise ValueError(f"invalid CNOT {op}")


def sample_neutral_arch(rng: np.random.Generator, space: Space) -> list[dict]:
    """Uniform draw from the neutral space (identical stream consumption to
    the reference `neutral_v3.sample_neutral_arch` at n=4)."""
    ops: list[dict] = [
        {"type": "R", "axis": str(rng.choice(AXES)), "q": int(rng.choice(space.wires))}
        for _ in range(space.n_rotations)
    ]
    for _ in range(space.n_cnots):
        control, target = space.directed_pairs[int(rng.integers(len(space.directed_pairs)))]
        ops.append({"type": "CX", "c": control, "t": target})
    order = rng.permutation(space.n_gates)
    architecture = [ops[i] for i in order]
    verify_capacity(architecture, space)
    return architecture


def mutate_one_choice(
    architecture: list[dict], rng: np.random.Generator, space: Space
) -> list[dict]:
    """Exactly one structural change (frozen Greedy move set)."""
    arch = [dict(op) for op in architecture]
    rotation_idx = [i for i, op in enumerate(arch) if op["type"] == "R"]
    cnot_idx = [i for i, op in enumerate(arch) if op["type"] == "CX"]
    wires = space.wires
    move = int(rng.integers(5))
    if move == 0:  # one rotation axis
        i = rotation_idx[int(rng.integers(len(rotation_idx)))]
        arch[i]["axis"] = str(rng.choice([a for a in AXES if a != arch[i]["axis"]]))
    elif move == 1:  # one rotation qubit
        i = rotation_idx[int(rng.integers(len(rotation_idx)))]
        arch[i]["q"] = int(rng.choice([w for w in wires if w != arch[i]["q"]]))
    elif move == 2:  # one CNOT control
        i = cnot_idx[int(rng.integers(len(cnot_idx)))]
        options = [w for w in wires if w != arch[i]["c"] and w != arch[i]["t"]]
        arch[i]["c"] = int(rng.choice(options))
    elif move == 3:  # one CNOT target
        i = cnot_idx[int(rng.integers(len(cnot_idx)))]
        options = [w for w in wires if w != arch[i]["t"] and w != arch[i]["c"]]
        arch[i]["t"] = int(rng.choice(options))
    else:  # swap two gate positions
        i, j = rng.choice(space.n_gates, size=2, replace=False)
        arch[int(i)], arch[int(j)] = arch[int(j)], arch[int(i)]
    verify_capacity(arch, space)
    return arch


def parse_candidate(entry: dict, space: Space) -> tuple[list[dict] | None, list[str]]:
    """Turn one JSON candidate into an architecture (frozen validation logic)."""
    if not isinstance(entry, dict):
        return None, ["candidate is not an object"]
    gates = entry.get("gates")
    if not isinstance(gates, list) or len(gates) != space.n_gates:
        return None, [f"gates must be a list of exactly {space.n_gates} entries"]
    top = space.n - 1
    architecture: list[dict] = []
    errors: list[str] = []
    for i, g in enumerate(gates):
        if not isinstance(g, dict):
            errors.append(f"gate {i} is not an object")
            continue
        kind = g.get("g")
        if kind in ("RX", "RY", "RZ"):
            q = g.get("q")
            if not isinstance(q, int) or q not in space.wires:
                errors.append(f"gate {i}: rotation qubit must be 0-{top}")
                continue
            architecture.append({"type": "R", "axis": kind[1], "q": q})
        elif kind == "CNOT":
            c, t = g.get("c"), g.get("t")
            if (not isinstance(c, int) or not isinstance(t, int)
                    or (c, t) not in space.directed_pairs):
                errors.append(f"gate {i}: CNOT needs control != target in 0-{top}")
                continue
            architecture.append({"type": "CX", "c": c, "t": t})
        else:
            errors.append(f"gate {i}: g must be RX, RY, RZ, or CNOT")
    if errors:
        return None, errors
    try:
        verify_capacity(architecture, space)
    except ValueError as exc:
        return None, [str(exc)]
    return architecture, []


def arch_distance(a: list[dict], b: list[dict]) -> float:
    same = sum(1 for x, y in zip(a, b, strict=True) if x == y)
    return 1.0 - same / len(a)


def mean_pairwise_distance(archs: list[list[dict]]) -> float | None:
    pairs = [(i, j) for i in range(len(archs)) for j in range(i + 1, len(archs))]
    if not pairs:
        return None
    return float(np.mean([arch_distance(archs[i], archs[j]) for i, j in pairs]))


# ------------------------------------------------------------- physics -----


def _kron_numpy(mats: list[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for matrix in mats[1:]:
        out = np.kron(out, matrix)
    return out


def _two_site(op_a: np.ndarray, op_b: np.ndarray, i: int, n: int) -> np.ndarray:
    ident = np.eye(2)
    mats = [ident] * n
    mats[i] = op_a
    mats[i + 1] = op_b
    return _kron_numpy(mats)


def _one_site(op: np.ndarray, i: int, n: int) -> np.ndarray:
    ident = np.eye(2)
    mats = [ident] * n
    mats[i] = op
    return _kron_numpy(mats)


def hamiltonian(family: str, n: int, param: float) -> np.ndarray:
    """The (real, symmetric) Hamiltonian of `family` on an open n-site chain.

    TFIM  : H = -sum_i Z_i Z_{i+1} - h sum_i X_i          (param = h)
    XXZ   : H = sum_i (X_i X_{i+1} + Y_i Y_{i+1} + D Z_i Z_{i+1})   (param = D)
    """
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.diag([1, -1]).astype(complex)
    dim = 2**n
    out = np.zeros((dim, dim), dtype=complex)
    if family == "TFIM":
        for i in range(n - 1):
            out -= _two_site(z, z, i, n)
        for i in range(n):
            out -= param * _one_site(x, i, n)
    elif family == "XXZ":
        for i in range(n - 1):
            out += _two_site(x, x, i, n)
            out += _two_site(y, y, i, n)
            out += param * _two_site(z, z, i, n)
    else:
        raise ValueError(f"unknown Hamiltonian family {family}")
    return out


def ground_state(family: str, n: int, param: float) -> np.ndarray:
    """Normalised, phase-fixed ground state (identical convention to the
    reference `pilot.tfim_ground_state` at family=TFIM, n=4)."""
    _, vectors = np.linalg.eigh(hamiltonian(family, n, param))
    state = vectors[:, 0]
    pivot = int(np.argmax(np.abs(state)))
    state *= np.exp(-1j * np.angle(state[pivot]))
    return state


def ground_state_gap(family: str, n: int, param: float) -> float:
    values = np.linalg.eigvalsh(hamiltonian(family, n, param))
    return float(values[1] - values[0])


def state_sets(
    seed: int, family: str, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train/val/test ground-state families. The Hamiltonian-parameter draws
    are the reference streams, so the physical parameters are PAIRED across
    every condition; only n and the family change the resulting states."""
    rng = np.random.default_rng(10_000 + seed)
    train_p = np.sort(rng.uniform(PARAM_LOW, PARAM_HIGH, 32))
    val_p = np.sort(rng.uniform(PARAM_LOW, PARAM_HIGH, 12))
    test_p = np.linspace(0.215, 1.985, 64)
    train = np.stack([ground_state(family, n, float(p)) for p in train_p])
    val = np.stack([ground_state(family, n, float(p)) for p in val_p])
    test = np.stack([ground_state(family, n, float(p)) for p in test_p])
    return train, val, test


# ------------------------------------------------------------- trainer -----


def _rotation(axis: str, theta: torch.Tensor) -> torch.Tensor:
    pauli = {"X": X, "Y": Y, "Z": Z}[axis]
    return torch.cos(theta / 2) * I2 - 1j * torch.sin(theta / 2) * pauli


def _kron_torch(mats: list[torch.Tensor]) -> torch.Tensor:
    out = mats[0]
    for matrix in mats[1:]:
        out = torch.kron(out, matrix)
    return out


def _single_full(axis: str, theta: torch.Tensor, wire: int, n: int) -> torch.Tensor:
    mats = [I2 if q != wire else _rotation(axis, theta) for q in range(n)]
    return _kron_torch(mats)


def _cnot_full(control: int, target: int, n: int) -> torch.Tensor:
    dim = 2**n
    unitary = np.zeros((dim, dim), dtype=np.complex128)
    for index in range(dim):
        bits = [(index >> (n - 1 - q)) & 1 for q in range(n)]
        output = bits.copy()
        if bits[control]:
            output[target] ^= 1
        out_index = 0
        for bit in output:
            out_index = (out_index << 1) | bit
        unitary[out_index, index] = 1
    return torch.tensor(unitary, dtype=torch.complex128)


_CNOT_CACHE: dict[int, dict[tuple[int, int], torch.Tensor]] = {}
_TRASH_CACHE: dict[int, np.ndarray] = {}


def cnots_for(n: int) -> dict[tuple[int, int], torch.Tensor]:
    if n not in _CNOT_CACHE:
        _CNOT_CACHE[n] = {
            (c, t): _cnot_full(c, t, n)
            for c in range(n) for t in range(n) if c != t
        }
    return _CNOT_CACHE[n]


def trash_zero_indices(n: int) -> np.ndarray:
    """Computational-basis indices whose trash qubits are all |0>."""
    if n not in _TRASH_CACHE:
        trash = Space(n).trash
        mask = 0
        for q in trash:
            mask |= 1 << (n - 1 - q)
        _TRASH_CACHE[n] = np.array(
            [i for i in range(2**n) if (i & mask) == 0], dtype=int
        )
    return _TRASH_CACHE[n]


def evaluate_architecture(
    architecture: list[dict],
    train_states: np.ndarray,
    val_states: np.ndarray,
    test_states: np.ndarray,
    *,
    seed: int,
    n: int,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> EvalResult:
    """Frozen trainer: Adam(lr=0.05), 60 epochs, uniform(-0.05, 0.05) init,
    best-validation checkpoint, loss = 1 - mean trash fidelity. Bit-identical
    to `pilot.evaluate_architecture` at n=4."""
    space = Space(n)
    verify_capacity(architecture, space)
    torch.manual_seed(seed)
    parameters = torch.nn.Parameter(torch.empty(space.n_rotations, dtype=torch.float64))
    torch.nn.init.uniform_(parameters, -0.05, 0.05)
    optimizer = torch.optim.Adam([parameters], lr=learning_rate)

    train = torch.tensor(train_states, dtype=torch.complex128)
    val = torch.tensor(val_states, dtype=torch.complex128)
    test = torch.tensor(test_states, dtype=torch.complex128)
    cnots = cnots_for(n)
    keep = trash_zero_indices(n)

    def apply(states: torch.Tensor) -> torch.Tensor:
        output = states
        p = 0
        for operation in architecture:
            if operation["type"] == "R":
                unitary = _single_full(operation["axis"], parameters[p], operation["q"], n)
                p += 1
            else:
                unitary = cnots[(operation["c"], operation["t"])]
            output = output @ unitary.T
        return output

    def loss_and_fidelity(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = apply(states)
        fidelity = (output[:, keep].abs() ** 2).sum(dim=1).mean()
        return 1 - fidelity, fidelity

    best_val = float("inf")
    best_parameters: torch.Tensor | None = None
    for _ in range(epochs):
        optimizer.zero_grad()
        train_loss, _ = loss_and_fidelity(train)
        train_loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_loss, _ = loss_and_fidelity(val)
        if float(val_loss) < best_val:
            best_val = float(val_loss)
            best_parameters = parameters.detach().clone()

    assert best_parameters is not None
    with torch.no_grad():
        parameters.copy_(best_parameters)
        train_loss, train_fid = loss_and_fidelity(train)
        val_loss, val_fid = loss_and_fidelity(val)
        test_loss, test_fid = loss_and_fidelity(test)

    return EvalResult(
        train_loss=float(train_loss), val_loss=float(val_loss), test_loss=float(test_loss),
        train_fid=float(train_fid), val_fid=float(val_fid), test_fid=float(test_fid),
    )


def train_seed(seed: int, architecture: list[dict]) -> int:
    """Frozen per-architecture training seed (identical to `pilot._train_seed`)."""
    digest = hashlib.sha256(
        f"{seed}:{architecture_key(architecture)}".encode()
    ).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
