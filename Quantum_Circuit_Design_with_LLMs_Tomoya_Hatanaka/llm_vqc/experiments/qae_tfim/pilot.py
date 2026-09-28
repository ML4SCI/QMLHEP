"""Capacity-controlled semantic-prior quantum autoencoder pilot.

Task: compress 4-qubit open-chain transverse-field Ising ground states into
latent q0,q1 while trash q2,q3 approaches |00>.

All methods use exactly 12 trainable rotations + 4 CNOTs. The LLM candidate
pool is frozen in SEMANTIC_CANDIDATES; random controls share the same capacity.

This module intentionally contains no LLM/network call. The scientific artifact
is the fixed prompt plus the fixed proposal pool. A version-pinned API replay is
a separate confirmatory step.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, wilcoxon

N_QUBITS = 4
LATENT_QUBITS = (0, 1)
TRASH_QUBITS = (2, 3)
BUDGET = 8
VERIFY_SEEDS = tuple(range(12))
EPOCHS = 60
LEARNING_RATE = 0.05

I2 = torch.eye(2, dtype=torch.complex128)
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)


def _kron_numpy(mats: list[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for matrix in mats[1:]:
        out = np.kron(out, matrix)
    return out


def tfim_ground_state(h: float, j: float = 1.0) -> np.ndarray:
    """Return the normalized ground state of the 4-qubit open-chain TFIM."""
    ident = np.eye(2)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.diag([1, -1]).astype(complex)
    hamiltonian = np.zeros((16, 16), dtype=complex)
    for i in range(3):
        mats = [ident] * 4
        mats[i] = z
        mats[i + 1] = z
        hamiltonian -= j * _kron_numpy(mats)
    for i in range(4):
        mats = [ident] * 4
        mats[i] = x
        hamiltonian -= h * _kron_numpy(mats)
    _, vectors = np.linalg.eigh(hamiltonian)
    state = vectors[:, 0]
    pivot = int(np.argmax(np.abs(state)))
    state *= np.exp(-1j * np.angle(state[pivot]))
    return state


def _rotation(axis: str, theta: torch.Tensor) -> torch.Tensor:
    pauli = {"X": X, "Y": Y, "Z": Z}[axis]
    return torch.cos(theta / 2) * I2 - 1j * torch.sin(theta / 2) * pauli


def _kron_torch(mats: list[torch.Tensor]) -> torch.Tensor:
    out = mats[0]
    for matrix in mats[1:]:
        out = torch.kron(out, matrix)
    return out


def _single_full(axis: str, theta: torch.Tensor, wire: int) -> torch.Tensor:
    mats = [I2 if q != wire else _rotation(axis, theta) for q in range(4)]
    return _kron_torch(mats)


def _cnot_full(control: int, target: int) -> torch.Tensor:
    unitary = np.zeros((16, 16), dtype=np.complex128)
    for index in range(16):
        bits = [(index >> (3 - q)) & 1 for q in range(4)]
        output = bits.copy()
        if bits[control]:
            output[target] ^= 1
        out_index = 0
        for bit in output:
            out_index = (out_index << 1) | bit
        unitary[out_index, index] = 1
    return torch.tensor(unitary, dtype=torch.complex128)


CNOTS = {(c, t): _cnot_full(c, t) for c in range(4) for t in range(4) if c != t}
DIRECTED_PAIRS = tuple(CNOTS)


def make_arch(
    axes_layers: list[list[str]],
    cnots_after: dict[int, list[tuple[int, int]]],
) -> list[dict]:
    architecture: list[dict] = []
    for layer, axes in enumerate(axes_layers):
        for q, axis in enumerate(axes):
            architecture.append({"type": "R", "axis": axis, "q": q})
        for control, target in cnots_after.get(layer, []):
            architecture.append({"type": "CX", "c": control, "t": target})
    return architecture


def _mk(axes: list[list[str]], first: list[tuple[int, int]], second: list[tuple[int, int]]):
    return make_arch(axes, {0: first, 1: second})


SEMANTIC_CANDIDATES = {
    "L1_chain_funnel": _mk(
        [["Y"] * 4, ["Y"] * 4, ["Y"] * 4], [(3, 2), (2, 1)], [(1, 0), (2, 0)]
    ),
    "L2_pair_then_funnel": _mk(
        [["Y"] * 4, ["Y"] * 4, ["Y"] * 4], [(3, 2), (1, 0)], [(2, 1), (3, 1)]
    ),
    "L3_local_sweep_left": _mk(
        [["Y"] * 4, ["Y"] * 4, ["Y"] * 4], [(3, 2), (2, 1)], [(1, 0), (3, 1)]
    ),
    "L4_latent_targets": _mk(
        [["Y"] * 4, ["Y"] * 4, ["Y"] * 4], [(2, 0), (3, 1)], [(3, 2), (2, 1)]
    ),
    "L5_pair_disentangle": _mk(
        [["Y"] * 4, ["Y"] * 4, ["Y"] * 4], [(2, 3), (0, 1)], [(3, 1), (2, 0)]
    ),
    "L6_bidirectional_local": _mk(
        [["Y"] * 4, ["Y"] * 4, ["Y"] * 4], [(3, 2), (1, 2)], [(2, 1), (1, 0)]
    ),
    "L7_real_plus_phase_check": _mk(
        [["Y"] * 4, ["Z"] * 4, ["Y"] * 4], [(3, 2), (2, 1)], [(1, 0), (2, 0)]
    ),
    "L8_real_plus_x_check": _mk(
        [["Y"] * 4, ["X"] * 4, ["Y"] * 4], [(3, 2), (2, 1)], [(1, 0), (2, 0)]
    ),
}


def architecture_key(architecture: list[dict]) -> str:
    payload = json.dumps(architecture, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def verify_capacity(architecture: list[dict]) -> None:
    rotations = sum(op["type"] == "R" for op in architecture)
    cnots = sum(op["type"] == "CX" for op in architecture)
    if rotations != 12 or cnots != 4:
        raise ValueError(f"capacity mismatch: rotations={rotations}, CNOTs={cnots}")


def sample_random_arch(rng: np.random.Generator, ry_only: bool = False) -> list[dict]:
    if ry_only:
        axes = [["Y"] * 4 for _ in range(3)]
    else:
        axes = [[str(rng.choice(["X", "Y", "Z"])) for _ in range(4)] for _ in range(3)]
    selected = rng.choice(len(DIRECTED_PAIRS), size=4, replace=False)
    pairs = [DIRECTED_PAIRS[int(i)] for i in selected]
    architecture = _mk(axes, pairs[:2], pairs[2:])
    verify_capacity(architecture)
    return architecture


TRASH00_INDICES = np.array([i for i in range(16) if (i & 0b0011) == 0], dtype=int)


@dataclass
class EvalResult:
    train_loss: float
    val_loss: float
    test_loss: float
    train_fid: float
    val_fid: float
    test_fid: float


def _train_seed(seed: int, architecture: list[dict]) -> int:
    digest = hashlib.sha256(f"{seed}:{architecture_key(architecture)}".encode()).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def evaluate_architecture(
    architecture: list[dict],
    train_states: np.ndarray,
    val_states: np.ndarray,
    test_states: np.ndarray,
    *,
    seed: int,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> EvalResult:
    verify_capacity(architecture)
    torch.manual_seed(seed)
    parameters = torch.nn.Parameter(torch.empty(12, dtype=torch.float64))
    torch.nn.init.uniform_(parameters, -0.05, 0.05)
    optimizer = torch.optim.Adam([parameters], lr=learning_rate)

    train = torch.tensor(train_states, dtype=torch.complex128)
    val = torch.tensor(val_states, dtype=torch.complex128)
    test = torch.tensor(test_states, dtype=torch.complex128)

    def apply(states: torch.Tensor) -> torch.Tensor:
        output = states
        p = 0
        for operation in architecture:
            if operation["type"] == "R":
                unitary = _single_full(operation["axis"], parameters[p], operation["q"])
                p += 1
            else:
                unitary = CNOTS[(operation["c"], operation["t"])]
            output = output @ unitary.T
        return output

    def loss_and_fidelity(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = apply(states)
        fidelity = (output[:, TRASH00_INDICES].abs() ** 2).sum(dim=1).mean()
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
        train_fid=float(train_fid), val_fid=float(val_fid), test_fid=float(test_fid)
    )


def state_sets(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(10_000 + seed)
    train_h = np.sort(rng.uniform(0.2, 2.0, 32))
    val_h = np.sort(rng.uniform(0.2, 2.0, 12))
    test_h = np.linspace(0.215, 1.985, 64)
    train = np.stack([tfim_ground_state(float(h)) for h in train_h])
    val = np.stack([tfim_ground_state(float(h)) for h in val_h])
    test = np.stack([tfim_ground_state(float(h)) for h in test_h])
    return train, val, test


def run_seed(seed: int) -> list[dict]:
    train, val, test = state_sets(seed)
    rows: list[dict] = []
    for name, architecture in SEMANTIC_CANDIDATES.items():
        result = evaluate_architecture(architecture, train, val, test, seed=_train_seed(100 + seed, architecture))
        rows.append({"seed": seed, "method": "LLM-semantic", "candidate": name, **result.__dict__})

    rng = np.random.default_rng(20_000 + seed)
    for method, ry_only in (("Random", False), ("RY-random", True)):
        seen: set[str] = set()
        proposal = 0
        while proposal < BUDGET:
            architecture = sample_random_arch(rng, ry_only=ry_only)
            key = architecture_key(architecture)
            if key in seen:
                continue
            seen.add(key)
            result = evaluate_architecture(architecture, train, val, test, seed=_train_seed(100 + seed, architecture))
            proposal += 1
            rows.append({"seed": seed, "method": method, "candidate": f"R{proposal}", **result.__dict__})
    return rows


def analyze(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    selected = frame.sort_values("val_loss").groupby(["seed", "method"], as_index=False).first()
    pivot = selected.pivot(index="seed", columns="method", values="test_fid")
    stats: dict[str, dict] = {}
    for other in ("Random", "RY-random"):
        diff = pivot["LLM-semantic"] - pivot[other]
        exact = wilcoxon(diff, alternative="two-sided", method="exact")
        sign = binomtest(int((diff > 0).sum()), len(diff), 0.5, alternative="two-sided")
        stats[other] = {
            "mean_paired_fidelity_gain": float(diff.mean()),
            "median_paired_fidelity_gain": float(diff.median()),
            "llm_wins": int((diff > 0).sum()),
            "n": int(len(diff)),
            "wilcoxon_exact_two_sided_p": float(exact.pvalue),
            "sign_test_two_sided_p": float(sign.pvalue),
        }
    return selected, stats


def main(output_dir: str = "outputs/qae_tfim_semantic_pilot_v1") -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for seed in VERIFY_SEEDS:
        rows.extend(run_seed(seed))
    frame = pd.DataFrame(rows)
    selected, stats = analyze(frame)
    frame.to_csv(destination / "candidate_results.csv", index=False)
    selected.to_csv(destination / "selected_results.csv", index=False)
    (destination / "paired_stats.json").write_text(json.dumps(stats, indent=2))
    (destination / "semantic_candidate_pool.json").write_text(json.dumps(SEMANTIC_CANDIDATES, indent=2))
    print(selected.groupby("method")["test_fid"].agg(["mean", "median", "std"]))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
