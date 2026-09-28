"""Meyer-Wallach entangling-capability diagnostic (Sim et al. 2019 Eq. 21;
Meyer & Wallach 2002; Brennen 2003 single-qubit-purity reformulation).

The Meyer-Wallach measure Q of a pure n-qubit state, in the numerically
stable Brennen form used here, is the average single-qubit linear entropy:

    Q(|psi>) = (2 / n) * sum_{k=1}^{n} ( 1 - Tr[ rho_k^2 ] )

where rho_k is the reduced density matrix of qubit k. This is provably
equal to Meyer & Wallach's original Q (Sim et al. Eq. 19; Brennen 2003)
and has the properties: Q in [0, 1]; Q = 0 iff |psi> is a product state;
Q(Bell) = Q(GHZ) = 1. It is invariant under single-qubit local unitaries
and under global phase.

The **entangling capability** of a parameterized circuit is the average Q
over sampled trainable-weight vectors (Sim et al. Eq. 21):

    Ent = (1 / |S|) sum_{theta in S} Q(|psi_theta>).

**Important interpretation caveat** (also in the README, per the task
brief's requirement not to call this "entanglement" unqualified): Q is a
single global pure-state measure of average single-qubit mixedness. It is
deliberately *undiscerning* about entanglement *structure* — Sim et al.
note that a 4-qubit GHZ state and a tensor product of two Bell pairs both
give Q = 1, even though they differ in Schmidt structure. Q measures
entangling capability in the specific averaged-single-qubit-mixedness
sense, not every notion of multipartite entanglement.

**One-qubit circuits:** a single qubit's reduced state is the whole
(pure) state, so 1 - Tr[rho^2] = 0 and Q = 0 for every one-qubit circuit
by construction (a single qubit cannot be entangled with anything). The
implementation returns 0 for n_qubits == 1 without sampling.

**Uncertainty:** two complementary measures are reported. Within one
replicate the Q values across the |S| samples are i.i.d., so the standard
error of the mean (std/sqrt(|S|)) is reported; across `n_replicates`
independent seed-sets, the between-replicate standard deviation of the
mean-Q is also reported (repeated-seed dispersion), consistent with the
expressibility diagnostic.
"""

from __future__ import annotations

import numpy as np

from llm_vqc.diagnostics.config import EntanglementConfig
from llm_vqc.diagnostics.sampling import diagnostic_input, sample_weight_vectors, statevector
from llm_vqc.diagnostics.seeds import DiagnosticSeeds
from llm_vqc.ir.expand import CircuitProgram, build_program
from llm_vqc.ir.schema import CircuitIR


def _single_qubit_purity(state: np.ndarray, qubit: int, n_qubits: int) -> float:
    """Tr[rho_k^2] for qubit `k`, from a full statevector (PennyLane
    big-endian index convention: qubit 0 is the most significant bit)."""
    tensor = state.reshape([2] * n_qubits)
    # Move the target qubit axis to the front, flatten the rest, and form
    # rho_k = sum over the environment of the outer product.
    moved = np.moveaxis(tensor, qubit, 0).reshape(2, -1)
    rho = moved @ moved.conj().T
    return float(np.real(np.trace(rho @ rho)))


def meyer_wallach(state: np.ndarray, n_qubits: int) -> float:
    """Q(|psi>) = (2/n) sum_k (1 - Tr[rho_k^2]). Returns a value in [0, 1]."""
    if n_qubits == 1:
        return 0.0
    total = 0.0
    for qubit in range(n_qubits):
        total += 1.0 - _single_qubit_purity(state, qubit, n_qubits)
    value = (2.0 / n_qubits) * total
    # Clip tiny numerical excursions outside [0, 1] from floating-point error.
    return float(min(1.0, max(0.0, value)))


class EntanglementOutcome:
    def __init__(
        self,
        ent_mean: float,
        ent_std_between_replicates: float,
        mean_standard_error: float,
        per_replicate: list[float],
        n_samples_completed: int,
    ) -> None:
        self.ent_mean = ent_mean
        self.ent_std_between_replicates = ent_std_between_replicates
        self.mean_standard_error = mean_standard_error
        self.per_replicate = per_replicate
        self.n_samples_completed = n_samples_completed


def _q_values_for_replicate(
    ir: CircuitIR,
    program: CircuitProgram,
    config: EntanglementConfig,
    seeds: DiagnosticSeeds,
) -> np.ndarray:
    inputs = diagnostic_input(program, seeds.diagnostic_input)
    vectors = sample_weight_vectors(
        program, config.n_samples, config.param_low, config.param_high, seeds.parameter_sampling
    )
    q_values = np.empty(config.n_samples, dtype=np.float64)
    for i, weights in enumerate(vectors):
        state = statevector(ir, inputs, weights, program)
        q_values[i] = meyer_wallach(state, ir.n_qubits)
    return q_values


def compute_entangling_capability(
    ir: CircuitIR,
    config: EntanglementConfig,
    replicate_seed_sets: list[DiagnosticSeeds],
    program: CircuitProgram | None = None,
) -> EntanglementOutcome:
    program = program if program is not None else build_program(ir)

    if ir.n_qubits == 1:
        # Every one-qubit state has Q = 0; no sampling needed.
        return EntanglementOutcome(
            ent_mean=0.0,
            ent_std_between_replicates=0.0,
            mean_standard_error=0.0,
            per_replicate=[0.0 for _ in replicate_seed_sets],
            n_samples_completed=0,
        )

    per_replicate_means: list[float] = []
    all_q: list[np.ndarray] = []
    for seeds in replicate_seed_sets:
        q_values = _q_values_for_replicate(ir, program, config, seeds)
        all_q.append(q_values)
        per_replicate_means.append(float(np.mean(q_values)))

    concatenated = np.concatenate(all_q)
    sem = float(np.std(concatenated) / np.sqrt(concatenated.size)) if concatenated.size else 0.0

    return EntanglementOutcome(
        ent_mean=float(np.mean(per_replicate_means)),
        ent_std_between_replicates=float(np.std(per_replicate_means)),
        mean_standard_error=sem,
        per_replicate=per_replicate_means,
        n_samples_completed=int(concatenated.size),
    )
