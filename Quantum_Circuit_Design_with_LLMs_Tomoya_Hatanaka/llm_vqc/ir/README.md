# Circuit IR (`llm_vqc.ir`)

**Phase:** 1 of the LAQS-Bench implementation roadmap
(`LLM-VQC_MASTER_PLAN.md`, Section 9). See `DECISIONS.md` for the
Phase 1 completion record and every consequential design decision made
while building this package.

## Purpose

Every future LAQS-Bench search arm — `random`, `evolutionary`, `greedy`,
`llm_iter`, `llm_evo` — must propose variational circuits through this
**one shared, typed intermediate representation**. That constraint is the
scientific core of the project (master plan Section 5.2): if the LLM arms
could emit free-form PennyLane code while the classical arms drew from a
different, narrower grammar, any performance difference between arms
would be confounded with unequal expressivity rather than a genuine
difference in search strategy. The IR removes that confound by
construction — every arm draws from, and is validated against, the exact
same grammar.

A secondary purpose: replacing "the agent fixes its own syntax errors
over 1-3 iterations" (a pattern Knipfer et al. 2026 report and that
silently eats into a budget-matched comparison) with deterministic,
structured, pre-compilation validation.

## Schema

Defined in `schema.py` as pydantic v2 models. Top level:

```python
CircuitIR(
    n_qubits: int,              # 2-10 inclusive
    encoding: EncodingSpec,     # how classical data enters the circuit
    layers: list[Layer],        # ordered rotation / entangle / repeat blocks
    measurements: MeasurementSpec,
    metadata: dict[str, str] = {},
)
```

**`EncodingSpec`** — `{type: "angle"|"amplitude", gate: RX|RY|RZ|None,
wires: WireSpec, reupload: int >= 0}`. `gate` is required for `"angle"`
and forbidden for `"amplitude"`. `reupload` must be `0` for `"amplitude"`
(Phase 1 limitation — see below).

**`RotationLayer`** — `{type: "rot", gates: list[RX|RY|RZ|H], wires:
WireSpec}`. Each gate in `gates` is applied, in order, to every wire in
`wires`.

**`EntangleLayer`** — `{type: "entangle", pattern, gate: CNOT|CZ|CRZ,
wires: WireSpec, center: int|None, pairs: list[[int,int]]|None}`.
`pattern` is one of `ring | line | star | all_to_all | pairs | none`; see
"Entangle patterns" below for exact edge sets. `center` is used (and
required) only by `star`; `pairs` only by `pairs`.

**`RepeatBlock`** — `{type: "repeat", times: int >= 1, body:
list[RotationLayer | EntangleLayer]}`. Repeats `body` `times` times, in
place, in the top-level `layers` sequence. **Nesting a RepeatBlock inside
another RepeatBlock's body is not supported** (kept out of scope to bound
complexity — a flat `times` count covers the vast majority of layered
ansatz structures without needing recursive repetition).

**`MeasurementSpec`** — `{observable: X|Y|Z, wires: WireSpec}`. The same
single observable type is applied to every wire in `wires` (per-wire
different observables is not supported — not needed by the master plan's
illustrative grammar, and adding it would be scope creep for Phase 1).

**`WireSpec`** — either the literal string `"all"` (every wire
`0..n_qubits-1`) or an explicit list of qubit indices, e.g. `[0, 2, 3]`.

### Gate sets

| Rotation gates | Parameterized? | Entangle gates | Parameterized? |
|---|---|---|---|
| RX, RY, RZ | yes (1 param/application) | CRZ | yes (1 param/application) |
| H | no | CNOT, CZ | no |

### Grammar extension beyond the master plan's illustrative example

The master plan's Section 5.2 illustrative IR example lists only RX/RY/RZ
as layer gates. Re-encoding the one published reference circuit with
sufficient detail to reproduce exactly (Knipfer et al. 2026's Claude
Simple-QNN best circuit — see `tests/fixtures/knipfer_circuits.py`)
required a Hadamard superposition-initialization step that is not
expressible as any RX/RY/RZ rotation. `H` was added to
`ROTATION_GATE_NAMES` as a fixed, non-parameterized gate to cover this.
This is additive (does not invalidate any previously-valid IR) and is
recorded as a Phase 1 design decision in `DECISIONS.md`. It is exactly
the kind of gap the master plan's own Phase 1 risk note anticipated
("grammar too narrow — mitigated by the fixture requirement").

### Wire-group interpretation

The master plan's illustrative example uses a bare string `"data"` as a
shorthand wire-group name (`"wires": "data"`) without ever defining a
named-group mini-language elsewhere in the document. Phase 1 interprets
"arbitrary gate subsets and wire groups" (master plan wording) as: `"all"`
plus explicit index lists, and nothing else. An explicit list like `[0,
1, 2, 3, 4]` covers the "data qubits" example by direct substitution and
is strictly more expressive (any subset, not just a fixed named
partition) and unambiguous. This is a low-level engineering decision left
open by the plan; recorded in `DECISIONS.md`.

### `reupload` semantics (Phase 1 interpretation)

`reupload=k` means the entire encoding block is applied `k + 1` times,
**consecutively, at the start of the circuit** — not interleaved with
variational layers. This is a deliberate simplification: true
interleaved data re-uploading (Pérez-Salinas et al. style, as referenced
in Knipfer et al.'s background section) requires the encoding step itself
to be splittable and re-insertable between arbitrary variational layers,
which is a materially larger grammar extension with no task/data
dependency yet to justify it in Phase 1 (data only exists from Phase 2
onward). **Known limitation:** circuits that specifically require
interleaved re-uploading (e.g. Knipfer et al.'s "recurrent-style" Full
Quantum QNN circuit, which re-uploads one new data point per variational
block across 21 sequential steps) cannot be expressed by this IR as-is.
That circuit was deliberately excluded from the fixture set for this
reason (see `tests/fixtures/knipfer_circuits.py`).

## Entangle patterns (exact edge sets)

Given resolved wires `w_0, ..., w_{m-1}` (in list order) and, for `star`,
a `center` wire `c`:

- **`none`** — no edges.
- **`pairs`** — exactly the proposer-supplied `(control, target)` list,
  verbatim, in order.
- **`ring`** — `(w_0,w_1), (w_1,w_2), ..., (w_{m-2},w_{m-1}), (w_{m-1},w_0)`
  — `m` edges including the wraparound. (A 2-wire ring therefore produces
  two distinct directed edges, `(w_0,w_1)` and `(w_1,w_0)` — not a
  degenerate no-op; two applications of a directed 2-qubit gate in
  opposite directions is not generally the identity.)
- **`line`** — the same as `ring` without the final wraparound edge —
  `m - 1` edges.
- **`star`** — `(c, w)` for every resolved wire `w != c` — `m - 1` edges.
- **`all_to_all`** — `(w_i, w_j)` for every `i < j` — `m(m-1)/2` edges.

All patterns except `pairs`/`none` require at least 2 resolved wires.

## Validation behavior

Two layers, deliberately separate:

1. **Schema validation** (`schema.py`, via pydantic) — field types, known
   gate/pattern names (`Literal` enums), required fields, per-field
   numeric bounds (`n_qubits` in [2,10], `RepeatBlock.times >= 1`, etc.),
   and `extra="forbid"` (unknown fields are rejected, not silently
   ignored — this catches typos and accidental extra keys from LLM
   output). This is what a raw dict must pass just to become a typed
   `CircuitIR` object at all.
2. **Semantic validation** (`validators.py`) — everything that needs
   cross-field context: qubit index bounds, duplicate wire entries,
   pattern-specific required/forbidden fields (`star` needs `center` and
   nothing else; `pairs` needs `pairs` and nothing else), self-loop
   detection, and circuit size limits (`MAX_TOP_LEVEL_LAYERS = 20`,
   `MAX_EXPANDED_GATE_APPLICATIONS = 500` — Phase 1 engineering defaults,
   not master-plan-mandated numbers).

**Both layers are reached through one entry point,
`validate_proposal(raw: dict | CircuitIR) -> ValidationResult`.** Unlike
a fail-fast validator, semantic validation collects *every* issue found
in one pass — a search algorithm or LLM agent proposing a circuit that
violates three separate rules gets told about all three at once, not one
at a time across three retries. This never repairs or reinterprets a
malformed proposal; it only reports, deterministically and structurally,
why a proposal was rejected.

```python
from llm_vqc.ir import validate_proposal

result = validate_proposal({
    "n_qubits": 3,
    "encoding": {"type": "angle", "gate": "RY"},
    "layers": [{"type": "entangle", "pattern": "star", "gate": "CNOT"}],  # missing 'center'
    "measurements": {"observable": "Z"},
})
result.valid   # False
result.issues  # [ValidationIssue(code="entangle.missing_center", path="layers[0]", message=...)]
result.ir      # None
```

## Serialization: canonical form and structural hash (`canonicalize.py`)

`canonical_json(ir)` produces a deterministic, sorted-key JSON string;
`structural_hash(ir)` is its SHA-256 hex digest.

**This is syntactic identity, not physical equivalence** — read the full
explanation in `canonicalize.py`'s module docstring before relying on it
for anything beyond duplicate-proposal detection. In short: an
`entangle` layer with `pattern="star", center=0` and an equivalent
explicit `pattern="pairs"` layer describing the same edges hash
*differently*, because detecting "these two circuits behave identically
for every possible parameter binding" is a fundamentally harder problem
than syntactic comparison and is out of scope for Phase 1. What
structural hashing *is* for: catching when a search arm — especially an
LLM — resubmits the exact same architecture, which is exactly the
"exploration collapse" failure mode Knipfer et al. report qualitatively
and this project measures quantitatively (master plan RQ2/H3, ablation
A2).

## Compiler responsibilities

Two backend compilers walk the *same* linearized instruction sequence
(`expand.py::build_program`), so they cannot silently diverge on gate
order or parameter assignment:

- **`compiler_pennylane.py`** — `to_qnode(ir) -> qml.QNode` callable as
  `qnode(inputs, weights)`, returning one expectation value per measured
  wire. Matches the `circuit(inputs, weights)` convention used throughout
  Knipfer et al.'s own tool docstrings.
- **`compiler_qiskit.py`** — `to_qiskit_symbolic(ir)` (unbound
  `Parameter` placeholders), `to_qiskit_bound(ir, inputs, weights)`
  (validated, fully-bound concrete circuit), and
  `qiskit_observables(ir)` (the measurement spec as `SparsePauliOp`s,
  kept separate from the state-prep circuit — Qiskit has no QNode-like
  object bundling "circuit + what to measure").

Stages are explicitly separated, matching the compiler pipeline the
Phase 1 task brief specifies: **validation** (`validators.py`) →
**linearization/normalization** (`expand.py`) → **backend compilation**
(`compiler_*.py`, structural circuit only) → **parameter binding**
(`to_qiskit_bound`'s `inputs`/`weights` arguments, validated for correct
length and finiteness) → *(transpilation/hardware mapping: out of Phase 1
scope — these functions return backend-native circuit objects, ready for
a future evaluation harness or hardware-constraint search arm to
transpile)* → *(execution/evaluation: Phase 2, not here)*.

## Cost metrics and budget accounting (Phase 1 scope: interfaces only)

`metrics.py::circuit_cost_summary(ir)` returns `n_qubits`, `depth` (via
Qiskit's own DAG-based `.depth()` — not reimplemented here),
`gate_count`, `two_qubit_gate_count`, `parameter_count`, `input_count`.
Purely structural — no evaluation metrics (loss, accuracy) live in this
package.

`budget.py::BudgetLedger` is an in-memory counter (proposed / valid /
invalid / unique / duplicate). It is **not** the full benchmarking system
— no persistence, no resume, not wired into `llm_vqc.runner` yet. That
wiring, plus checkpointing, is explicitly future-phase work (master plan
Section 11: "budget ledger checkpointed after every evaluation").

## Random sampler = the future `random` search arm

`sampler.py::sample_random_ir(rng: numpy.random.Generator) -> CircuitIR`
is not a testing convenience — per the master plan's Phase 1 deliverable
list, this function **is** the uninformed baseline's proposal generator.
It takes an explicit RNG (no hidden global state, so draws are
reproducible given a seed) and is guaranteed valid by construction (build
→ validate → resample on the rare invalid draw, verified empirically at
100% first-try validity over thousands of draws in
`tests/test_ir_sampler.py`).

## Examples

**Valid:**

```python
from llm_vqc.ir.schema import CircuitIR, EncodingSpec, RotationLayer, EntangleLayer, MeasurementSpec

ir = CircuitIR(
    n_qubits=4,
    encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
    layers=[
        RotationLayer(gates=["RY", "RZ"], wires="all"),
        EntangleLayer(pattern="star", gate="CNOT", center=0),
        RotationLayer(gates=["RX"], wires="all"),
    ],
    measurements=MeasurementSpec(observable="Z", wires="all"),
)
```

**Invalid** (multiple independent issues, all reported together):

```python
{
    "n_qubits": 3,
    "encoding": {"type": "angle", "wires": [0, 0, 9]},  # missing 'gate'; duplicate 0; 9 out of bounds
    "layers": [{"type": "entangle", "pattern": "star", "gate": "CNOT"}],  # missing 'center'
    "measurements": {"observable": "Z"},
}
# -> valid=False, issues = [encoding.missing_gate, wires.duplicate,
#                            wires.out_of_bounds, entangle.missing_center]
```

## What Phase 1 does not do

- No task/dataset loading, no training loop, no loss functions (Phase 2).
- No diagnostics — expressibility, entangling capability, gradient
  variance (Phase 3).
- No search drivers beyond the random sampler — no evolutionary,
  greedy, or LLM-guided proposal logic (Phase 4/5).
- No transpilation, hardware-topology constraints, or noise models.
- No persistence/resume for the budget ledger (interfaces only).
- No physical-equivalence-aware circuit deduplication (syntactic hashing
  only — see "Serialization" above).
- No interleaved data re-uploading (see `reupload` semantics above).
- No per-wire heterogeneous measurement observables.
- No nested `RepeatBlock`s.
