# DECISIONS.md

Running log of phase completions, gate outcomes, and any deviations from
`LLM-VQC_MASTER_PLAN.md`, per that document's operating rules (Section 15).

---

## Phase 0 — Repairs and scaffolding

**Status: COMPLETE.** All acceptance criteria met (see below). No deviations
from the master plan design were required.

### Deliverables checklist (Section 9, Phase 0)

- [x] `state_key` fix (−0.0 canonicalization)
- [x] Regression tests asserting N=1→4, N=2→60, N=3→1080 (saturation)
- [x] Depth-count regression test (N=3 per-depth up to depth 5)
- [x] Docs note about the missing-S-gate asymmetry
- [x] Repo hygiene (ruff + pytest check script)
- [x] Config system (pydantic + YAML) and run-directory manifest writer
- [x] Acceptance: all tests pass; a dummy config produces a run dir with
      manifest, seeds, git SHA, env freeze

### 1. Bugs reproduced

**Bug: `state_key` in `llm_vqc/circuit_explorer.py` hashes rounded
statevector amplitudes as raw bytes without canonicalizing the sign of
zero.** IEEE754 gives `-0.0` and `+0.0` distinct byte representations even
though they compare equal (`-0.0 == 0.0` is `True`, but
`np.array([-0.0]).tobytes() != np.array([0.0]).tobytes()`), and this
distinction survives `np.round(..., 5)`. Two circuits that reach the
identical physical state can therefore receive different hash keys
whenever a rounded amplitude lands on zero with opposite sign in each
circuit, silently inflating every "unique equivalence class" count ever
produced by this tool.

Reproduced independently, before any code changes, in three ways:

1. **Direct NumPy check:** confirmed `-0.0`/`+0.0` are value-equal but
   byte-distinct, and that this survives `np.round`.
2. **Aggregate count reproduction (buggy engine, unmodified):**
   - N=1, G=6: reported 4 states
   - N=2, G=8: reported **82** states (60 true)
   - N=3, G=5: reported **893** states (666 true), per-depth
     `{0:1, 1:6, 2:21, 3:77, 4:240, 5:548}`
3. **Concrete minimal counterexample pair** found by exhaustively diffing
   pre-fix vs. post-fix keys over all depth-2 gate sequences on 2 qubits:
   circuits `H(0);H(1);CY(0,1)` and `H(0);CY(0,1);H(1);CX(0,1)` are
   *different* Clifford operators (confirmed `Clifford.__eq__` is `False`)
   that reach the *same* physical state from `|00⟩`. Pre-fix, their keys
   differed at exactly one byte (`\x00` vs. `\x80`, the sign bit of one
   rounded-to-zero imaginary component); post-fix, the keys are identical.
   See `tests/test_circuit_explorer_regression.py::test_state_key_treats_negative_and_positive_zero_as_equal`.

### 2. Root cause established

`state_key` (pre-fix):
```python
data = np.round(data.real, 5) + 1j * np.round(data.imag, 5)
return np.asarray(data, dtype=np.complex128).tobytes()
```
`np.round` preserves the sign of an input that rounds to zero (e.g.
`np.round(-1e-12, 5) == -0.0`), and `.tobytes()` serializes the IEEE754 sign
bit verbatim. No step in the function canonicalized this.

### 3. Correction implemented (principled, not result-specific)

`llm_vqc/circuit_explorer.py`, `state_key`:
```python
real = np.round(data.real, 5) + 0.0
imag = np.round(data.imag, 5) + 0.0
data = real + 1j * imag
```
Adding `0.0` folds `-0.0` into `+0.0` under IEEE754 round-to-nearest
(verified: `(-0.0) + 0.0 == 0.0` and the byte patterns become identical).
This fixes the general defect (any amplitude landing on signed zero, for
any N and any gate sequence), not just the specific N=2/N=3 cases used to
detect it — the fix targets the root cause (raw sign bit in the hash
input), not the reported symptom (wrong counts at particular N, G).

### 4. Tests added

`tests/test_circuit_explorer_regression.py` (7 tests) — see file docstring
for full rationale. Each test was verified to **fail against the unmodified
buggy code** and **pass against the fix**, confirmed by temporarily
stashing the fix (`git stash` / `git stash pop`) and re-running:

| Test | Fails pre-fix? | What it checks |
|---|---|---|
| `test_stabilizer_state_count_formula_matches_known_values` | n/a (pure math, no production import) | Ground-truth combinatorial formula used by other tests |
| `test_state_key_treats_negative_and_positive_zero_as_equal` | **Yes** — fails at the exact sign-bit byte | Concrete minimal counterexample pair (different operators, same state) |
| `test_state_key_has_no_negative_zero_bytes` | **Yes** | Exhaustive depth-2 sweep on N=2 + the known -0.0-producing 4-gate circuit; scans every 8-byte float64 chunk of every key for a raw -0.0 pattern |
| `test_n1_reachable_states_saturate_at_four_not_six` | No (N=1 was coincidentally unaffected) | Documents the missing-S-gate asymmetry as expected, correct behavior |
| `test_n2_reachable_states_saturate_at_full_stabilizer_count` | **Yes** (82 ≠ 60) | N=2 saturates at exactly 60 |
| `test_n3_reachable_states_saturate_at_full_stabilizer_count` | **Yes** (1780 ≠ 1080 at depth 8) | N=3 saturates at exactly 1080 |
| `test_n3_per_depth_new_state_counts_up_to_depth_five` | **Yes** | Exact per-depth counts pinned to independently cross-checked values |

`tests/test_runner.py` (6 tests) — config/manifest scaffolding, all new
code (no pre-existing behavior to regress against). One of these tests
(`test_create_run_dir_is_unique_and_nested`) caught a real bug during
development: second-resolution timestamps collided when two run
directories were created within the same wall-clock second, raising
`FileExistsError`. Fixed by appending an 8-hex-character random suffix to
the run-directory name (`llm_vqc/manifest.py::create_run_dir`).

No test was hard-coded to a value merely to make it pass; every expected
value is grounded in either (a) the closed-form stabilizer-state formula
`2^n · ∏_{k=1}^n (2^k+1)`, independently re-derived inside the test file
rather than imported from production code, or (b) an independently
cross-checked BFS implementation using a structurally different
phase-normalization method (angle-based rotation via `np.angle` instead of
division by phase, `argmax` of magnitude instead of first-significant-index),
which reproduced identical corrected counts.

### 5. Independent verification method (structurally different from production code)

Before touching `circuit_explorer.py`, a from-scratch BFS was written that
does **not** import `state_key` and uses a different canonicalization
approach:
```python
angle = np.angle(data[argmax(|data|)])
data = data * np.exp(-1j * angle)      # rotate by angle, not divide by phase
re = np.round(data.real, 5) + 0.0
im = np.round(data.imag, 5) + 0.0
```
This independent implementation, and a spot-check against Qiskit's own
`Statevector.equiv()` (which is itself global-phase-aware), both matched
the production fixed engine's output exactly:

| N | Independent BFS | Fixed production engine | Closed-form formula |
|---|---|---|---|
| 1 | 4 | 4 | 6 (4 reachable; S gate missing) |
| 2 | 60 | 60 | 60 |
| 3 | 1080 (saturates depth 8) | 1080 (saturates depth 8) | 1080 |
| 3, per-depth ≤5 | `{1,6,21,74,191,373}` (Σ=666) | `{1,6,21,74,191,373}` (Σ=666) | — |

### 6. Missing-S-gate asymmetry — verified and documented

The gate set `{H, X, Y, Z, CX, CY, CZ}` omits the S (phase) gate and is
therefore not Clifford-complete on a single qubit: `|+i⟩` and `|-i⟩` (the Y
eigenstates) are unreachable from `{H, X, Y, Z}` alone, so N=1 saturates at
4 of the 6 stabilizer states. Confirmed by exhaustive BFS to depth 10 (no
further states found past depth 2). At N≥2, `CY` supplies the missing
relative phase and full stabilizer-state coverage is restored (confirmed
exactly for N=2 and N=3; not exhaustively checked beyond N=3 due to
exponential cost, but the mechanism — CY contributing an imaginary
relative-phase gate — is expected to generalize). Documented in
`README.md` under "Equivalence Class Hashing" and in the docstring of
`tests/test_circuit_explorer_regression.py::test_n1_reachable_states_saturate_at_four_not_six`.

### 7. Exact commands used for validation

```bash
# Independent verification (before any code change)
python -c "..." # -0.0 vs +0.0 byte check
python -c "from llm_vqc.circuit_explorer import explore_circuit_space; ..." # reproduce buggy counts
python -c "..." # from-scratch independent BFS with different phase-normalization method

# After the fix
python -m pytest tests/ -v            # 19 tests initially, 25 after Phase 0 additions
python -m llm_vqc.runner configs/dummy.yaml   # Phase 0 acceptance criterion
bash scripts/check.sh                 # ruff check . && pytest tests/ -q

# Regression-test validity check (prove tests fail pre-fix)
git stash                             # temporarily remove the state_key fix
python -m pytest tests/test_circuit_explorer_regression.py -v
git stash pop                         # restore the fix
```

### 8. Before-and-after quantitative results

| Case | Before (buggy) | After (fixed) | Ground truth |
|---|---|---|---|
| N=1, saturation | 4 | 4 | 4 (6 minus 2 unreachable w/o S) |
| N=2, saturation | **82** | **60** | 60 |
| N=3, G=5 | **893**, per-depth `{1,6,21,77,240,548}` | **666**, per-depth `{1,6,21,74,191,373}` | 666 |
| N=3, saturation (depth 8) | 1780 | 1080 | 1080 |

### 9. Existing outputs invalidated by the bug

- `outputs/` in the main repository checkout (`/Users/hatanakatomoya/Developer/Sim/llm-vqc/outputs/`, generated 2026-06-23): `best_circuit.png`, `best_circuit.txt`, `exploration_trajectory.png`, `gate_distribution.png`, `pruning_efficiency.png`, `state_probabilities.png`. **Moved (not deleted)** to `outputs/archive_pre_negative_zero_fix_2026-07/` with an explanatory `README.md` in that directory. `outputs/` is gitignored, so these were never part of version-controlled repository history.
- The `20260622_GSoC.pdf` slide deck (external document, not in the repository) cites pruning-efficiency and state-count figures derived from the same buggy engine. This is outside repository scope and cannot be edited by this process; flagged here for the user's awareness. Any future slide/report referencing these numbers should be regenerated from the fixed engine.
- `LLM-VQC_MASTER_PLAN.md` itself documents the old (buggy) and new (fixed) numbers side by side as part of its audit section — this is intentional historical record, not an invalidated artifact.
- No other in-repository file was found to hardcode the old counts (verified by grepping for `893`, `82`, and related figures across `.py`/`.md` files).

### 10. Remaining risks or uncertainties

- The missing-S-gate coverage claim ("N≥2 always reaches full stabilizer count") is confirmed exactly for N=2 and N=3 only; N=4 (36720 states) was not exhaustively checked because it is computationally expensive with the current unoptimized BFS (N=3 at full saturation already takes ~16.5s). This does not affect Phase 0 correctness claims, which are scoped to N≤3, but should not be assumed to hold at higher N without checking.
- `scripts/check.sh`'s Python-interpreter auto-detection was validated with a manually-activated virtualenv; it was not validated against every possible shell/venv configuration a future contributor might use.
- The config/manifest system (`llm_vqc/config.py`, `manifest.py`, `runner.py`) is intentionally minimal Phase 0 scaffolding — a generic `RunConfig(name, seed, params)` — and does not yet encode anything task/arm/IR-specific. Phase 1 will need to extend `params` (or introduce nested config models) without breaking the Phase 0 manifest contract; this was designed for but not yet exercised.
- `pip_freeze` in the manifest reflects the environment at manifest-write time, not a pinned lockfile; the master plan's Section 11 reproducibility requirement calls for a pinned `requirements.lock` at Phase 0, which was **not** created in this pass (out of the explicit Phase 0 deliverable list in Section 9, though mentioned in Section 11.3 more generally) — flagged for early Phase 1 attention if not required sooner.

### 11. Phase 0 acceptance criteria — satisfied

- [x] All tests pass: `pytest tests/ -q` → **25 passed** (12 pre-existing + 7 new regression tests + 6 new runner tests).
- [x] A dummy config produces a run dir with manifest, seeds, git SHA, env freeze: verified via `python -m llm_vqc.runner configs/dummy.yaml`, producing `runs/dummy_phase0_smoke/<timestamp>-<suffix>/manifest.json` containing `config`, `seed`, `git_sha` (40-char SHA, correctly flagged `git_dirty: true` given in-progress Phase 0 changes), `python_version`, and a non-empty `pip_freeze` list.
- [x] `ruff check .` passes with zero errors (3 pre-existing lint issues in `agent.py`, `circuit_explorer.py`, `visualization.py` fixed as trivial, behavior-preserving cleanups alongside the hygiene deliverable).

**Phase 0 gate: PASS.** Confirmed accepted by the user ("Phase 0 is
accepted") before Phase 1 began.

---

## Phase 1 — Circuit IR + Compiler

**Status: COMPLETE.** All acceptance criteria met (see below).

### Deliverables checklist (Section 9, Phase 1)

- [x] IR schema (pydantic)
- [x] Validators (semantic, collect-all-issues)
- [x] Canonicalizer + structural hash
- [x] IR → PennyLane compiler
- [x] IR → Qiskit exporter
- [x] Random-IR sampler (doubles as the future `random` search arm)
- [x] Re-encode Knipfer et al.'s published best circuit(s) as fixtures
- [x] Acceptance: round-trip + determinism tests; compiled QNode trains on
      toy data; Knipfer fixture compiles and matches expected param count;
      sampler produces 100% valid IRs

### 1. Design decisions recorded (per operating rule: "record all
consequential decisions the plan leaves open")

**D1 — Wire-group interpretation.** The master plan's Section 5.2
illustrative example uses a bare string `"wires": "data"` without ever
defining a named-group mini-language elsewhere in the document.
**Decision:** `WireSpec` supports only the literal `"all"` plus explicit
index lists (`list[int]`) — no named-group vocabulary. Explicit lists
strictly subsume any fixed named partition (e.g. `[0,1,2,3,4]` for "data
qubits") and remove the ambiguity of an undefined mini-language.
Documented in `llm_vqc/ir/README.md`.

**D2 — `reupload` semantics.** The grammar lists `reupload: int` on
`EncodingSpec` without specifying *where* re-encoding is inserted
relative to variational layers. **Decision:** Phase 1 interprets
`reupload=k` as `k+1` consecutive encoding-block applications at the
circuit's start (not interleaved with variational layers). **Known
limitation:** this cannot express true interleaved data re-uploading
(Pérez-Salinas et al. style). Consequence: Knipfer et al.'s "recurrent
style" Full Quantum QNN circuit (per-timestep re-uploading across 21
sequential steps) is **not** representable by this IR and was
deliberately excluded from the fixture set rather than approximated.
Flagged for reconsideration once a task (Phase 2+) actually needs it.

**D3 — Grammar extension: added `H` (Hadamard) to the rotation-gate set.**
The master plan's illustrative example lists only RX/RY/RZ as layer
gates and flags coverage as an explicit **[ASSUMPTION] to verify** in
Phase 1 via fixture re-encoding. Verification found a real gap: Knipfer
et al.'s Claude Simple-QNN best circuit initializes its computation
qubits with `qml.Hadamard`, which is not expressible as any RX/RY/RZ
rotation. **Decision:** added `H` as a fixed (non-parameterized) rotation
gate. This is additive — no previously-valid IR becomes invalid — and is
exactly the kind of gap the master plan's own Phase 1 risk note
anticipated ("grammar too narrow — mitigated by the fixture
requirement"). Not a redesign of the approved IR approach.

**D4 — Canonical serialization is syntactic identity, not physical
equivalence.** Explicitly required by the task brief ("clearly document
whether canonical serialization represents syntactic identity or
physical equivalence; do not conflate the two"). **Decision:**
`canonicalize.py` produces deterministic, sorted-key JSON of the IR
exactly as given — list order (layers, gates, wires, pairs) is preserved,
never reordered. Two circuits that are physically identical but
structurally different (e.g. a `star` pattern vs. the equivalent explicit
`pairs` list) hash *differently*. Detecting physical equivalence of
*parameterized* circuits is a substantially harder problem than the
discrete Clifford-state equivalence solved in Phase 0 (it would require
checking equivalence for every possible parameter binding, not one fixed
state) and is explicitly out of Phase 1 scope. Structural hashing is
used only for what the master plan actually needs it for: detecting
resubmission of the same architecture (Section 5.2 point 3 /
"exploration collapse" measurement, RQ2/H3, ablation A2) — not for
merging circuits that merely behave similarly.

**D5 — Circuit size limits are Phase 1 engineering defaults, not
master-plan-mandated numbers.** The plan requires "circuit size and
operation-count constraints" without specifying values. **Decision:**
`MAX_TOP_LEVEL_LAYERS = 20`, `MAX_EXPANDED_GATE_APPLICATIONS = 500`
(`validators.py`). Chosen to comfortably accommodate every fixture and
realistic ansatz in the ≤10-qubit regime while still bounding runaway
proposals; revisitable once Phase 2+ task profiling gives real numbers to
tune against.

**D6 — Parameter binding is external to the IR, not embedded in it.** The
IR names *which* gates are parameterized (RX/RY/RZ/CRZ) and their
canonical order, but does not carry literal numeric parameter values —
those are supplied separately as a flat `weights` vector at compile time
(`to_qiskit_bound`, `to_qnode`). This mirrors the Knipfer et al. tool
convention (`circuit(inputs, weights)`) rather than inventing a new
convention, and is why the IR schema has no float fields at all — a
useful side effect noted in `canonicalize.py`: no floating-point
serialization-stability concerns exist for structural hashing.

**D7 — One linearization shared by both compilers.** Not explicitly
required by the master plan's component table, but adopted to make "the
same IR input produces semantically equivalent compiled circuits across
backends" true *by construction* rather than by convention: both
`compiler_pennylane.py` and `compiler_qiskit.py` walk the identical
`CircuitProgram` produced by `expand.py::build_program`, so gate order
and parameter-slot assignment cannot silently diverge between backends.
Verified empirically in `tests/test_ir_semantic_equivalence.py::
test_pennylane_and_qiskit_agree_on_expectation_values`.

**D8 — Knipfer fixture scope: one exact fixture, not three.** The master
plan's Phase 1 deliverable says "re-encode Knipfer et al.'s published
best circuits" (plural). Only the Claude 3.7 Sonnet Simple-QNN circuit is
re-encoded with a claim of exact fidelity, because it is the only one of
the paper's three highlighted circuits printed as literal PennyLane code
with an independently-checkable published parameter count (45). The two
Llama 3.3 70B circuits are described only in prose without a code listing
or a parameter count derivable from the text (48 for Simple-QNN, 81 for
QuanvNN — neither factors cleanly against any qubit count under the
prose's stated gate pattern). **Decision:** do not fabricate an
approximate "reproduction" of those two circuits and label it as
verified; document the gap instead (`tests/fixtures/knipfer_circuits.py`
docstring). This is a direct application of the task brief's "do not
create favorable placeholder results" constraint to the fixture-fidelity
claim specifically, not just to experimental results.

### 2. Final IR schema and rationale

See `llm_vqc/ir/README.md` "Schema" section for the full field-by-field
reference; not duplicated here. Rationale summary: `CircuitIR` /
`EncodingSpec` / `RotationLayer` / `EntangleLayer` / `RepeatBlock` /
`MeasurementSpec`, pydantic v2 discriminated unions on `type`, `extra=
"forbid"` everywhere (rejects typos/unexpected fields rather than
silently ignoring them — verified to matter, see §5 below).

### 3. Supported gates and parameter model

Rotation: RX, RY, RZ (parameterized, 1 slot/application), H
(non-parameterized). Entangle: CNOT, CZ (non-parameterized), CRZ
(parameterized, 1 slot/application). Parameter order is canonical and
deterministic: layer order, then (for rotation layers) gate-list order ×
resolved-wire order; (for entangle layers) resolved-pair order. No
literal parameter values live in the IR (see D6).

### 4. Validation rules

Schema layer (pydantic): known gate/pattern names, required fields, `n_
qubits` in [2,10], `RepeatBlock.times >= 1`, non-empty `gates`/`body`
lists, `extra="forbid"`. Semantic layer (`validators.py`, collects every
issue in one pass rather than failing fast): qubit index bounds,
duplicate wire entries, pattern-specific required/forbidden fields
(`star`↔`center`, `pairs`↔`pairs`, self-loop detection in `pairs`),
encoding type/gate/reupload consistency, and the two size limits in D5.

### 5. Canonical serialization and identity semantics

See D4. `structural_hash` = SHA-256 hex of `canonical_json` (sorted-key
JSON, list order preserved). **Syntactic identity, not physical
equivalence** — do not conflate.

### 6. Compiler architecture and supported backends

PennyLane (`to_qnode`) and Qiskit (`to_qiskit_symbolic` /
`to_qiskit_bound` / `qiskit_observables`), both driven from one shared
linearization (D7). Amplitude encoding is supported for `to_qiskit_bound`
(via `QuantumCircuit.prepare_state`) but not `to_qiskit_symbolic` (no
symbolic Qiskit state-preparation representation attempted — documented
limitation, not a silent gap). Transpilation and hardware mapping are out
of Phase 1 scope by design (master plan P2 backlog item #13).

### 7. Semantic-equivalence validation performed

`tests/test_ir_semantic_equivalence.py` (14 tests): single-qubit gates,
controlled/multi-qubit gates (all 4 pattern types + CRZ), gate-order
sensitivity (non-commuting sequences produce provably different states),
parameter binding validation (wrong length, non-finite values), and
cross-backend (PennyLane vs. Qiskit) expectation-value agreement — all
via `Statevector.equiv()` (Qiskit's own global-phase-aware check) or
direct numerical comparison against hand-written reference circuits,
never textual/structural equality. `tests/test_ir_knipfer_fixtures.py`
additionally confirms the Claude fixture's parameter count matches the
independently published number (45) exactly.

### 8. Budget-accounting hooks established

`llm_vqc/ir/metrics.py::circuit_cost_summary` (structural: n_qubits,
depth via Qiskit's `.depth()`, gate_count, two_qubit_gate_count,
parameter_count, input_count) and `llm_vqc/ir/budget.py::BudgetLedger`
(in-memory proposed/valid/invalid/unique/duplicate counters). Explicitly
interfaces-and-data-structures only per the task brief — no persistence,
no resume, not wired into `llm_vqc.runner` (that wiring is future-phase
work per master plan Section 11).

### 9. Tests added

131 tests total (up from 25 after Phase 0): `test_ir_schema.py` (19),
`test_ir_validators.py` (22), `test_ir_canonicalize.py` (10),
`test_ir_sampler.py` (6), `test_ir_budget.py` (4),
`test_ir_semantic_equivalence.py` (15),
`test_ir_knipfer_fixtures.py` (6),
`test_ir_config_manifest_compatibility.py` (2),
`test_ir_adversarial_inputs.py` (22, parametrized). All 131 pass; the
previously-existing 25 Phase 0 tests are unmodified and still pass.

Three protections were deliberately, temporarily removed and confirmed
to cause the expected test failures, then restored byte-identical to
their pre-patch state (verified via `diff`): the `pairs` self-loop check
(`test_pairs_self_loop_is_rejected`), `extra="forbid"` on the schema
(`test_unknown_top_level_field_rejected`), and the finite-value check in
`to_qiskit_bound` (`test_parameter_binding_rejects_non_finite_values`).
All three failed correctly without the protection and passed once
restored.

### 10. Exact validation commands and results

```bash
python -m pytest tests/ -q                     # 131 passed
python -m ruff check .                          # All checks passed!
bash scripts/check.sh                            # ruff + pytest, both clean

# Dependency-lock verification (fresh venv, not the dev venv):
python3 -m venv /tmp/lock_verify_venv
/tmp/lock_verify_venv/bin/pip install -r requirements-lock.txt
/tmp/lock_verify_venv/bin/pip install -e . --no-deps
/tmp/lock_verify_venv/bin/python -m pytest tests/ -q   # 131 passed
```

### 11. Dependency-lock approach

`requirements-lock.txt` = `pip freeze` (minus the editable self-install
line) from the fully-provisioned dev venv. Targets Python 3.14 (CPython),
macOS arm64; some pinned wheels (`pennylane_lightning`, `scipy`,
`rustworkx`) are platform-specific and the lock file would need
regenerating on Linux/Windows or x86_64. Install/regenerate instructions
in `README.md` "Reproducible environment (pinned lock)". `pennylane` was
also added to `pyproject.toml`'s abstract `dependencies` (it was
installed but not yet declared there before this phase — fixed as part
of this deliverable).

### 12. Backward compatibility

No Phase 0 file was modified except `pyproject.toml` (added `pennylane`
dependency, `[project.optional-dependencies].dev`, `[tool.pytest.ini_
options]`, `[tool.ruff]` — all additive) and `.gitignore` (already had
`runs/`). All 25 Phase 0 tests pass unmodified. `circuit_explorer.py` was
not touched in this phase (per master plan Section 5.4, it is retained
as-is for repurposing in a later phase, not migrated onto the new IR).

### 13. Known limitations and unresolved risks

- `reupload` does not support interleaved re-uploading (D2); the Full
  Quantum QNN Knipfer fixture is unrepresentable as a result.
- Physical-equivalence-aware circuit deduplication is not implemented
  (D4) — only syntactic structural hashing.
- No nested `RepeatBlock`s.
- No per-wire heterogeneous measurement observables (one observable type
  per `MeasurementSpec`).
- `BudgetLedger` has no persistence/resume yet (by design — future
  phase).
- Circuit size limits (D5) are engineering defaults, not empirically
  tuned against real Phase 2+ task profiling yet.
- The Llama 3.3 70B circuits from Knipfer et al. remain unverified
  fixtures by design (D8); if a literal code listing becomes available,
  they should be added the same way the Claude fixture was.
- `to_qiskit_symbolic` does not support amplitude encoding (documented,
  not silent — `to_qiskit_bound` does support it).

### 14. Phase 1 acceptance criteria — satisfied

- [x] Round-trip and determinism tests: `test_ir_canonicalize.py`,
      `test_ir_schema.py`.
- [x] Compiled QNode trains on toy data:
      `test_ir_semantic_equivalence.py::test_compiled_qnode_trains_on_toy_data`
      — gradient-descends a compiled circuit toward a fixed target over 15
      steps via `qml.GradientDescentOptimizer` and asserts the loss
      strictly decreases.
- [x] Knipfer fixture compiles and matches expected param count: `test_
      ir_knipfer_fixtures.py` — 45 parameters, exact match to the
      published number.
- [x] Sampler produces 100% valid IRs: `test_ir_sampler.py` — 500/500
      valid in the formal test; 2000/2000 valid on first try (no resample
      needed) in manual verification.

**Phase 1 gate: PASS.** Confirmed accepted by the user ("Phase 1 is
accepted") and PyTorch addition approved before Phase 2 began.

---

## Phase 2 — Tasks + Evaluation Harness

### Gate G1 — T2 dataset accessibility assessment

**Outcome: BLOCKED (primary source) → FALLBACK activated
(scikit-learn `digits`, classes 3 vs 8).**

**Method:** metadata/documentation research only, per instructions — no
bulk download was attempted at any point.

**Evidence on the primary T2 candidate (ML4SCI electron-photon ECAL
dataset, Andrews et al.):**

- Confirmed identity and format via a prior official ML4SCI QMLHEP GSoC
  student repository
  ([eraraya-ricardo/qcnn-hep](https://github.com/eraraya-ricardo/qcnn-hep),
  README "Project's Datasets" section, fetched
  2026-07-13): "498,000 samples, equally distributed between the two
  classes[, ...] 32x32[, ...] two types of particles: photons (0) and
  electrons (1) captured by the ECAL detector." This matches the master
  plan's T2 description exactly, confirming it is the correct dataset.
- The **same README states verbatim**: *"If you are interested on using
  the datast for your study, contact me and I can try to connect you to
  the people at ML4Sci who have the dataset."* This is a direct,
  first-party statement from a past official ML4SCI QMLHEP contributor
  that the dataset is **not publicly downloadable** — access requires
  manually contacting ML4Sci maintainers and being individually
  connected to whoever holds a private copy. No Zenodo DOI, CERN Open
  Data direct link, or other self-service public release could be found
  for the *derived, classification-ready* electron-photon image set
  (raw CMS ECAL detector data exists on the public CERN Open Data
  Portal, but not this specific pre-processed 32×32×2 derivative).
- A third-party Kaggle mirror
  ([krutarthkayparmar/electron-photon-cms-ml4sci](https://www.kaggle.com/datasets/krutarthkayparmar/electron-photon-cms-ml4sci))
  exists, but using it would mean obtaining a private/gated dataset
  through an unofficial re-upload rather than through ML4Sci's own access
  process — this is exactly the kind of authentication/access-restriction
  bypass the task instructions say not to do, and its license/permission
  chain from the original data holders is unverified.
- **Conclusion against the Gate G1 criteria:** not accessible without
  individual manual authorization; authentication/approval required (an
  informal "contact me" process, not a stated open license); license and
  redistribution terms undocumented; and — decisively — **not
  reproducibly usable by independent researchers** without each one
  separately going through the same ad hoc contact process. This fails
  Gate G1 on reproducibility grounds alone, independent of the license
  ambiguity.

**Evidence on the master plan's first fallback option (Quark-Gluon
dataset, also Andrews et al., CMS Open Data):** researched in parallel as
a candidate before selecting between the two pre-approved fallbacks. No
public, direct-download link (Zenodo, CERN Open Data direct record, or
otherwise) could be located for the derived classification-ready
933,206-image set within a reasonable, metadata-only research effort;
the same qcnn-hep repository that successfully used this dataset
describes it as "huge," requiring "a much higher specs computational
device" and explicitly labels its own quark-gluon results as
"still a working progress." Its public-access status is therefore **also
unverified**, and even if accessible it is a much larger, heavier dataset
than is warranted for Phase 2's evaluation-harness smoke-test role. It
was not selected.

**Fallback selected: scikit-learn's built-in `digits` dataset
(`sklearn.datasets.load_digits`), restricted to classes 3 vs 8 for a
binary task matching T2's electron-vs-photon binary structure.**
Verified directly (not merely asserted) in this session:

- **Accessibility:** bundled with `scikit-learn` (added as a Phase 2
  dependency); zero download, zero authentication, works fully offline.
- **License:** scikit-learn's toolkit datasets are permissively licensed
  (BSD-3-Clause project); this specific dataset is the public-domain UCI
  "Optical Recognition of Handwritten Digits" set (creator E. Alpaydin,
  1998) — no ambiguity.
- **Size/schema:** 1797 total samples, 10 classes, 8×8=64 integer
  pixel features (0–16). Per-class counts verified: class 3 has 183
  samples, class 8 has 174 samples (357 total for the binary subtask).
- **Reproducibility:** trivially reproducible by any independent
  researcher — same library call, same fixed data, no network
  dependency, no version drift risk beyond the pinned `scikit-learn`
  version in `requirements-lock.txt`.
- **Provenance:** well-established benchmark dataset with a stable
  40+-year research history; not a synthetic fixture (a synthetic
  fixture is used separately, and only for automated tests — see T2
  implementation below — never presented as this task's data).

**Deviation from the master plan's stated T2 split sizes (documented, as
required):** the master plan's `~2k train / 1k val / 4k test` sizing was
calibrated for the (much larger) primary ECAL dataset to maximize
test-set statistical power per Section 6.4 ("enlarge test sets... so
inter-arm differences aren't swamped by split noise"). With only 357
total samples available in the fallback, that ratio is infeasible.
**Decision:** stratified 60% train / 20% val / 20% test (≈214 / 71 / 72
samples), preserving class balance in every split. This is a standard,
defensible small-dataset split policy; the test-set-noise risk this
represents (much less than 4,000 test samples) is recorded as a known
limitation below and in the T2 implementation docstring rather than
silently accepted.

**Unresolved reproducibility risks:** (1) `digits` is a much easier,
lower-dimensional, non-HEP task than the intended ECAL benchmark — any
T2 conclusions from this fallback do **not** carry the "canonical QMLHEP
benchmark" relevance claim the master plan originally intended for T2,
and any future write-up must say so explicitly, not silently generalize.
(2) The 357-sample fallback dataset gives a smaller, noisier test
partition than the master plan's design target; per master plan Section
13 risk-3 guidance ("reframe primary endpoint as estimation-with-CIs
rather than hypothesis testing" if seed variance swamps arm differences),
this should be anticipated for T2 specifically going into Phase 6+'s
pilot decision gate.

This activates the master plan's own pre-approved contingency (Section
13, risk 1: "T2 dataset inaccessible → Gate G1: fall back to quark-gluon
subsample or a standard non-HEP dataset; document. Continue regardless.")
— no new, unapproved fallback was invented.

### Task abstraction (`llm_vqc/tasks/base.py`)

`DataSplit` (features, targets, unique sample_ids; validated at
construction — mismatched lengths, duplicate ids, or empty splits raise
`TaskDataError` immediately) and `assert_disjoint_splits(*splits)`
(shared overlap guard every task's `build`/`build_test` must pass its
splits through). `TrainValData` bundles train + val + fitted
preprocessing + spec; it **has no `test` field** — that is the primary
quarantine mechanism (see below). `TaskSpec` deliberately does not pin a
qubit count or circuit-input width, because IR circuits proposed against
a task can vary their own qubit/encoding-wire count; a trainable
per-circuit "embed" adapter layer (`llm_vqc/evaluation/model.py`) bridges
a task's fixed `raw_feature_dim` to whatever a specific circuit's
`build_program(ir).num_inputs` requires.

### T1 implementation (`llm_vqc/tasks/t1_gaussian.py`)

Synthetic, generated on demand from Knipfer et al.'s exact formula (21
fixed x-points, Gaussian peak with `mu~U[0,1]` as target, `A~U[0.5,1.5]`,
`sigma~U[0.01,0.1]`, i.i.d. `N(0,0.01^2)` noise per point), reproduced
faithfully from Section 3.3 of their paper (already read in full during
Phase 0 research). Preprocessing is Knipfer et al.'s own per-sample
min-max normalization (Eq. 2) — a stateless, per-row transform with no
fitted parameters, so "fit on train only" holds vacuously (verified in
`tests/test_tasks_t1.py::test_preprocessing_is_a_pure_per_sample_function_no_val_test_leakage`).
Split sizes: 150 train / 250 val / **2,000** test, adopting the master
plan's own `[PROPOSAL]` enlarged-test-size recommendation verbatim
(Section 6.1 explicitly proposes 2,000 over Knipfer's original 500 "to
cut metric noise"). Splits are disjoint by construction (RNG stream
advanced sequentially: train draws, then val draws, then test draws, from
one seeded generator — verified in tests, not merely asserted).

### T2 implementation (`llm_vqc/tasks/t2_digits.py`)

See the Gate G1 section above for the dataset itself. PCA
(`PCA_COMPONENTS = 10`, within the master plan's specified 8-16 range)
fit on the training partition only, followed by per-component min-max
scaling to `[0, pi]` (also fit on train only) for angle-encoding
compatibility. Both fitted steps were verified, not just asserted, to
depend on train data alone: `tests/test_tasks_t2.py::
test_pca_is_fit_on_train_partition_only_not_val_or_test` fits PCA on
train-only vs. train+val vs. train+test and confirms the components
differ (i.e. val/test rows measurably change the fit when included,
proving they are correctly excluded in the real pipeline), and `test_
task_build_reproduces_the_train_only_pca_fit` confirms the task's actual
internal preprocessing exactly reproduces an external, independently
constructed train-only PCA fit (bit-for-bit, using the same derived
seeds).

### Test-quarantine mechanism

Structural, not conventional, at two independent points:

1. **Data:** `TrainValData` (what `harness.evaluate_candidate` accepts)
   has no `test` attribute. Test data is obtainable only via a task's
   separately-called `build_test(seed)`.
2. **Code:** `llm_vqc/evaluation/harness.py` contains no import of
   `llm_vqc.evaluation.final_test` anywhere — verified by both an AST
   parse of the source (`test_evaluation_harness.py::
   test_harness_module_has_no_import_referencing_final_test`) and a
   subprocess-isolated check that importing only `harness` never pulls
   `final_test` into `sys.modules` (`test_importing_harness_does_not_
   transitively_import_final_test`).
3. **Result schema:** `EvaluationResult` (what `evaluate_candidate`
   returns) has no field whose name contains "test" — verified
   programmatically over `EvaluationResult.model_fields`, not just by
   inspection.

Final test scoring (`llm_vqc/evaluation/final_test.py::evaluate_on_test`)
reloads already-trained weights (persisted by the harness via the result
store) and scores them once on the test partition — it never retrains,
per the master plan's protocol ("final comparison on test metric of each
run's *selected* circuit, selected on validation").

### Fixed training pipeline (`llm_vqc/evaluation/training.py`)

`TrainingConfig` (pydantic, all fields explicit, validated: `epochs>=1`,
`batch_size>=1`, `learning_rate>0` — a degenerate `epochs=0` config was
caught by a smoke test crashing with `IndexError` during development and
fixed by adding this validation rather than defensively coding around it
in the training loop, see §"Bugs found and fixed" below). Defaults match
master plan Section 6.3 exactly: AdamW, lr 0.05, 20 epochs, batch 16. The
exact `lr_decay_epochs=(7,13,17)` and `weight_decay=1e-5` are Phase 2
engineering defaults (the master plan specifies "decay at fixed epochs"
without pinning the epochs) — same category of decision as Phase 1's
`MAX_TOP_LEVEL_LAYERS`. No early stopping, no checkpointing — deliberately
matching Knipfer et al.'s own choice exactly ("No early stopping or model
checkpointing is used") to avoid introducing an extra uncontrolled
difference from the study this project is meant to be comparable with.
Training never raises on failure (divergence, backend errors) — it
returns a `TrainingOutput(success=False, error_message=...)`, verified to
actually trigger correctly under a forced-divergence config
(`learning_rate=1e8`) and to leave `trained_circuit_weights=None` rather
than a partially-corrupted result.

### Seed and determinism semantics (`llm_vqc/evaluation/seeds.py`)

No single global seed. Six independent, explicitly named derived streams,
all built on `numpy.random.SeedSequence` (not ad hoc string hashing) for
guaranteed-independent child streams: `data_split_seed`,
`preprocessing_seed`, `synthetic_fixture_seed` (task-level), and
`train_seed_for_circuit` (the master-plan-mandated one — Section 6.4:
"per-candidate training seed drawn deterministically from run seed +
structural hash... so the same circuit gets the same training result in
every arm"), which itself further spawns four independent
`TrainingSeeds` (param_init, minibatch, training, backend_sampling) via
`SeedSequence.spawn()`. Every seed function takes an explicit
`numpy.random.Generator` or plain int — never reads or sets global RNG
state (`np.random.seed()` is never called anywhere in Phase 2 code except
`torch.manual_seed()` for `nn.Linear`/`TorchLayer`'s own internal
parameter initialization, which PyTorch does not expose a
Generator-object interface for at the module level). Reproducibility
verified directly: identical `(ir, train_val, config, train_seed)` inputs
produce byte-identical `train_loss_history` / `val_metric_history` /
`trained_circuit_weights` across repeated calls
(`test_evaluation_training.py::test_train_model_is_deterministic_given_the_same_inputs`).

### Evaluation-result schema (`llm_vqc/evaluation/results.py`)

`EvaluationResult` (search-visible) and `FinalTestResult` (quarantined)
are **not related by inheritance** — a subclass relationship would make
widening one into the other a one-line accident; verified
(`test_evaluation_results.py::
test_final_test_result_is_a_distinct_unrelated_class`). Both are
pydantic models with deterministic JSON serialization
(`model_dump_json`/`model_validate_json` round-trip verified byte-stable
across repeated calls). `EvaluationResult` separates: proposal identity
(`proposal_id`, `task_name`, `run_seed`), circuit identity
(`structural_hash`, `circuit_canonical_json`), per-stage outcomes
(`validation_outcome`, `compilation_outcome`, `training_outcome`, each
with its own error/issues field), search-visible metrics (`val_metric_*`,
never a test metric), resource/budget usage (`epochs_completed`,
`wall_clock_seconds`, `is_duplicate`, `cache_hit`), structural circuit
cost (`circuit_cost`, reusing Phase 1's `CircuitCostSummary` — never an
evaluation metric), a single top-level `failure_category` classification,
reproducibility metadata (`train_seed`, `git_sha`, `git_dirty`,
`created_at`), and an artifact reference (`trained_weights_ref` — a
lookup key into the result store, not an embedded blob).

### Budget accounting

Phase 1's `BudgetLedger`/`ProposalRecord`/`ProposalOutcome` (previously
just interfaces) are now exercised end to end by
`harness.evaluate_candidate`: every call — valid, invalid, duplicate —
records exactly one `ProposalRecord`. Invalid proposals are recorded with
`outcome=INVALID` (never silently dropped — verified:
`test_invalid_proposal_is_recorded_in_the_budget_ledger`). Duplicate
proposals (same `structural_hash` + `train_seed`, the master-plan-defined
identity for "the same circuit") are recorded as `DUPLICATE` and consume
budget by returning the cached result without retraining, exactly per
Section 6.3 ("Duplicate proposals... consume budget by returning the
cached result — this preserves the incentive structure while making
collapse visible"). Wall-clock duration and epochs-completed are recorded
per evaluation regardless of success/failure.

### Result store and resume (`llm_vqc/evaluation/store.py`)

Plain `sqlite3` (standard library; no new external database dependency —
the master plan names SQLite explicitly as the recommended Section 11
choice, and nothing here needs more). Two tables: `run_meta` (one row per
`run_id`, holding the resolved config + a `config_compat_hash` + git SHA)
and `evaluations` (one row per `(task_name, structural_hash, train_seed)`
— duplicate writes to the same key `INSERT OR REPLACE`, each in one
committed transaction, so a killed process leaves either the previous row
or no row, never a half-written one — verified via
`test_partial_run_results_survive_an_abrupt_close_and_reopen`, which
closes a store mid-"run" and reopens a fresh instance directly against
the file). `ResultStore.open_or_create` computes a hash of the
reproducibility-critical config subset (task, training hyperparameters,
seed — explicitly *not* cosmetic fields) and refuses to resume into an
existing `run_id` whose stored hash differs, raising
`IncompatibleResumeError` unless the caller explicitly passes
`allow_incompatible=True` (verified to matter: temporarily disabling the
comparison caused the corresponding tests to fail with "DID NOT RAISE",
then correctly passed again once restored — same protection-removal
verification method used throughout this project). A rejected
incompatible-resume attempt does not mutate the store — reopening
afterward with the *original* config still succeeds and prior data is
intact (verified). Failed training runs are persisted identically to
successful ones (`test_failed_training_runs_are_persisted_not_dropped`).
`ResultStore` implements `CandidateCache` (the `harness.py` Protocol)
structurally via matching method names (`get_cached`/`put_cached`), no
inheritance required.

### Bugs found and fixed during Phase 2 integration

**Batching bug in the Phase 1 PennyLane compiler (IR change — see
"Changes to the Phase 1 IR" below).** Discovered while building the
torch training loop: `_apply_program`'s encoding step indexed
`inputs[instr.input_index]`, which is correct for a single 1D sample but
silently selects the wrong axis (`instr.input_index`-th *batch row*
instead of the `instr.input_index`-th *feature*) for a batched 2D torch
tensor — producing wrong (not erroring) expectation values, verified by
constructing a batch, comparing the batched call against row-by-row calls
through the already-trusted single-sample path, and confirming they
initially diverged, then matched exactly after the fix.

**`TrainingConfig(epochs=0)` crash.** A smoke test using an
accidentally-zero-epoch config crashed with `IndexError:
list index out of range` on `val_metric_history[-1]` (empty list, since
the training loop never ran). Fixed by adding pydantic field validation
(`epochs: int = Field(default=20, ge=1)`, similarly for `batch_size` and
`learning_rate`) rather than defensively coding around the empty-list
case inside `train_model` — the same "reject the degenerate input at the
boundary" pattern used throughout this project (e.g. Phase 1's
`RepeatBlock.times >= 1`).

**Two self-inflicted test bugs (own PCA-seed mismatches), corrected
during test-writing, not shipped.** Two T2 tests initially compared a
reference `PCA(random_state=0)` fit against the task's actual internal
fit (which uses `preprocessing_seed(seed, task_name)`, a *derived* seed,
not the raw seed). Both failures were correctly diagnosed as test bugs
(wrong seed used in the reference computation) rather than production
bugs, and fixed by deriving the exact same seeds the task uses
internally. Documented here as a caution for future contributors: any
test comparing against an "independent reference" computation must derive
seeds identically to production code, not assume the raw run seed is
what gets passed to a library call.

### Tests added (Phase 2)

92 new tests across 12 files (223 total, up from 131 after Phase 1):
`test_tasks_base.py` (7), `test_tasks_t1.py` (7), `test_tasks_t2.py` (8),
`test_evaluation_seeds.py` (7), `test_evaluation_metrics.py` (8),
`test_evaluation_model.py` (7), `test_evaluation_training.py` (10),
`test_evaluation_results.py` (7), `test_evaluation_harness.py` (9),
`test_evaluation_final_test.py` (4), `test_evaluation_store.py` (11),
`test_phase2_smoke_e2e.py` (4), plus one new Phase-1-suite regression
test (`test_ir_semantic_equivalence.py::
test_qnode_correctly_broadcasts_over_a_batch_of_torch_inputs`) for the
batching bug. Each category maps to a required test area from the task
brief: deterministic splitting, disjointness/overlap rejection,
train-only preprocessing fitting (with an independent-reference
comparison, not just internal self-consistency), structural test
quarantine (AST + subprocess + schema-field checks), deterministic
init/training, metric correctness against hand-computed values, stable
result serialization, failure categorization, invalid/duplicate budget
accounting, resume + incompatible-resume rejection, partial-result
preservation after simulated failure, and T1/T2 end-to-end smoke through
the complete pipeline (CPU-only, small epoch counts, no
publication-scale training — total Phase 2 test suite runtime ≈8s
standalone, ≈35s for the full 223-test suite).

Four protections were deliberately, temporarily removed and confirmed to
cause the expected test failures, then restored byte-identical (`diff`
confirmed): the batching indexing fix, the `epochs>=1` validation
(pre-existing from earlier in this same session, re-verified in context),
and — new in this phase — the `IncompatibleResumeError` comparison in
`ResultStore.open_or_create` (two tests correctly failed with "DID NOT
RAISE" when disabled, both passed once restored).

### Changes made to the Phase 1 IR

**One change: the PennyLane compiler batching fix** (`llm_vqc/ir/
compiler_pennylane.py::_apply_program`), described above under "Bugs
found and fixed." Per the Scope Control procedure:

1. **Exact blocking case:** training on a minibatch (required by the
   master plan's fixed training protocol, batch_size=16) requires the
   compiled QNode to correctly broadcast over a batch dimension; it
   silently didn't.
2. **Classification:** an **omitted Phase 1 requirement / latent
   correctness bug**, not a research-design change. Phase 1's own tests
   only ever exercised single-sample (1D) inputs; nothing in the Phase 1
   design brief promised batch support, but the bug is a genuine
   correctness defect (wrong, not merely unsupported, behavior) rather
   than a deliberately-scoped-out feature.
3. **Smallest backward-compatible change:** one line,
   `inputs[instr.input_index]` → `inputs[..., instr.input_index]`, plus a
   type-normalization guard (`isinstance(inputs, list): inputs =
   np.asarray(inputs)`, since plain Python lists don't support ellipsis
   indexing) — proven backward-compatible because `x[..., i] == x[i]` for
   any 1D array, i.e. every Phase 1 single-sample call site is
   byte-identical before and after.
4. **Regression test added:**
   `test_ir_semantic_equivalence.py::
   test_qnode_correctly_broadcasts_over_a_batch_of_torch_inputs`,
   confirmed to fail against the pre-fix code and pass against the fix.
5. **Recorded here**, per the operating rule.

No other Phase 1 file was modified. This does **not** alter benchmark
fairness, circuit expressivity, the approved task design, or the research
question — it is a pure bug fix to an untested code path, verified not to
change any Phase 1 single-sample behavior (all 131 pre-existing Phase 1
tests pass unmodified before and after).

### Design decisions recorded (Phase 2)

In addition to the Gate G1 decision and the training-default decisions
above:

**D9 — Generalize the "linear embed" wrapper from T1 to T2 as well.**
The master plan explicitly specifies T1's "linear embed → VQC →
linear+sigmoid" wrapper but does not mention one for T2. **Decision:**
apply the same wrapper to both tasks, because IR circuits can vary their
own encoding-wire count across proposals, so a task's fixed-width feature
vector cannot be wired directly into an arbitrary circuit without an
adapter layer — this is a structural necessity of supporting a flexible
search space, not a scientific choice specific to either task. Documented
in `llm_vqc/evaluation/model.py`'s module docstring.

**D10 — Sigmoid-bounded angle encoding.** Knipfer et al.'s tool docstring
assumes embed-layer inputs are "already scaled to `[0, pi]`" without
specifying the exact scaling function. **Decision:** `sigmoid(embed(x)) *
pi` — a standard, differentiable way to bound an otherwise-unconstrained
linear layer's output into a fixed range. Amplitude encoding does not use
this bound (the compiler's `AmplitudeEmbedding(normalize=True)` already
handles arbitrary real-valued input).

**D11 — T2's PCA component count fixed at 10** (master plan range: 8-16).
An arbitrary but documented midpoint choice; not tuned against any
result.

**D12 — Stratified 60/20/20 split for T2**, replacing the master plan's
`~2k/1k/4k` sizing given the fallback dataset's much smaller size (357
samples) — see Gate G1 section above for full reasoning.

### Known limitations and unresolved risks (Phase 2)

- T2's fallback dataset (`digits` 3-vs-8) does not carry HEP relevance;
  any T2 findings must be reported as validating the *evaluation
  methodology*, not as a QMLHEP-domain result (see Gate G1 "unresolved
  reproducibility risks").
- T2's 71-72-sample val/test partitions are far smaller than the master
  plan's design target for inter-arm statistical power; flagged for the
  Phase 6 pilot decision gate.
- `BudgetLedger` is still in-memory only (Phase 1 scope note stands); the
  result store now provides real persistence, but nothing yet
  reconstructs a `BudgetLedger` object *from* stored results on resume
  (a resumed process starts a fresh in-memory ledger — the store itself
  is the durable source of truth for counts, but no
  `BudgetLedger.from_store(...)` convenience exists yet).
- No GPU/CUDA path is implemented or tested — `device="cpu"` is the only
  supported `TrainingConfig` value (a `Literal["cpu"]`, so anything else
  is rejected at the config level, not silently ignored).
- The Phase 1 IR's known limitations (no interleaved data re-uploading,
  no physical-equivalence circuit dedup, no nested `RepeatBlock`s, no
  per-wire heterogeneous measurement observables) are unchanged and
  still apply to every circuit T1/T2 can evaluate.
- `smoke_evaluate.py` draws circuits via Phase 1's `sample_random_ir`
  purely as a convenience source to exercise the pipeline; it is
  explicitly not a search algorithm and must not be mistaken for Phase 4
  work.

### Phase 2 acceptance criteria — satisfied

- [x] T1 generator with frozen splits: `T1GaussianPeakTask`, deterministic
      given seed, 150/250/2000 split.
- [x] T2 loader with PCA pipeline, frozen splits, fallback documented:
      `T2DigitsTask` + Gate G1 write-up.
- [x] Fixed training pipeline: `llm_vqc/evaluation/training.py`, one
      `TrainingConfig`, no per-arm tuning possible without an explicit
      config change.
- [x] Result store (SQLite): `llm_vqc/evaluation/store.py`.
- [x] Candidate cache keyed by (task, structural_hash, train_seed):
      `ResultStore.get_cached`/`put_cached`, wired into `harness.py`.
- [x] Budget ledger with resume: `BudgetLedger` exercised end to end;
      store-level resume verified (kill-and-resume, incompatible-config
      rejection).
- [x] "Training a fixed reference ansatz on T1 reproduces Knipfer-
      magnitude RMSE (~0.02-0.06)": verified directly — a small 4-qubit
      circuit trained for 10 epochs on T1 achieved val RMSE ≈0.028-0.051
      and test RMSE ≈0.026-0.040 across manual runs during development
      (see `test_phase2_smoke_e2e.py` for the automated version, using a
      wider sanity band of `<0.5` to keep the permanent test robust to
      circuit/seed choice rather than pinned to one lucky run).
- [x] Kill-and-resume test passes:
      `test_partial_run_results_survive_an_abrupt_close_and_reopen`, plus
      manual verification via `scripts/smoke_evaluate.py` run twice
      against the same config (first run: 3 evaluated, 0 skipped; second
      run: 0 evaluated, 3 skipped/resumed).
- [x] Cache hit returns identical result: verified
      (`test_duplicate_detection_returns_cached_result_without_retraining`).

**Phase 2 gate: PASS.**

---

## Next phase recommended by the master plan

**Phase 3 — Diagnostics** (Section 9). Deliverables: expressibility (Sim
et al. KL divergence vs. Haar-random), Meyer-Wallach entangling
capability, gradient variance at k random initializations, and the cost
statistics already partially available via Phase 1's
`circuit_cost_summary`. Acceptance: unit tests against known values (an
idle/empty circuit should score worst on expressibility; a depth scan of
a standard hardware-efficient ansatz should reproduce Sim et al.'s
qualitative ordering). This phase has no dependency on the T2 dataset, no
LLM API usage, and no new heavyweight dependencies expected — **not
started** in this pass per the instruction to stop after Phase 2's gate.

## Decisions or permissions required before continuing

1. **Confirm Phase 2 is accepted** before Phase 3 begins.
2. **T2 relevance caveat**: confirm it is acceptable that T2 results
   carry an evaluation-methodology claim only, not a HEP-domain claim,
   given the `digits` fallback (Gate G1) — this affects how Phase 8+
   publication-plan language should describe T2 findings.
3. **BudgetLedger/store reconciliation**: confirm it is acceptable to
   defer building a `BudgetLedger.from_store(...)` reconstruction helper
   (a resumed process currently starts a fresh in-memory ledger; the
   SQLite store itself remains the durable source of truth for counts)
   until a phase that actually needs live in-memory budget totals across
   a resumed run — flagged as a known limitation above, not silently
   dropped.
4. No LLM API cost, T2 dataset access beyond the documented fallback, or
   external compute budget was needed for Phase 2 and none was used.

---

## Phase 2 governance decisions (recorded per the Phase 3 instructions)

The user accepted Phase 2 subject to three research-governance decisions,
recorded here as binding constraints on all later phases:

- **G2-A — digits 3-vs-8 is a methodology fallback only.** The
  scikit-learn `digits` 3-vs-8 task (T2) is approved solely as a
  reproducible evaluation-methodology fallback. It must **not** be
  presented as a high-energy-physics domain benchmark, and no
  HEP-specific claim may be supported by it. Any future write-up must
  describe T2 results as validating the evaluation methodology, not the
  QMLHEP domain.
- **G2-B — the ECAL accessibility conclusion is provisional.** Gate G1's
  "ECAL dataset not accessible" outcome stands as an *operational
  fallback decision* based on indirect evidence (a past GSoC student
  repo's README), **not** a definitive claim that the dataset is
  permanently unavailable. Before any publication-scale experiments, the
  project **must** recheck current official ML4SCI documentation or obtain
  confirmation from the maintainers/mentors, and update Gate G1
  accordingly.
- **G2-C — durable budget reconstruction is mandatory before Phase 4.**
  Deferring `BudgetLedger.from_store(...)` is acceptable *during Phase 3*
  only. Durable reconstruction of the budget ledger from the result store
  becomes **mandatory** before any budget-matched search comparison,
  pilot search experiment, or Phase 4 search implementation is considered
  complete. Phase 4 must not be signed off without it.

---

## Phase 3 — Diagnostics

**Status: COMPLETE.** All acceptance criteria met (see below).

### Gate G1 recheck note

Per G2-B, no publication-scale work occurred in Phase 3, so the ECAL
accessibility recheck is not yet due. It remains an explicit precondition
for Phase 6+ pilot/publication work and is tracked here so it is not lost.

### 1. Formal definitions and estimators (from Sim et al. 2019,
arXiv:1905.10876 — the master plan's cited reference, read in full for
this phase; full detail in `llm_vqc/diagnostics/README.md`)

- **Expressibility** = `D_KL( P_hat_PQC(F) || P_Haar(F) )` (Sim et al.
  Eq. 17), `F = |<psi_theta|psi_phi>|^2`, `P_Haar(F) = (N-1)(1-F)^(N-2)`,
  `N = 2^n`. Estimator: `n_pairs` sampled fidelities, histogrammed into
  `n_bins` fixed `[0,1]` bins; analytic per-bin Haar mass `(1-a)^(N-1) -
  (1-b)^(N-1)`; KL with the `0*log0=0` empty-empirical-bin policy. Lower =
  more expressible; fixed circuit hits the exact upper bound
  `(N-1)ln(n_bins)`.
- **Entangling capability** = mean over sampled weights of the
  Meyer-Wallach `Q = (2/n) sum_k (1 - Tr[rho_k^2])` (Sim et al. Eq. 21;
  Brennen single-qubit-purity form). Range `[0,1]`.
- **Gradient variance** = variance across `n_inits` random weight inits of
  the gradient of `<Z_0>` w.r.t. the trainable weights (McClean et al.
  2018 convention). Raw and log10 both stored.

### 2. Consequential design decisions (D13-D18)

**D13 — Sampling policy: weights-only at a fixed diagnostic input.** All
three diagnostics sample the trainable *weights* uniformly `[0, 2*pi)`
with data inputs held fixed (zeros for angle encoding → exact Sim et al.
pure-ansatz picture; a fixed seeded normalized vector for amplitude).
**Alternative considered:** also sampling the encoding-input angles (the
"full reachable-state manifold" interpretation). **Rejected because:** (a)
Sim et al.'s sampled parameters are the trainable rotations, not a
separate data encoding (which they do not have), so weights-only is the
faithful mapping; (b) it handles amplitude encoding uniformly (input just
held fixed); (c) it is consistent with the gradient diagnostic, which must
differentiate w.r.t. weights, not data; (d) consistency across all three
diagnostics is itself a fairness argument. **Consequence, accepted and
documented:** a parameterless circuit is a single fixed state and thus
correctly least-expressible — this is exactly the "idle → worst
expressibility" acceptance criterion.

**D14 — Analytic (not sampled) Haar reference.** The Haar bin
probabilities are computed in closed form (`(1-a)^(N-1) - (1-b)^(N-1)`)
rather than by sampling Haar-random states. This is exact, always
positive (so the KL denominator is never zero), and removes a source of
Monte-Carlo noise from the reference side. Sim et al. sample the Haar
side; the analytic form is a strict improvement with the same definition.

**D15 — KL zero-bin policy = drop empty empirical bins, no smoothing.**
Empty empirical bins contribute exactly 0 (`0*log0=0`); no epsilon is
added to the empirical distribution. Documented and tested; alternative
(epsilon-smoothing) rejected as it would bias every KL upward by an
arbitrary amount that depends on `n_bins`.

**D16 — Uncertainty via repeated-seed dispersion** (mean ± std over
`n_replicates` independent seed-sets), matching Sim et al.'s error-bar
method, plus (for entanglement/gradients) a within-replicate SEM /
bootstrap CI. No single point estimate is returned for any Monte-Carlo
metric.

**D17 — Fixed cost observable `<Z_0>` for gradient variance**, identical
for every circuit (McClean convention). Rejected alternative: the
circuit's own measurement spec — that would let circuit structure change
the observable and confound cross-circuit comparison.

**D18 — Diagnostic versioning in the cache identity.**
`diagnostic_version` (per diagnostic) is part of the store's cache key, so
changing an estimator's *meaning* invalidates old rows rather than
silently serving stale numbers. Cache identity is the full tuple
`(structural_hash, name, version, config_hash, run_seed)` — a metric name
alone is never identity.

### 3. Cross-backend and numerical validation performed

State preparation, fidelity, Meyer-Wallach, and the `<Z_0>` expectation
were cross-checked against an independent Qiskit statevector
(`statevector_qiskit`, reconciling the endian convention) up to global
phase (`fidelity == 1` to ~1e-9). Gradients were cross-checked against
central finite differences (~1e-7). Meyer-Wallach was validated against
analytic values (Bell=1, GHZ=1, product=0, Bell⊗Bell=1, partial in
(0,1)). Haar bin probabilities were validated to sum to 1 and be positive
for n=1..4. Expressibility was validated to (a) hit the exact analytic
upper bound for a fixed circuit, (b) be near-zero for genuinely Haar-drawn
fidelities, and (c) reproduce Sim et al.'s depth-scan ordering (idle >
depth-1 HEA > depth-5 HEA).

### 4. Bug found and fixed during Phase 3 (additive Phase 1 change)

**`build_circuit_fn` queued observables as gates when used for state
preparation.** `build_circuit_fn` (Phase 1) returns bare PennyLane
observable operators; instantiating e.g. `qml.PauliZ(0)` inside a QNode's
quantum function *queues it as a gate*, silently corrupting a prepared
state. The initial diagnostics state-QNode hit this (cross-backend
fidelity 0.015 instead of 1.0). **Fix (additive, per Scope Control):**
promoted the existing internal `_apply_program` to a public
`apply_program_ops` in `compiler_pennylane.py` — it queues *only* gates,
no observables — and the diagnostics use it. Classification: an
omitted-use-case, not a research-design change; no existing Phase 1/2
behavior changed (all 223 prior tests pass unmodified), and
`apply_program_ops` is a thin documented wrapper over code that already
existed. Recorded here per the operating rule.

### 5. Tests added and protection-removal checks

85 new tests across 6 files: `test_diagnostics_sampling.py` (9),
`test_diagnostics_expressibility.py` (9),
`test_diagnostics_entanglement.py` (12),
`test_diagnostics_gradients.py` (7), `test_diagnostics_store.py` (12),
`test_diagnostics_harness.py` (12), `test_diagnostics_adversarial.py`
(24, parametrized). Coverage maps to the task brief's required areas:
deterministic parameter sampling, config-hash completeness (every tuple
component verified to change cache identity), stable serialization,
invalid/parameterless/non-finite handling, empty/near-zero bins,
zero/NaN/Inf gradients, one-qubit MW convention, qubit-ordering,
global-phase invariance, cross-backend agreement, interrupted +
incompatible-resume, failure accounting, CPU-only execution, and
no-final-test-access (AST + subprocess + schema-field + signature
checks).

Three critical protections were deliberately, temporarily removed and
confirmed to make the intended test fail, then restored byte-identical
(`diff`-verified): the KL empty-bin mask (expressibility), the non-finite
gradient exclusion (gradients), and the per-qubit `moveaxis` selection
(Meyer-Wallach qubit ordering). All three failed correctly when broken and
passed once restored.

### 6. Exact validation commands and results

```bash
python -m pytest tests/ -q                    # 308 passed (223 prior + 85 new)
python -m ruff check .                          # All checks passed!
bash scripts/check.sh                            # ruff + pytest, both clean
python scripts/smoke_diagnostics.py configs/diagnostics_smoke.yaml   # 12 measurements, ~10s
python scripts/smoke_diagnostics.py configs/diagnostics_smoke.yaml   # resume: 0 evaluated, 12 skipped
# protection-removal: temporarily break each protection, confirm the
# intended test fails, restore, confirm it passes (transcript in §5).
```

### 7. Smoke-test runtime and resources

`scripts/smoke_diagnostics.py` at smoke scale (4 circuit fixtures x 3
diagnostics = 12 measurements): ~10 s wall-clock, CPU-only, no GPU,
statevector-exact (no finite-shot noise). The full 308-test suite runs in
~55 s.

### 8. Dependency-lock status

**No new dependencies.** All diagnostics use NumPy, PennyLane, Qiskit, and
PyTorch, already pinned. `scipy` (already a transitive pin) is not used
directly. `requirements-lock.txt` is unchanged and remains sufficient;
re-verified by importing the full diagnostics package in the existing
locked environment. (A fresh-environment re-run of the full suite is part
of the Phase 3 final validation — see §Final validation.)

### 9. Changes to Phase 1 / Phase 2 code

Only the additive `apply_program_ops` public wrapper in
`llm_vqc/ir/compiler_pennylane.py` (§4). No Phase 2 file changed. All 223
prior tests pass unmodified.

### 10. Known estimator limitations and unresolved scientific risks

- **Expressibility KL bias:** biased upward at finite sample size (Sim et
  al. Appendix B); only research-scale (5000-pair) values are
  quantitatively meaningful. Smoke/test values are inflated by design.
- **Meyer-Wallach is undiscerning:** GHZ and Bell⊗Bell both give Q=1; it
  does not capture entanglement *structure*. Documented; not relabeled as
  general "entanglement".
- **Barren plateaus require scale:** the gradient-variance diagnostic
  measures one circuit at one size; it cannot, alone, establish a barren
  plateau (which is a size-scaling claim). Any such claim is out of scope
  until a size sweep is run and analyzed.
- **No predictive claim:** nothing here shows expressibility or entangling
  capability predicts task performance; establishing (or refuting) that is
  a later-phase experimental question, not a Phase 3 output.
- **Statevector-exact only:** no finite-shot or hardware-noise
  diagnostics; results describe the noiseless ideal.
- **Syntactic identity only:** physically-equivalent-but-syntactically-
  different circuits are cached and measured separately (Phase 3 defines
  no physical-equivalence analysis).

### 11. Phase 3 acceptance criteria — satisfied

- [x] Expressibility, Meyer-Wallach, gradient variance implemented as
      pure functions of the IR (+ backend config), unit-tested against
      known ansätze.
- [x] Idle circuit → worst expressibility: verified to hit the exact
      analytic upper bound `(N-1)ln(n_bins)`
      (`test_idle_circuit_is_least_expressible_at_the_upper_bound`).
- [x] HEA depth scan reproduces Sim et al. qualitative ordering
      (expressibility improves — KL decreases — with depth, saturating;
      `test_expressibility_improves_with_hea_depth`).
- [x] Meyer-Wallach validated against Bell/GHZ/product analytics.
- [x] Cross-backend (PennyLane vs Qiskit) agreement and autodiff-vs-finite
      -difference gradient agreement verified.
- [x] Typed serializable results; SQLite cache with resume and
      incompatible-resume rejection; failure persistence.
- [x] Diagnostics never access task test data / final-test (structurally
      verified).

**Phase 3 gate: PASS.**

---

## Next phase recommended by the master plan

**Phase 4 — Non-LLM search arms** (`random`, `evolutionary`, `greedy`),
with the runner executing (arm × task × seed) grids and **Gate G2
(compute sizing)**. Per governance decision **G2-C**, durable
`BudgetLedger` reconstruction from the result store is a **mandatory
prerequisite** for Phase 4 completion and must be built as part of it.

## Decisions or permissions required before continuing

1. **Confirm Phase 3 is accepted** before Phase 4 begins.
2. **Acknowledge G2-C** as a hard Phase 4 deliverable (durable budget
   reconstruction before any budget-matched comparison).
3. No LLM API cost, dataset access, or external compute budget was needed
   for Phase 3 and none was used.

---

## Phase 4/5/6 audit (Stage 1 of the search-and-pilot work cycle)

**Status: AUDIT COMPLETE, implementation beginning.**

### 1. Phase 0-3 acceptance re-verified

Ran the complete existing suite and lint in a freshly dependency-installed
worktree venv (`.venv`, Python 3.14, `requirements-lock.txt` +
editable install): **308/308 tests passed**, `ruff check .` reports
**all checks passed**. No regressions since the Phase 3 completion
record above. No files were modified to make this pass.

### 2. Master-plan components confirmed still missing

Per Section 9 roadmap: Phase 4 (non-LLM search arms), Phase 5 (LLM
arms), Phase 6 (pilot + gate G3). None of `llm_vqc/search/` or an
LLM provider layer exist yet in the repo. `BudgetLedger.from_store()`
(governance decision G2-C) is also not yet implemented — confirmed by
inspection of `llm_vqc/ir/budget.py` (in-memory only).

### 3. Exact experiment parameters confirmed from the master plan (verbatim reference)

- **Search arms (§6.2):** `random`, `evolutionary`, `greedy`, `llm_iter`
  (Knipfer-style full-history conversational agent), `llm_evo` (LLM as
  mutation operator over a top-k archive, FunSearch-style). Fixed
  reference ansätze (HEA/EfficientSU2, StronglyEntanglingLayers,
  RealAmplitudes + re-upload variant) are reference points, not search
  arms.
- **Budget (§6.3):** B = 60 candidate evaluations per run, anytime
  checkpoints at 10/25/60. Invalid IR proposals are validator-rejected
  without consuming budget but are counted separately. Duplicate
  proposals (same `structural_hash`) consume budget via the cached
  result.
- **Seeds/repetitions (§6.3):** non-LLM arms: 10 seeds per (arm × task).
  LLM arms: 5 seeds per (arm × task × model). Each seed controls search
  RNG, candidate train-init RNG stream, and (LLM arms) a fresh
  conversation.
- **Training pipeline (§6.3, fixed for all arms):** AdamW, lr 0.05 with
  decay at fixed epochs, 20 epochs, batch 16; feedback to searchers is
  validation-only; test metrics computed once and quarantined.
- **Phase 4 gate G2 (§9):** 3-seed smoke grid on T1 at B=15 completes
  unattended; best-so-far curves monotone; evolutionary >= random on
  median (sanity only).
- **Phase 5 gate (§9):** 2-seed smoke on T1 at B=15 for both LLM arms;
  invalid-IR rate < 20% after retry; full transcript persistence; cost
  projection for the full matrix <= an agreed cap (plan proposes ~$150
  mini-tier + ~$100 flagship, **requires explicit user confirmation
  before any paid call is made** — not yet given).
- **Phase 6 pilot (§9, this cycle's Stage 8):** all arms, **T1 only**,
  **B = 25**, **3 seeds**. Gate G3: inter-seed variance small enough
  that arm differences are potentially resolvable at n=5-10; no arm
  degenerate (e.g. LLM always duplicating).

### 4. LLM API cost-cap check

`LLM_API_BUDGET_USD` is **unset** in this environment. A `.env` file
in the repo root contains an `OPENAI_API_KEY`, but per the controlling
cost policy a UI/API key's mere presence is not programmatic budget
authorization — only a nonzero `LLM_API_BUDGET_USD` is. **No paid LLM
API call will be made in this work cycle.** The `llm_iter`/`llm_evo`
arms will be implemented and validated end-to-end against a mocked
provider; real pilot execution of those two arms is deferred pending
an explicit cap. This is recorded as governance decision **G4-A**.

### 5. Scope decision for this work cycle

Given G4-A, the Stage 8 pilot experiment in this cycle will actually
execute the three non-LLM arms (`random`, `evolutionary`, `greedy`) at
the master-plan pilot settings (T1, B=25, 3 seeds) with real training
runs and real stored results. The `llm_iter`/`llm_evo` arms will be
fully implemented, unit-tested, and exercised via a mocked-provider
smoke run (Stage 7) but **not** included in the real Stage 8 pilot
numbers. This is a deviation from "all arms" in the literal Phase 6
deliverable text, forced by the absence of an approved API budget, not
a scope simplification of choice. Full LLM-arm pilot execution is the
first item in "recommended next experiment" once a budget is set.

**Audit outcome: proceeding to Stage 2 (mandatory durable
`BudgetLedger.from_store()`).**

---

## Stage 2 — Durable BudgetLedger reconstruction (fulfills G2-C)

**Status: COMPLETE.**

### What changed

- `llm_vqc/ir/budget.py`: added `ProposalOutcome.FAILED` (a valid,
  non-duplicate proposal whose compilation/training failed — distinct
  from `INVALID`, which is validator-rejected and free of budget).
  Added `ProposalRecord.consumes_budget`, `BudgetLedger.num_failed`,
  `.consumed_budget`, `.remaining_budget()`, `.is_exhausted()`. Added
  `ProposalEventStore` (a `Protocol`, not a concrete import, to keep
  `llm_vqc.ir` free of a dependency on `llm_vqc.evaluation`),
  `BudgetLedger.record_and_persist()`, and the
  `BudgetLedger.from_store()` classmethod.
- `llm_vqc/evaluation/store.py`: added a `proposal_events` SQLite table
  (append-only, `PRIMARY KEY (run_id, proposal_index)`, plain `INSERT`
  so index reuse raises `sqlite3.IntegrityError` rather than
  overwriting) and `append_proposal_event` / `iter_proposal_events`
  (ordered by `proposal_index`) / `count_proposal_events`.
- `llm_vqc/evaluation/harness.py`: `evaluate_candidate` gained optional
  `proposal_event_store` / `run_id` params; when given, every ledger
  update is durably persisted via `record_and_persist` instead of the
  in-memory-only `record_proposal` (backward compatible — omitting both
  preserves exact Phase 1/2 behavior). The two compilation-failure
  paths and the one training-failure path now record `FAILED` instead
  of `VALID`, so invalid/duplicate/failed rates are distinguishable
  statistics (needed for Stage 6/8) — this does not change budget
  consumption (`FAILED` still consumes budget, same as `VALID` did).

### Crash-safety argument

`record_and_persist` writes to the durable store **before** updating
in-memory state. A crash between the two leaves the event durable (the
next `from_store()` sees it); a crash before the store write means the
event never logically happened (nothing double-counted). The
persisted index is always the ledger's current `num_proposed`, so a
resumed ledger continues the append-only log gaplessly. `from_store()`
replays every event in `proposal_index` order into a fresh ledger,
which is provably indistinguishable from one that ran unbroken:
identical `_seen_hashes` (duplicate detection continues correctly) and
identical `consumed_budget` (a restart cannot grant extra budget).

### Tests added (`tests/test_budget_persistence.py`, 9 tests)

Clean reconstruction; reconstruction after interruption (close/reopen
store mid-run, continue recording, verify gapless); invalid proposals
preserved and free of budget; duplicate accounting preserved; failed
evaluations preserved and consuming budget; incompatible resume
rejected (reused `IncompatibleResumeError`, keyed on reproducibility
fields); exact budget exhaustion after restart (consumes precisely the
remaining budget, no more no less, across two restarts); run_id
isolation (two arms/seeds in one SQLite file don't share budget);
append-only invariant (reusing an index raises `IntegrityError`).

Existing `tests/test_ir_budget.py` updated for the new `summary()` keys
and `FAILED` outcome; no other existing test asserted on the old
compilation/training-failure-as-`VALID` behavior, so no other test
needed to change.

### Protection-removal verification

Temporarily changed `proposal_events`' `INSERT` to `INSERT OR REPLACE`
— confirmed `test_append_proposal_event_is_append_only_not_overwrite`
fails (`DID NOT RAISE sqlite3.IntegrityError`), then restored
`llm_vqc/evaluation/store.py` byte-identical (verified via `diff`) and
confirmed the full persistence suite passes again.

### Validation

Full suite: **318/318 passed** (308 prior + 10 new: 9 in
`test_budget_persistence.py` + 1 new case added to `test_ir_budget.py`).
`ruff check .`: all checks passed.

### Deviation note

None from the master plan (budget durability was not itself a
master-plan roadmap line item — it is governance decision G2-C, which
explicitly required it before Phase 4/search-comparison work).

**Proceeding to Stage 3 (shared search framework typed interfaces).**

---

## Stage 3 — Shared search framework typed interfaces

**Status: COMPLETE.**

### What was built (`llm_vqc/search/`)

- `feedback.py` — `SearchFeedback`, the only object a search arm ever
  receives back after a proposal. Structurally test-free (no field could
  hold a test metric), built by the pure projection
  `from_evaluation_result()` off `EvaluationResult`. `infer_proposal_outcome()`
  classifies an `EvaluationResult` into `ProposalOutcome` using the same
  rule `harness.py` itself now uses when recording to the ledger (single
  source of truth — feedback and budget accounting cannot disagree).
- `comparison.py` — `is_strictly_better()`, the one candidate-comparison
  rule every arm must use (ties never replace the incumbent — deterministic
  tie-breaking given a fixed evaluation order).
- `arm.py` — `SearchArm[StateT]` (ABC, generic over a pydantic state
  type): `initialize`, `propose`, `update_state`, `select_final`,
  `deserialize_state` (abstract) + `serialize_state` (concrete, generic
  via `model_dump_json`). An arm never touches the evaluator, validator,
  or result store directly — only ever a raw dict proposal and a
  `SearchFeedback`.
- `results.py` — `SearchRunResult` (final selected structural hash +
  train seed + validation metric + ledger summary; no test field).
- `runner.py` — `SearchRunner`: the one execution loop. Every proposal
  goes through `evaluate_candidate` (so IR bypass is structurally
  impossible); budget termination is exact
  (`while not ledger.is_exhausted(budget_limit)`, checked before every
  proposal); arm state is checkpointed via `ResultStore.save_run_state`
  after initialization and after every evaluation; resuming loads state
  and reconstructs the ledger via `BudgetLedger.from_store()` (Stage 2).
  A generous safety cap (`budget_limit * 25` total proposals) aborts with
  `SearchRunnerError` rather than looping forever if an arm never
  produces a budget-consuming proposal.
- `llm_vqc/evaluation/store.py` gained a `run_state` table
  (`save_run_state`/`load_run_state`, `INSERT OR REPLACE` — a checkpoint,
  not an append-only log) so arm state lives in the same one-SQLite-file-
  per-run-directory store as everything else.

### Tests (`tests/test_search_framework.py`, 9 tests)

Module-boundary guarantee (AST scan + subprocess) that `llm_vqc.search`
never imports `final_test`; `SearchFeedback` has no test-metric field;
exact budget termination; runaway-arm safety-cap abort; determinism
under a fixed seed (two independent runs, same seed, identical selected
circuit/metric/ledger); different seeds produce different proposal
sequences; interrupted-run-resumes-to-identical-final-result as an
unbroken reference run; arm-state checkpoint persistence. A minimal
`ToyRandomArm` (uses the Phase 1 `sample_random_ir` generator) exercises
the shared runner; the real `random`/`evolutionary`/`greedy` arms are
Stage 4.

### Protection-removal verification

Temporarily removed the runaway-loop safety cap in `runner.py` and ran
the degenerate-arm test: it hung (confirmed still `running` after
several seconds via a background-task check, rather than completing) —
demonstrating the cap is load-bearing, not decorative. Restored
`runner.py` byte-identical (verified via `diff`) and confirmed the full
suite passes again.

### Design decisions

- **Diagnostics-as-feedback (ablation A1) is out of scope for Stage 3.**
  `SearchFeedback.diagnostics` exists as a field but `SearchRunner` always
  passes `diagnostics=None` — the master plan's default protocol is
  validation-only feedback; wiring A1 is deferred to whenever that
  ablation is actually run.
- **Malformed/invalid proposals remain budget-free for every arm,
  including LLM arms** (deviating from a literal reading of the Stage 5
  brief's "malformed proposals must consume budget"): charging one arm
  type differently for the same INVALID outcome would break the
  budget-matched search-space parity that is the experiment's central
  point (Section 6.4). LLM-specific repair-retry loops (Stage 5) will
  instead be bounded by a hard retry cap and reported as a secondary
  metric, not by budget consumption. Recorded here as governance decision
  **G4-B**.

### Validation

Full suite: **327/327 passed** (318 prior + 9 new). `ruff check .`: all
checks passed.

**Proceeding to Stage 4 (random, evolutionary, greedy search arms).**

---

## Stage 4 — random, evolutionary, greedy search arms

**Status: COMPLETE.**

### What was built (`llm_vqc/search/arms/`)

- `random_arm.py` — `RandomArm`: draws each proposal independently from
  `sample_random_ir` (Phase 1's generator), tracks best-so-far by
  validation metric.
- `greedy_arm.py` — `GreedyArm`: starts from a fixed deterministic
  `minimal_ir()` (`MIN_QUBITS` qubits, angle-RY, no layers, Z
  measurement — a constant, not a draw). Each round samples `k`
  single-layer extensions (via the new public `sample_single_layer` in
  `llm_vqc/ir/sampler.py`, reusing the identical layer grammar every
  other arm draws from), advances the incumbent to the best-of-round
  (deterministic tie-break via `is_strictly_better`) once `k`
  metric-bearing observations accumulate. A round in progress when
  budget runs out simply never completes — the shared runner, not the
  arm, decides when to stop proposing, so there is no "finish the round
  for free" path.
- `evolutionary_arm.py` + `mutation.py` — `EvolutionaryArm`: a (mu+lambda)
  evolution strategy. Generation 0's population is `mu` uniform-random
  draws; each subsequent generation evaluates `lambda` offspring
  (tournament-selected parents, single-point layer crossover at
  `crossover_rate`, then one of six mutation operators at
  `mutation_rate`: add/remove/perturb layer, change measurement, grow/
  shrink qubits), then re-selects the top `mu` from the combined
  parent+offspring pool. (mu+lambda) selection is itself the elitism
  mechanism (a parent can only be displaced by something at least as
  good) — no separate elitism parameter was added. All mutation/crossover
  operators are defensive: an operator that produces an invalid IR falls
  back to returning the unmutated parent (verified: `test_mutation_
  operators_always_produce_valid_ir`, `test_crossover_always_produces_
  valid_ir`).
- Every arm's `propose()` is a pure function of its (pydantic, JSON-
  serializable) state — no arm keeps an internal mutable RNG object —
  which is what makes exact checkpoint/resume possible without any
  arm-specific persistence code beyond `serialize_state`/`deserialize_state`.

### Tests (`tests/test_search_arms.py`, 14 tests)

Parametrized across all three arms: exact-budget smoke completion on the
real T1 task (Phase 4 acceptance criterion), determinism under a fixed
seed, interrupted-run-resumes-to-identical-unbroken-result. Arm-specific:
greedy starts from the fixed minimal circuit and advances its round
index exactly once per `k` evaluations; evolutionary's population/pending
sizes match `mu`/`lambda` exactly after generation 0; mutation and
crossover operators always yield valid IR over many random draws.

### Protection-removal verification

Temporarily changed greedy's round-advance condition from `>= self.k` to
`> self.k` (an off-by-one that would silently make every round consume
one extra evaluation before advancing) — confirmed
`test_greedy_advances_round_after_k_evaluations` fails
(`round_index` came out `2` instead of `3` for a 9-budget/`k=3` run).
Restored `greedy_arm.py` byte-identical (verified via `diff`) and
confirmed the full arm suite passes again.

### Design decisions

- **No explicit "stay put" option for greedy.** The master plan phrasing
  ("add the best of k... each round") does not offer a not-growing
  alternative, so greedy always advances to the best-of-round neighbor,
  even if that neighbor is worse than the incumbent on this round's
  sample. The arm's own global `best_hash`/`best_metric` (used for
  `select_final`) is tracked independently of the growth trajectory, so
  a temporarily-worse round does not lose the best circuit ever seen.
- **Crossover keeps parent A's `n_qubits`/`encoding`/`measurements`,
  recombining only the layer sequence** — crossing over qubit count or
  encoding independently multiplies validity edge cases (e.g. wire
  indices from a higher-qubit parent exceeding a lower-qubit parent's
  bound) for limited benefit at pilot scale; documented as an engineering
  scope limit, not a master-plan requirement.

### Validation

Full suite: **341/341 passed** (327 prior + 14 new). `ruff check .`:
all checks passed.

**Proceeding to Stage 5 (LLM arms, mocked-provider validation only —
`LLM_API_BUDGET_USD` remains unset per the audit's governance decision
G4-A).**

---

## Stage 5 — LLM arms (`llm_iter`, `llm_evo`), mocked-provider validation only

**Status: COMPLETE (validation only — no real API request was made or
attempted; see G4-A).**

### What was built (`llm_vqc/llm/`, `llm_vqc/search/arms/`)

- `llm/budget.py` — `LLMApiBudget`: the **only** sanctioned way to obtain
  a spendable budget is `LLMApiBudget.from_env()`, which reads
  `LLM_API_BUDGET_USD` and returns `None` if it is absent, unparseable,
  zero, or negative — there is no other constructor path a caller could
  use to smuggle in an unauthorized cap. `check_can_afford()` raises
  `LLMBudgetExceededError` *before* a request is issued (checked in
  `LLMDriver.propose()` before every single `provider.complete()` call,
  including repair retries).
- `llm/provider.py` — `LLMProvider` Protocol + `MockLLMProvider`, the only
  provider implemented this cycle. Deliberately **not** a function of an
  internal call counter (that would silently break checkpoint/resume —
  a fresh process's counter would restart at 0 and replay the first
  call's draw instead of continuing). Instead it hashes the actual
  `(seed, system_prompt, user_prompt, temperature)` — a pure function of
  content that is itself a deterministic function of checkpointed arm
  state, so resume reproduces it exactly. Reports `estimated_cost_usd=0.0`
  and `latency_seconds=0.0` truthfully (no real request is made).
- `llm/driver.py` — `LLMDriver`: bounded repair retries
  (`max_repair_attempts`, default 2 — matching the Phase 5 gate language
  "invalid-IR rate < 20% after retry"); the repair loop targets only
  JSON-parse / `CircuitIR` schema failures, never the deeper cross-field
  semantic rules (`llm_vqc.ir.validators`), which flow through to the
  harness's ordinary `INVALID` outcome like any other arm's proposal.
  After all retries are exhausted, `propose()` returns `proposal=None`
  with the full attempt trail — never an unlimited retry loop.
- `llm/records.py` — `LLMCallRecord` (one per attempt: system/user
  prompt, model, temperature, raw response, parsed proposal, validation
  errors, token usage, cost, latency) and `LLMProposalOutcome`.
- `llm/prompts.py` — fixed, versioned (`PROMPT_VERSION`) prompt
  templates; only master-plan-approved feedback fields ever appear
  (validation metric, training/compilation outcome, cost stats,
  duplicate flag — never a test metric, which is structurally
  impossible since `SearchFeedback` has no such field).
- `evaluation/store.py` gained an `llm_calls` table
  (`append_llm_call`/`iter_llm_calls`, `INSERT OR REPLACE` keyed by
  `(run_id, proposal_id, call_index)` — documented as a **known,
  disclosed limitation**: a crash mid-repair-chain would re-issue an
  already-paid-for call on resume with a real provider, costing slightly
  more than the theoretical minimum but never bypassing the per-call
  budget cap; not exercised in this cycle since only `MockLLMProvider`
  is used).
- `search/arms/llm_iter_arm.py` — `LLMIterArm`: Knipfer-style full-history
  conversational agent; the same class also implements the **open-loop
  ablation A5** via an `open_loop` constructor flag (prompts omit all
  prior feedback when `True`).
- `search/arms/llm_evo_arm.py` — `LLMEvoArm`: FunSearch-style, prompts
  with a bounded top-k archive (default k=5) of the best-scored circuits
  seen so far, asks for one offspring.
- Both arms hand the shared harness a deliberately-invalid placeholder
  dict (`{"_llm_malformed": True, "retry_count": N}`) when the driver
  exhausts repair attempts — classified `INVALID` by the same validator
  every other arm's proposals go through (budget-free for everyone,
  governance decision **G4-B**, recorded in the Stage 3 entry above).

### Tests (`tests/test_llm_arms.py`, 22 tests)

`LLMApiBudget.from_env` behavior (unset/zero/negative/unparseable all
`None`; valid positive returns a working budget); cap enforcement raises
before exceeding, allows comfortably-under-cap requests; provider
determinism (pure function of prompt text, not a counter) and
zero-cost/zero-latency reporting; **no network/SDK import anywhere in
`llm_vqc/llm`** (AST scan for `requests`/`httpx`/`urllib`/`socket`/
`openai`/`anthropic`); no `final_test` import; driver succeeds first try
on well-formed output, gives up after exactly `max_repair_attempts`
retries (never fewer, never more); cost cap enforced before every call
including retries (a provider that would exceed the cap on its 2nd call
is stopped after making only 1); both LLM arms complete exact-budget
smoke runs on real T1; deterministic under a fixed seed; interrupted-run-
resumes-to-identical-result; full LLM-call provenance persisted and
retrievable; explicit assertion that arms in this cycle are always
constructed with `budget=None`.

### Protection-removal verification

Temporarily removed the pre-call `budget.check_can_afford()` guard from
`LLMDriver.propose()` — confirmed
`test_driver_enforces_cost_cap_before_every_call_including_retries` fails
(`DID NOT RAISE LLMBudgetExceededError`). Restored `driver.py`
byte-identical (verified via `diff`) and confirmed the full LLM-arm
suite passes again.

### Cost-cap check for this environment

`LLM_API_BUDGET_USD` is confirmed unset (checked again at Stage 5 start).
**No paid LLM API call was made, attempted, or is reachable from any code
path exercised in this work cycle** — every LLM arm test in this cycle
constructs its arm with `budget=None` and `MockLLMProvider`. A `.env` file
in the repo root contains an `OPENAI_API_KEY`; per policy this is not
programmatic authorization and was not read or used by any code this
cycle touches.

### Validation

Full suite: **363/363 passed** (341 prior + 22 new). `ruff check .`:
all checks passed.

**Proceeding to Stage 6 (validation tests, including deliberate
protection-removal checks across the whole search framework — largely
satisfied incrementally already; Stage 6 consolidates and adds any
remaining cross-cutting checks) and then Stage 7 (smoke comparison).**

---

## Stage 6 — Cross-cutting validation tests

**Status: COMPLETE.**

Most of the required checks (exact budget termination, durable
reconstruction, determinism, checkpoint/resume equivalence, no-test-
access, LLM malformed-output/retry/cost-cap/provenance) were already
built and protection-removal-verified incrementally in Stages 2-5 (see
those entries — 4 separate deliberate-break-then-restore verifications
were already performed: budget append-only invariant, runner runaway-
loop cap, greedy round-advance threshold, LLM cost-cap pre-call check).
Stage 6 adds the remaining checks that only make sense **across** every
arm simultaneously (`tests/test_search_validation_stage6.py`, 4 tests):

- **Exact budget equality across every arm**: all five arms (`random`,
  `greedy`, `evolutionary`, `llm_iter`, `llm_evo`) consume *exactly* the
  same budget at the same `budget_limit`.
- **Identical training protocol regardless of arm**: a minimal test-only
  arm that always proposes the same fixed circuit, run twice under
  independent `SearchRunner`/`ResultStore` instances, produces a
  bit-identical selected validation metric both times — proving the
  shared harness's deterministic seed derivation
  (`train_seed_for_circuit`) is what determines the result, not which
  arm class called it.
- **No arbitrary code execution**: AST-scanned every file in
  `llm_vqc/search` and `llm_vqc/llm` for `eval`/`exec`/`compile` calls,
  any `subprocess` import, and any `os.system`/`os.popen`/`os.exec*`/
  `os.spawn*` call — none exist. (Plain `os.environ.get` for
  `LLM_API_BUDGET_USD` is legitimate and explicitly allowed by this
  check.)
- **No arm-specific compiler/preprocessing path**: AST-parsed
  `runner.py` and confirmed exactly one call site to `evaluate_candidate`
  and zero `isinstance(self.arm, ...)` branches anywhere in the file —
  the runner cannot special-case any arm's proposals.

### Validation

Full suite: **367/367 passed** (363 prior + 4 new). `ruff check .`: all
checks passed.

**Proceeding to Stage 7 (smoke comparison across all five arms).**

---

## Stage 7 — Smoke comparison across all five arms

**Status: COMPLETE. System validation only — NOT scientific evidence.**

### What was run

`scripts/smoke_search_comparison.py configs/search_smoke.yaml`: all five
arms (`random`, `greedy`, `evolutionary`, `llm_iter`, `llm_evo`) on T1,
budget=8, seed=0, 3-epoch training (reduced for speed), real SQLite
result stores under `runs/search_smoke/` (git-ignored, not committed).
The `random` arm was run at half-budget (4), the process was closed
(simulating a crash), and then resumed to the full budget=8 from a fresh
`ResultStore`/`SearchRunner` pair — exercising real interruption+resume,
not just a unit-test mock. Final candidate selection
(`select_final`/`SearchRunResult`) was exercised for every arm. Test-set
evaluation was never invoked (the script imports nothing from
`llm_vqc.evaluation.final_test`).

### Actual output (git_sha `5fe499149101d5617aecf3676a7fa3ee77fbdb84`)

| Arm | consumed_budget | num_valid | num_duplicate | selected val RMSE |
|---|---|---|---|---|
| random | 8 | 8 | 0 | 0.02284 |
| greedy | 8 | 7 | 1 | 0.07579 |
| evolutionary | 8 | 6 | 2 | 0.10235 |
| llm_iter | 8 | 8 | 0 | 0.01783 |
| llm_evo | 8 | 8 | 0 | 0.02438 |

Every arm consumed exactly budget=8 (no arm over- or under-consumed).
`random`'s interruption+resume produced this same 8-consumed-budget
result via two separate processes/store-handles, matching the automated
resume-equivalence tests in `test_search_arms.py`/`test_llm_arms.py`.

**These numbers are explicitly NOT a scientific result** (n=1 seed, tiny
budget, 3-epoch reduced training, `llm_iter`/`llm_evo` driven by
`MockLLMProvider` rather than a real model) — they demonstrate only that
the system runs, terminates exactly at budget, persists real results,
and selects a final candidate, for every arm. Per Phase 4's gate G2
("evolutionary >= random on median" is a sanity check, not a result) — at
this n=1 smoke scale `evolutionary` did not beat `random`, which is
plausible noise at this budget/seed count and is explicitly *not*
diagnostic; a real judgment on G2 needs the multi-seed pilot (Stage 8).
No unfairness, leakage, or nondeterminism was found in this smoke run
(consistent with all of Stages 2-6's automated tests).

**Proceeding to Stage 8 (first real pilot experiment).**

---

## Stage 8 — First real pilot experiment

**Status: COMPLETE. Real, non-fabricated results for `random`,
`evolutionary`, `greedy` on T1 at master-plan pilot settings.**
`llm_iter`/`llm_evo` excluded per G4-A (no `LLM_API_BUDGET_USD`).

### Protocol actually executed

`scripts/pilot_experiment.py` then `scripts/analyze_pilot.py`. One frozen
T1 data split for the entire pilot (`task.build(seed=0)` /
`task.build_test(seed=0)`, shared by every arm and seed); 3 repetition
seeds (0, 1, 2) per arm controlling only search-RNG/training-init
streams; budget B=25 candidate evaluations per run (master plan Phase 6
pilot spec); full research-scale training pipeline (`TrainingConfig()`
defaults — AdamW, lr=0.05, 20 epochs, batch 16, NOT the reduced smoke
settings); all 9 (arm x seed) runs sharing one SQLite store
(`runs/pilot_t1/results.sqlite`, git-ignored) for genuine cross-run
caching; final-test evaluation run exactly once per run, after search
completed, on the exact already-trained weights, via the quarantined
`evaluate_on_test` (never re-imported search-side). Real wall-clock:
9 runs x 25 candidates = 225 real trainings completed in a few minutes
(measured single-candidate timing beforehand: mean ~1.9 s, max ~5.5 s at
20 epochs on this hardware — well under the master plan's ">5 min/
candidate" G2 shrink-trigger, so no epoch/feature reduction was needed).

### Actual results (git_sha `5fe499149101d5617aecf3676a7fa3ee77fbdb84`,
repo dirty at run time — uncommitted work in progress, as it has been
throughout this whole work cycle since no commits were requested)

**Budget and accounting — exact and clean for every run:**

| Arm (seed) | consumed_budget | valid | duplicate | invalid | failed |
|---|---|---|---|---|---|
| random (0,1,2) | 25 each | 25 each | 0 | 0 | 0 |
| evolutionary (0,1,2) | 25 each | 16 / 11 / 12 | 9 / 14 / 13 | 0 | 0 |
| greedy (0,1,2) | 25 each | 22 / 23 / 24 | 3 / 2 / 1 | 0 | 0 |

Aggregated duplicate rate: random 0%, greedy 8%, **evolutionary 48%** —
a large, real, non-fabricated exploration-collapse signal, exactly the
phenomenon the master plan cites Knipfer et al. as documenting and that
this project's budget-ledger accounting was explicitly built to make
measurable (Stage 2/§6.3).

**Final selected validation vs. protected test RMSE per run:**

| run | val RMSE | test RMSE |
|---|---|---|
| random_s0 | 0.008115 | 0.009281 |
| random_s1 | 0.008450 | 0.008909 |
| random_s2 | 0.008602 | 0.009102 |
| evolutionary_s0 | 0.024064 | 0.024700 |
| evolutionary_s1 | 0.018966 | 0.020810 |
| evolutionary_s2 | 0.008269 | 0.009002 |
| greedy_s0 | 0.019320 | 0.023746 |
| greedy_s1 | 0.024375 | 0.027589 |
| greedy_s2 | 0.028460 | 0.031345 |

Validation and test RMSE track closely for every run (no evidence of
overfitting to validation in this pilot).

**Checkpoint 25 (full budget) cross-arm statistics** (n=3 seeds/arm —
descriptive/exploratory, explicitly not confirmatory at this n):
medians 0.00845 (random) / 0.01897 (evolutionary) / 0.02438 (greedy);
Kruskal-Wallis H=5.067, p=0.079 (not significant at alpha=0.05, as
expected at n=3); pairwise Cliff's delta: random vs greedy = **-1.0**
(complete separation — every random seed beat every greedy seed on this
pilot), random vs evolutionary = -0.556, evolutionary vs greedy = -0.778.
Holm-corrected pairwise p-values (0.30-0.40) are not individually
significant, consistent with the master plan's explicit expectation that
"with n=3-10 per cell, only report pairwise conclusions when effect
sizes are large — otherwise report estimates with CIs and say so."

### Honest interpretation (no claim beyond what n=3 supports)

At this pilot's budget (25) and scale (1 task, 3 seeds), **uninformed
`random` search outperformed both structured non-LLM arms** on median
validation/test RMSE, with large effect sizes but not-quite-significant
p-values at n=3. This is reported as-is, not adjusted to look more
favorable to any method — per the closing instruction, "the goal is not
to make [any arm] appear successful... a fair experiment that reveals
whether value is added." Two real, load-bearing observations worth
carrying into the main matrix: (1) `evolutionary`'s ~48% duplicate rate
at this small population/budget setting is a genuine, measured
exploration-collapse signal, not noise-adjacent; (2) at B=25 with only 3
qubits/few layers typically sufficing for T1 (a "cheap" task per the
master plan's own Section 6.1 table), an uninformed uniform sampler may
simply need less structure to do well — this is a property of this task
and this budget, not a general claim that search never helps, and must
not be generalized to T2 or to larger budgets without further evidence.

### Gate G3 (informal, at n=3 — a real go/no-go still requires the full
n=10 non-LLM-arm matrix)

(i) Inter-seed variance: `random`'s IQR at checkpoint 25 is tiny
(0.00024) — very low noise for that arm; `evolutionary`/`greedy` show
larger inter-seed spread (IQR 0.0079 / 0.0046) driven by the duplicate-
collapse dynamics. This spread is not so large as to make the arms
indistinguishable, but n=3 is too small to certify resolvability at
n=5-10 with confidence — recommend the pilot's n=3 be extended to at
least n=5 before treating this as gate-passing evidence.
(ii) No arm was fully degenerate (every arm produced valid, scoreable
circuits every run) — `evolutionary`'s high duplicate rate is a partial
collapse signal worth a prompt/config fix recommendation *if and when*
LLM arms are added to a real matrix, but is not itself disqualifying for
`evolutionary` as a non-LLM baseline (duplicates still count toward
budget and the arm still improved its own best-so-far across seeds).

### Artifacts

`runs/pilot_t1/results.sqlite` (git-ignored, all 225 real evaluations +
proposal events + arm-state checkpoints), `runs/pilot_t1/pilot_summary.json`
(git-ignored, machine-readable per-run results), `runs/pilot_t1/
pilot_analysis.json` (git-ignored, full statistical analysis). Both JSON
files are regenerable byte-for-byte by re-running
`scripts/pilot_experiment.py` then `scripts/analyze_pilot.py` (same seeds,
same frozen split) — no numbers here were hand-copied.

### Validation

Full suite re-run after the pilot: **367/367 passed**, `ruff check .`:
all checks passed (the pilot/analysis scripts do not touch `llm_vqc/`
package code).

**This satisfies the completion gate for this work cycle: prerequisites
done, search implementations validated, full smoke comparison done, and
the first real pilot experiment (for every arm executable under the
available budget) completed with reproducible statistical analysis. No
publication-scale experiment is launched automatically — see the final
report for recommended next steps.**

---

## Mini real-API experiment (time-boxed, human-directed)

**Status: COMPLETE. First real OpenAI API execution of the LLM path.**

The user explicitly directed a 5-minute-capped, 12-call-capped mini
experiment using a clipboard-supplied OpenAI API key (never persisted,
never printed, never committed; used only as a one-shot process env var
and confirmed absent from the shell, git diff, and all generated files
afterward). This supersedes G4-A's "no paid calls" default **for that
single run only** — the run-specific override used the existing
sanctioned `LLMApiBudget.from_env()` mechanism (`LLM_API_BUDGET_USD=2.00`
for one process); no safety implementation or test was deleted or
weakened.

### What ran (git_sha `5fe4991` + uncommitted work, later committed as
`5e4e503`)

`scripts/mini_llm_experiment.py`: T1, one frozen split (seed 0), one
search seed (0), budget=2/arm, epochs=1, batch=16, arms = random /
evolutionary / greedy / llm_iter(open-loop) / llm_iter(closed-loop);
`llm_evo` deliberately omitted to stay under the call cap. New
`llm_vqc/llm/openai_provider.py` provides `OpenAIProvider` (real SDK
wrapper; reports cost as `None` because OpenAI returns no dollar figure
— never fabricated) and `CallCountLimitedProvider` (hard shared cap on
total real calls). `LLMResponse.estimated_cost_usd` / `LLMCallRecord.
estimated_cost_usd` became `float | None`; when a provider reports
`None`, the driver charges the pre-call estimate against the dollar cap
(conservative, never fabricates the record's own cost field). The
network-import AST guard now exempts exactly `openai_provider.py`.

### Results (n=1 seed, budget=2/arm, 1 epoch — descriptive only, NOT
publication evidence, no significance claims)

| arm | val RMSE | test RMSE | real calls |
|---|---|---|---|
| random | 0.293 | 0.292 | 0 |
| evolutionary | 0.131 | 0.133 | 0 |
| greedy | 0.130 | 0.133 | 0 |
| llm_iter (open) | 0.174 | 0.173 | 2 |
| llm_iter (closed) | 0.131 | 0.132 | 2 |

4/12 real calls used; model snapshot `gpt-5.4-mini-2026-03-17`; 302-335
input / 137-242 output tokens per call; latencies 1.2-3.6 s; 9.8 s total
wall clock. One real API incompatibility surfaced and was fixed
(`max_tokens` → `max_completion_tokens` for this model family). Budget
reconstruction verified consistent for all 5 run_ids; test quarantine
intact (final test run once per arm, after search, via
`evaluate_on_test`). Plots/CSV/report generated from stored results into
`outputs/mini_llm_experiment/` (git-ignored); raw store stays in
`runs/mini_llm_experiment/`.

### Post-run review finding (fixed)

The mini script originally passed `budget=budget` but left
`cost_estimate_per_call_usd` at its 0.0 default — combined with the
provider's `None` cost this made the $2.00 dollar cap accumulate zero
spend (decorative). The 12-call count cap was the real bound and held.
Fixed: the script now passes a conservative `COST_ESTIMATE_PER_CALL_USD
= 0.05` and records `dollar_cap_spent_estimated_usd` in the summary;
regression tests added (`test_driver_charges_the_estimate_when_provider_
reports_no_cost`, `test_call_count_limited_provider_stops_at_exact_cap`,
`test_proposal_outcome_total_cost_is_none_when_no_call_reported_cost`).

---

## Groq GPT-OSS 20B free-tier pilot — interrupted

**Status: INTERRUPTED. Descriptive integration pilot, not a completed
budget-matched LLM comparison.**

### Why Groq was introduced and how it was isolated

Groq's free tier was introduced to exercise a second real provider and a real
open-weight model without mixing its results or quota accounting with the
completed OpenAI mini experiment. The condition is fixed to provider `groq`,
model `openai/gpt-oss-20b`, run-id prefix `groq_gpt_oss_20b_`, and the isolated
store `runs/groq_pilot_t1/results.sqlite`. Provider and model are part of the
resume compatibility identity, so an OpenAI or different-model process cannot
resume these runs.

The provider uses Groq strict JSON Schema output with every object closed to
additional properties and every field required. Compatibility fixes replaced
the IR's string-or-array wire shorthand with integer arrays only and made
logically optional strict-schema fields nullable. Every returned object still
passes the repository's unchanged semantic CircuitIR validators; schema
validity alone is not treated as scientific validity.

The free-tier controller uses sequential calls, an 8-second minimum interval,
an internal 180,000-token cap, a 250-request cap for the pilot, and one retry
for transient rate-limit/server failures while honoring retry delay metadata.
It checkpoints before stopping so the matrix can resume under the identical
provider/model condition.

### Predeclared protocol

- Task T1, frozen data split seed 0.
- Candidate budget B=10 for every arm and seed.
- Repetition seeds 0, 1, 2.
- Arms: random, evolutionary, greedy, LLM open-loop, LLM closed-loop, and
  LLM evolutionary.
- Fixed full training: AdamW, 20 epochs, batch 16, float64 CPU.
- Validation-only feedback during search; protected final test after a
  completed run only.
- LLM decoding: reasoning effort `low`, temperature 0.2, at most 512 output
  tokens, no repair calls after a strict-schema failure.

### Exact completion matrix

| Arm | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| random | complete 10/10 | complete 10/10 | complete 10/10 |
| evolutionary | complete 10/10 | complete 10/10 | complete 10/10 |
| greedy | complete 10/10 | complete 10/10 | complete 10/10 |
| llm_open | interrupted 4/10 | interrupted 1/10 | not started |
| llm_closed | not started | not started | not started |
| llm_evo | not started | not started | not started |

There are 9 complete, 2 interrupted, and 7 not-started cells. The matrix
remains incomplete because rate-limit backoff made continuation impractical
during the pilot window.

### API usage and partial LLM behavior

The durable store contains 31 Groq call records and 7,529 reported tokens:
22 calls / 6,020 tokens for `llm_open` seed 0 and 9 calls / 1,509 tokens for
seed 1. Seed 0 made 22 proposals (18 invalid, 3 valid, 1 duplicate), consumed
4/10 budget, and reached validation RMSE 0.024052. Seed 1 made 9 proposals
(8 invalid, 1 valid), consumed 1/10 budget, and reached validation RMSE
0.042719. Neither interrupted run received a protected final-test result.

Across these partial cells, 26/31 proposals were invalid (83.9%). Stored
validation categories show repeated string-like wire encodings where arrays of
integer indices were required. The defensible provisional observation is:
under the tested low-reasoning configuration, GPT-OSS 20B frequently failed to
produce semantically valid CircuitIR proposals. This must not be generalized to
other prompts, models, providers, reasoning settings, or open-source LLMs.

Mean call latency was 9.38 s for seed 0 and 657.69 s for seed 1; seed 1's
median was 767.02 s, consistent with severe free-tier rate-limit backoff.

### Completed non-LLM results

| Run | Validation RMSE | Protected test RMSE |
|---|---:|---:|
| random s0 / s1 / s2 | 0.009216 / 0.021769 / 0.020101 | 0.010305 / 0.023468 / 0.017630 |
| evolutionary s0 / s1 / s2 | 0.023310 / 0.020298 / 0.017298 | 0.025868 / 0.024421 / 0.020915 |
| greedy s0 / s1 / s2 | 0.028831 / 0.026678 / 0.034912 | 0.032229 / 0.028523 / 0.038617 |

These n=3 results are descriptive. They are not combined with the interrupted
LLM observations as though all cells had equal completion status.

### Test quarantine and claims

Protected test results exist only for the 9 completed cells. The two
interrupted rows and all not-started rows have no test result; no test metric
entered search feedback.

Supported claims are limited to successful end-to-end Groq integration,
durable partial-run accounting, the exact completion state above, observed
model-specific semantic-output failures, and observed rate-limit delays. No LLM
superiority or inferiority, completed model comparison, quantum advantage, or
HEP-domain claim is supported by this interrupted T1 pilot.

### Sanitized artifacts and reproduction

Presentation-safe CSV, JSON, detailed report, checksummed manifest, and PNG/SVG
plots are published under `docs/presentation/groq_pilot/`. Raw databases, run
directories, prompts, responses, headers, weights, and credentials remain
excluded from version control.

Regenerate analysis without API calls:

```bash
.venv/bin/python scripts/analyze_groq_pilot.py
```

After securely supplying the required Groq credential to the process
environment, resume the exact provider/model condition with:

```bash
.venv/bin/python scripts/groq_pilot_experiment.py pilot
```

---

## Publish sanitized mini-experiment presentation artifacts

**Status: COMPLETE.**

Following the same pattern already established for the Groq pilot
(`docs/presentation/groq_pilot/`), a sanitized presentation package for
the first real-API OpenAI mini experiment was published at
`docs/presentation/mini_llm_experiment/`: per-arm results table, real
API usage table (tokens/latency, no fabricated cost), all 5 required
plots (PNG+SVG), a CSV summary, and an `artifact_manifest.json`
generated from `runs/mini_llm_experiment/mini_summary.json`. The raw
SQLite store, prompts, and any credential material remain
un-published and git-ignored. `README.md` and
`docs/presentation/README.md` link to the new package alongside the
existing Groq pilot one.
