# QAE-TFIM semantic-prior pilot protocol

Status: frozen verification protocol used after an initial development check.

## Research question
Does task-aware language-model reasoning help VQC architecture search when the task exposes meaningful quantum structure rather than only a flat gate grammar?

## Task
Compress the ground-state family of the 4-qubit open-chain transverse-field Ising model

`H(h) = - Σ_{i=0}^{2} Z_i Z_{i+1} - h Σ_{i=0}^{3} X_i`, `h ∈ [0.2,2.0]`

from four input qubits to two latent qubits `q0,q1`. Qubits `q2,q3` are trash qubits. The encoder is trained so the trash subsystem is `|00>`. The score is trash-state fidelity `F_trash = P(q2 q3 = 00)` after the encoder; higher is better.

## Shared circuit budget
Every candidate has exactly:
- 4 qubits
- 3 single-qubit rotation layers × 4 rotations = 12 trainable angles
- 4 fixed CNOT gates
- 16 gates total
- identical optimizer and training schedule

Allowed rotation axes are X/Y/Z. CNOT direction is part of architecture.

## Search methods
- **Random:** 8 candidates; axes sampled from X/Y/Z, 4 directed CNOT pairs sampled without replacement.
- **RY-random:** 8 candidates; all 12 rotations RY, topology random. This controls for the simple prior that the Ising ground states can be chosen real.
- **LLM-semantic:** fixed pool of 8 candidates generated once from the prompt in `LLM_QAE_SKILL.md` before the frozen verification run. Model: GPT-5.6 Sol in this ChatGPT session. No protected-test result was shown during pool generation.

## Training and selection
For every candidate:
- Adam, learning rate 0.05
- 60 epochs
- parameters initialized uniformly in [-0.05,0.05]
- best epoch selected by validation loss `1 - F_trash`

For each of 12 verification seeds:
- 32 fresh `h` values sampled uniformly for training
- 12 fresh `h` values sampled uniformly for validation
- fixed 64-point unseen `h` grid in `[0.215,1.985]` for protected test
- each method receives B=8 candidate evaluations
- winner selected only by validation loss

## Scope
This is a small, noiseless 4-qubit simulation and an exploratory semantic-prior pilot. It is not evidence of general LLM superiority, quantum advantage, or hardware performance. The exact prompt should be replayed through a version-pinned API before publication-quality claims.

## v2 — version-pinned API verification (completed)

`llm_vqc/experiments/qae_tfim/api_v2.py` replays the frozen prompt through the
real OpenAI API and extends the method matrix under the identical budget:

- **LLM-API-open:** one call to `gpt-5.4-mini-2026-03-17` (temperature 0.7)
  requesting 8 distinct JSON candidates; pool frozen, then evaluated on all
  12 seeds.
- **LLM-API-closed:** per seed, 8 sequential calls receiving only validation
  trash fidelity + duplicate flags of that seed's earlier candidates. Invalid
  or duplicate proposals consume budget and are replaced by flagged random
  circuits.
- **Evolutionary:** 4 random parents + 4 mutations (one axis or one CNOT pair)
  of the top-2 parents by validation loss; exactly 8 evaluations.
- **Reference-QAE:** one hand-designed textbook encoder (RY–RZ–RY layers,
  brickwork nearest-neighbour CNOTs), no search.
- **Random / RY-random / frozen chat pool:** identical rng streams to the pilot.

Hard cost cap via `LLM_API_BUDGET_USD`; every raw call, token count, and model
snapshot stored under `outputs/qae_tfim_api_v2/llm_calls/`. Results:
`outputs/qae_tfim_api_v2/REPORT.md`.
