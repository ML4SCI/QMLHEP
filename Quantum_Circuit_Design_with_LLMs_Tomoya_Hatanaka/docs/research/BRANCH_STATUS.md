# Branch status — post-consolidation (final)

**`main` is the single branch of this repository.** Every other branch has been
audited and deleted. Nothing you need is on another branch, and no document in
this repository asks you to check one out.

Historical development did happen across many branches. This file is the
permanent record of what each one was, where its content lives now, and the
evidence used to decide it was safe to delete. Original tip SHAs are recorded so
the decision stays auditable.

For the final work product see
[`../../GSoC2026_FINAL_REPORT.md`](../../GSoC2026_FINAL_REPORT.md). The immutable
end-of-GSoC snapshot is the annotated tag **`gsoc-2026-final`**.

---

## How the audit was done

For every remote branch:

1. `git rev-list --count main..<branch>` — unique commits;
2. `git merge-base main <branch>` — connected or disconnected history;
3. a **filename-set** comparison against `main`
   (`comm -23 <branch files> <main files>`) — does it hold any file `main` lacks?
4. for every path present on **both**, a **blob-SHA** comparison — does `main`
   hold a *different* version, i.e. did `main` lose data?

A branch was deleted only after (3) and (4) both came back empty, or after its
unique content was copied onto `main`. **No branch was deleted because of its
name.**

---

## Group A — connected history, fully represented on `main`

These branched from a commit on `main` and were merged by **squash**, so their
commits have different SHAs on `main` but identical content. Verified: **0**
files present on the branch and absent from `main`, and **0** shared files under
`outputs/` whose content differs from `main`.

| Branch (deleted) | Tip SHA | Merged as | Where it lives now |
|---|---|---|---|
| `claude/llm-vqc-architecture-plan-8417fd` | `056b045` | — (0 unique commits; tip is an ancestor of `main`) | already on `main` |
| `codex/publish-output-graphs` | `1b5f3a8` | PR [#1](https://github.com/dorakingx/llm-vqc/pull/1) | already on `main` |
| `experiment/capacity-controlled-t1-v1` | `1d21acc` | PR [#4](https://github.com/dorakingx/llm-vqc/pull/4) | `outputs/capacity_controlled_t1_v1/` |
| `experiment/capacity-controlled-t1-v2` | `4a87380` | PR [#4](https://github.com/dorakingx/llm-vqc/pull/4) | `outputs/capacity_controlled_t1_v2/` |
| `experiment/capacity-controlled-t2-v1` | `afe406b` | PR [#4](https://github.com/dorakingx/llm-vqc/pull/4) | `outputs/capacity_controlled_t2_v1/` |
| `research/unified-qae-v1` | `e514c72` | PRs [#4](https://github.com/dorakingx/llm-vqc/pull/4), [#5](https://github.com/dorakingx/llm-vqc/pull/5) | `outputs/qae_tfim_semantic_pilot_v1/`, `paper/` |
| `claude/llm-vqc-consolidate-qae-673a04` | `c44a4a5` | PRs [#5](https://github.com/dorakingx/llm-vqc/pull/5), [#8](https://github.com/dorakingx/llm-vqc/pull/8), [#9](https://github.com/dorakingx/llm-vqc/pull/9) | `outputs/qae_tfim_api_v2/`, `_neutral_v3/`, `_neutral_v4/` |
| `v3-on-main` | `5206940` | PR [#10](https://github.com/dorakingx/llm-vqc/pull/10) | `outputs/qae_tfim_neutral_v5/` — its **tree is identical to `main`'s**, 0 differing files |

## Group B — connected history, fast-forwarded into `main`

A clean linear chain on top of `main`, each branch a strict superset of the one
above it. `main` was **fast-forwarded** to the tip of the chain, so **all commits
and authorship are preserved as-is** — nothing was squashed or rewritten.

| Branch (deleted) | Tip SHA | Contribution | PR |
|---|---|---|---|
| `claude/qae-experiments-presentation-ba33c0` | `e846c27` | **QAE robustness study (the primary result)** + 2026-09-04 deck | [#11](https://github.com/dorakingx/llm-vqc/pull/11) |
| `analysis/qae-budget-target-20260908` | `d2ef21b` | Validation-only budget audit; froze the B=6 / XXZ B=10 follow-up protocol | [#12](https://github.com/dorakingx/llm-vqc/pull/12) |
| `claude/llm-vqc-min-budget-search-eb95ad` | `9ec31bc` | **Executed** those two boundary cells; report, figures, deck | [#12](https://github.com/dorakingx/llm-vqc/pull/12) |

Verified chain: `main`(`86cf5e2`) → `e846c27` → `d2ef21b` → `9ec31bc`, each an
ancestor of the next.

> **Note on the follow-up specification.** `analysis/qae-budget-target-20260908`
> ended with the commit *"Preserve targeted B6 and XXZ B10 follow-up
> specification as unexecuted"*. Those cells were **subsequently executed** on
> `claude/llm-vqc-min-budget-search-eb95ad` (commit `035e6db`), and their measured
> results are in
> [`outputs/qae_budget_targets_v2_20260908/REPORT.md`](../../outputs/qae_budget_targets_v2_20260908/REPORT.md).
> They are therefore reported as **completed**, not as future work. The work that
> genuinely remains unexecuted is listed in
> [§12 of the final report](../../GSoC2026_FINAL_REPORT.md#12-remaining-work).

## Group C — disconnected history, copied into `archive/`

An author-identity rewrite early in the project split the history in two. These
branches are the **pre-rewrite** side: they share no merge base with `main`, and
they hold real experimental artifacts `main` never had.

They were **not** force-merged — merging unrelated histories to tidy the graph
would have been cosmetic. Instead their unique content (630 files, ~29 MB) was
copied verbatim onto `main` under
[`archive/pre-consolidation/`](../../archive/pre-consolidation), at its original
repository-relative paths, before the branches were deleted.

| Branch (deleted) | Tip SHA | Status |
|---|---|---|
| `experiment/bench-v2-real-llm-cost-minimal` | `6afa91d` | Tip of the line — **source of 592 archived files** |
| `experiment/mini-llm-api-vqc-demo-v1` | `112cf2d` | Second tip — **source of 38 further archived files** |
| `experiment/capacity-controlled-higgs-v1` | `932610b` | ancestor of `6afa91d` |
| `experiment/higgs-data-scale-qualification-v1` | `24d6f1e` | ancestor of `6afa91d` |
| `experiment/free-amplitude-fixed-readout-v1` | `2b9084a` | ancestor of `6afa91d` |
| `research/rigorous-qas-benchmark-v2` | `da6ef34` | ancestor of `6afa91d` |
| `presentation/rebuild-gsoc-benchmark-v2` | `a87a947` | ancestor of `6afa91d` |

The two tips agreed **byte-for-byte on all 137 paths they shared**, so the union
copied to `archive/` is unambiguous. What each archived line contains, and why it
was superseded, is in [`archive/README.md`](../../archive/README.md).

Archived content is **read-only**: it is excluded from lint and from pytest
collection, and reproducing those studies means checking out the SHA above, not
running the copies.

---

## Research lineage (all on `main`)

```
Clifford equivalence-class exploration (origin tooling, llm_vqc/agent.py)
   → LAQS-Bench LLM-vs-classical search
   → capacity-controlled benchmarks T1/T2      outputs/capacity_controlled_*
   → HIGGS task qualification (archived)       archive/… + HIGGS_ARCHIVE_SYNTHESIS.md
   → rigorous QAS benchmark v2 (BLOCKED)       archive/… + BENCHMARK_V2_PROTOCOL.md
   → QAE v2 API verification                   outputs/qae_tfim_api_v2/
   → QAE v3 neutral space (diagnostic)         outputs/qae_tfim_neutral_v3/
   → QAE v4 multi-start (diagnostic)           outputs/qae_tfim_neutral_v4/
   → QAE v5 free-form redesign                 outputs/qae_tfim_neutral_v5/
   → QAE ROBUSTNESS STUDY  ★ PRIMARY           outputs/qae_robustness/
   → minimum-budget-to-target (supporting)     outputs/qae_budget_targets_v2_20260908/
```

**The headline changed at the last step, and that is deliberate.** v5 reported
the *first detected closed-loop advantage*. The robustness study — which reuses
v5 bit-for-bit as its reference cell — then showed that this advantage **does not
generalise**: Closed − Open holds in only 2/6 varied conditions and significantly
**reverses** at 6 and 8 qubits. v5 is therefore preserved as the primary study's
reference cell and as a correct result *at its own operating point*, but it is
**no longer the headline**. What survives everywhere is the *open-loop* semantic
advantage (Open − Random, 6/6).

## Protocol-freezing commits

Each protocol was written, committed and frozen in Git **before** the
corresponding run. This is a commit-order claim anyone can check from a clone;
it is **not** a registration with an external preregistration service, and no
such service was used.

| Protocol | Frozen at | Reachable from `main`? |
|---|---|---|
| `QAE_PROTOCOL_V3.md` | `6620cd4` | No — squash-merged via PR [#8](https://github.com/dorakingx/llm-vqc/pull/8) |
| `QAE_PROTOCOL_V5.md` | `33740dc` | No — squash-merged via PR [#10](https://github.com/dorakingx/llm-vqc/pull/10) |
| `QAE_ROBUSTNESS_PROTOCOL.md` (primary) | [`6933d3f`](https://github.com/dorakingx/llm-vqc/commit/6933d3f) | **Yes** — fast-forwarded, in `git log main` |
| `QAE_BUDGET_TARGET_PROTOCOL.md` | [`f9a8d81`](https://github.com/dorakingx/llm-vqc/commit/f9a8d81) | **Yes** — fast-forwarded, in `git log main` |

The two squash-merged SHAs are **not** in `git log main` and are not recoverable
from a plain clone. They remain viewable on GitHub through their pull request,
whose head ref GitHub retains after branch deletion. The **protocol files
themselves are on `main`** in `docs/research/`, and the squash commit messages
record the lineage — so the protocol-freezing claim is verifiable from `main`
alone; only the individual pre-merge commit object is not.

## Pull requests

Merged: [#1](https://github.com/dorakingx/llm-vqc/pull/1),
[#4](https://github.com/dorakingx/llm-vqc/pull/4),
[#5](https://github.com/dorakingx/llm-vqc/pull/5),
[#6](https://github.com/dorakingx/llm-vqc/pull/6),
[#8](https://github.com/dorakingx/llm-vqc/pull/8),
[#9](https://github.com/dorakingx/llm-vqc/pull/9),
[#10](https://github.com/dorakingx/llm-vqc/pull/10),
[#11](https://github.com/dorakingx/llm-vqc/pull/11),
[#12](https://github.com/dorakingx/llm-vqc/pull/12).

Closed without merging, superseded: [#2](https://github.com/dorakingx/llm-vqc/pull/2),
[#3](https://github.com/dorakingx/llm-vqc/pull/3),
[#7](https://github.com/dorakingx/llm-vqc/pull/7).
