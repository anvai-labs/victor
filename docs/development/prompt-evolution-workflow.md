# Prompt Evolution Workflow

How an evolved system-prompt section goes from a locally-learned candidate to a
shipped change in Victor's source. This is the operational companion to
[FEP-0025](https://github.com/anvai-labs/victor/blob/main/feps/fep-0025-prompt-evolution-as-controlled-experiment.md)
(the design) and the
[2026-07-25 audit](https://github.com/anvai-labs/victor/blob/main/docs/analysis/2026-07-25-prompt-evolution-audit.md)
(the measurements).

## The two places evidence and prompts live

Prompt evolution is **local** by default: as Victor runs, the GEPA prompt
optimizer accumulates candidate rewrites of evolvable system-prompt sections in
one operator's SQLite file. Shipping an improvement means moving a vetted
candidate's text into version-controlled source, which is reviewed and installed
with the package.

| | Path | Role |
|---|---|---|
| **Evidence (local)** | `~/.victor/victor.db` → `agent_prompt_candidate` | Evolved candidates + Thompson/benchmark scores. Rebuildable, per-machine. |
| **Shipped prompts (source)** | `victor/agent/prompt_section_texts.py` | The prompts everyone gets. Version-controlled, reviewed, released. |

Two surfaces sit at the two ends of this pipeline. They do **not** overlap.

## Two surfaces

### `/prompt-optimize` — generate & observe (runtime)

An in-session slash command that runs and inspects evolution against the live
learner and the local evidence DB. It never edits source.

| Command | Does |
|---|---|
| `/prompt-optimize` | Run an evolution cycle over all evolvable sections |
| `/prompt-optimize <SECTION>` | Evolve one section (e.g. `ASI`, `GROUNDING_RULES`) |
| `/prompt-optimize --status` | Show current candidates and their scores |
| `/prompt-optimize --pareto` | Show the Pareto frontier |
| `/prompt-optimize --show <SECTION>` / `--diff <SECTION>` | Inspect a section's current/evolved text |
| `/prompt-optimize --tier [economic\|balanced\|performance]` | Show or set the GEPA reflection model tier |

Use it to *drive and watch* evolution while working. The candidates it produces
stay in `~/.victor/victor.db`.

### `scripts/prompt_candidates.py` — audit & ship (maintainer)

An offline CLI for a repo maintainer. It reads the local evidence DB and can
write the shipped section module. Because promotion produces a **code change**
(to review and PR), it lives here rather than in an interactive session.

```bash
python scripts/prompt_candidates.py audit            # verdict per candidate vs the shipped baseline
python scripts/prompt_candidates.py show <hash>      # full text + diff vs shipped baseline
python scripts/prompt_candidates.py export <hash>    # paste-ready Python literal
python scripts/prompt_candidates.py propose <SECTION># register a hand-written candidate for measurement
python scripts/prompt_candidates.py promote <hash>   # write it into prompt_section_texts.py (dry-run)
python scripts/prompt_candidates.py purge --apply    # drop rejected candidates (backs up first)
```

`audit` classifies each candidate against the shipped baseline; only a candidate
that has **beaten** the shipped prompt on a paired benchmark run is a `PROMOTE`.
Anything that merely *differs* is a proposal, not an improvement.

The DB defaults to `~/.victor/victor.db`; override with `VICTOR_DB` or `--db`.

## Shipping a candidate (runbook)

```bash
# 1. Run / observe evolution in a session (optional — candidates also accrue
#    passively as Victor runs).
#    /prompt-optimize
#    /prompt-optimize --status

# 2. See what is worth shipping.
python scripts/prompt_candidates.py audit

# 3. Inspect the winner and its diff against the shipped prompt.
python scripts/prompt_candidates.py show <hash>

# 4. Dry-run the promotion (prints the source diff; writes nothing).
python scripts/prompt_candidates.py promote <hash>

# 5. Apply it — writes victor/agent/prompt_section_texts.py.
python scripts/prompt_candidates.py promote <hash> --apply

# 6. Verify before committing.
python -c 'import victor.agent.prompt_section_texts'   # module still imports
make test-quick                                        # prompt/section tests pass

# 7. Open a PR (feature -> develop), review the diff by hand, and merge.
```

### Safety gates

- **Dry-run by default.** `promote` prints a diff and writes nothing until
  `--apply`.
- **Evidence gate.** `promote` refuses a candidate that is not `PROMOTE`
  (i.e. has not beaten the shipped prompt on a paired benchmark run). `--force`
  overrides only after you have reviewed the diff by hand.
- **Marker safety.** `COMPLETION_GUIDANCE` is an f-string that interpolates the
  completion markers. Promotion re-templatizes rendered marker values back into
  `{PLACEHOLDER}` form and **refuses** rather than hardcode a marker into source
  (which would quietly end `completion_markers.py`'s role as the single
  definition of those tokens). This codegen lives in, and is tested by,
  `victor/framework/rl/prompt_promotion.py`.
- **Reversible purge.** `purge` copies rows to a timestamped backup table before
  deleting, and does nothing without `--apply`.

## Where this fits

`/prompt-optimize` is *generate-and-observe*; the script is the reviewed bridge
that carries a measured winner into the shipped prompts (FEP-0025's "promotion
into source", closing the "nothing reaches other users" gap). Promotion is
deliberately human-reviewed: the repo is the artifact, the local DB is only
evidence.

## Related

- [FEP-0017 — prompt-optimization reward loop](https://github.com/anvai-labs/victor/blob/main/feps/fep-0017-prompt-optimization-reward-loop.md)
- [FEP-0025 — prompt evolution as a controlled experiment](https://github.com/anvai-labs/victor/blob/main/feps/fep-0025-prompt-evolution-as-controlled-experiment.md)
- [ADR-027 — prompt-optimization strategy fidelity](../architecture/adr/027-prompt-optimization-strategy-fidelity.md)
