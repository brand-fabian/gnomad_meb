<!-- graft:start -->
## Graft — repo context graph

This repo is indexed in `graft/`: small linked markdown nodes that explain each
system and carry exact file:line spans, kept in sync with the code through git.

For ANY task here — understanding how something works, finding where code lives,
or scoping a change — get context from the graph before grepping or opening
source files. Re-ask freely (it's cheap) and reuse literal identifiers you
already have (symbol, error string, file name) as the query. New to this repo?
Run `graft map` first — a token-budgeted orientation (dir clusters, hubs,
hotspots), no LLM, no key.

- Run `graft ask "<your question>" --source` → ranked nodes with the relevant
  code spans inlined (each hit's ≤8-line crux by default; `--full` for whole
  definitions when the crux isn't enough). Match the tool to the task shape:
  for understanding or editing, the top node IS the answer — cite its
  `covers:` file:line spans and edit straight from `--source`. For
  exhaustive tasks ("every occurrence / every caller of this pattern"), ranked
  results are top-N, not complete — run `graft grep "<literal>"` instead
  (exhaustive over indexed files, grouped by enclosing symbol), falling back
  to raw `grep -rn` only for unindexed files.
- `graft skeleton <file>` → every definition's signature + span, ~10× cheaper
  than reading the file; use it to skim an API surface.
- `graft callers <symbol>` gives precomputed, exact edges — who calls this.
  Add `--direction out` for what it calls, or `--depth N` to walk
  transitively for the full blast radius. For structural questions, skip
  ranking and use this directly.
- Or browse: `graft/INDEX.md` lists every node; follow the links.
- Monorepos and folders of multiple repos rank fairly across sub-projects —
  hits carry `[scope/]` labels naming which one they're from. Narrow with
  `graft ask "<task>" --in <scope>/` once you know where you're working.

If a returned span is truncated ("+N more lines"), open the file at that exact
range before finalizing. Only open source files when a node genuinely lacks a
needed detail, and then at the exact file:line the node points to — never
re-read whole files.

After big code changes, refresh the graph with `graft build` (deterministic,
no API key, $0).
<!-- graft:end -->

## Workflow

Before implementing any changes to the codebase, it is mandatory to first sketch
the proposed changes to the human for review. For review, provide a detailed
breakdown of the suggested code changes and the paths in the software that are
about to be changed and why.

## Coding style

- **No underscore-prefixed globals.** Do not add a `_private` name at module
  scope (a constant, helper or sentinel). A module's public surface is its
  contract, and a leading underscore hides a name that other modules and tests
  still need to reach — commonly a value shared between a decorator and the code
  that reads its output. If a shared constant is genuinely needed, put it in the
  module's `constants.py` (public name, one per concern, with a short docstring
  comment) and import it; otherwise inline it or keep it local to the function.
  Underscore-prefixed *methods* on a class are fine, and test modules keep their
  conventional `_fixtures`/`_helpers`.

## Third-party submodules

`third_party/gnomad_qc`, `third_party/gnomad_methods` and `third_party/genohype`
are git submodules. Initialise them after cloning with
`git submodule update --init --recursive`. The recorded gitlink commit is the
compatibility pin; there is no branch/tag/version configuration.

`genohype` ships no binary in the checkout. Before using `browser run`, build it
from `third_party/genohype` with:

```
cargo build --release --locked --package genohype-cli --bin genohype
```

This produces `third_party/genohype/target/release/genohype`, which
`gnomad_meb.lib.browser.genohype` locates relative to the repo root.

`browser run` writes the browser artifacts under `{project}/browser`; point the
browser's `data_dir` there so its `duckdb` backend finds `variants.parquet/`. The
pipeline does not fingerprint its inputs, so regenerating is the operator's job:
pass `--overwrite` whenever `--mt`, `[browser_export].fields`, the reference
genome, or the genohype checkout changes. A contig-set change (a reference-genome
change) additionally requires removing the old `{project}/browser/variants.parquet/`
directory first, or files from the previous contig set remain in the table.

## Environment caveat: do not run the VDS/hail stack here

Do **not** run `tests/integration/test_vds.py`, the end-to-end suite, or any
other test that spins up the full VDS/hail stack on this machine — it crashes
the host. The system default is Java 25, which hail/Spark cannot drive
(`[JAVA_GATEWAY_EXITED]`), and the integration suite also expects a container
with Java 11 and a provisioned reference root (see
`docs/adr/0002-integration-tests-run-in-a-minimal-container.md`). Run those in
the container; locally stick to `tests/unit` and lint.

## Playbooks (marimo notebooks)

Triage, exploration, and one-off analysis notebooks are marimo `.py` notebooks.
Place them under `playbooks/<group>/`, where `<group>` names a logical group of
notebooks (e.g. `playbooks/bug/` for bug triage). Use short, lowercase group
names, one folder per group. A notebook may ship a sibling helper/worker module
(e.g. a child-process probe) in the same group folder.

## Agent skills

### Issue tracker

Issues and specs live as markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical labels (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
