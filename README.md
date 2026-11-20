# gnomad-pipelines

Adaption and configuration of gnomad-like processing pipelines for local
vcf and gvcf data.

## Dependencies

External code lives in git submodules under `third_party/`
(`gnomad_qc`, `gnomad_methods`, `genohype`). Initialise them with
`git submodule update --init --recursive`. The recorded gitlink commit is the
compatibility pin.

The `genohype` binary is not checked in. Before `browser run`, build it from
`third_party/genohype`:

```
cargo build --release --locked --package genohype-cli --bin genohype
```

## Browser export

`browser run` writes the browser's variant artifacts under `{project}/browser`:
the sites-only `sites.ht` for the browser's `hail` backend, and the per-contig
`variants.parquet/locus.contig=<contig>.parquet` dataset. Point the browser's
`data_dir` at `{project}/browser` so its `duckdb` backend finds
`variants.parquet/`.

The pipeline does not fingerprint its inputs, so regenerating after a change is
the operator's job:

- Pass `--overwrite` whenever `--mt`, `[browser_export].fields`, the reference
  genome, or the genohype checkout changes. It recomputes every contig step.
- When the contig set itself changes (a reference-genome change), also remove the
  old `{project}/browser/variants.parquet/` directory first. Otherwise files from
  the previous contig set remain and DuckDB reads them as part of the table.

Loadability through the browser's `duckdb` backend is an explicit,
container-gated acceptance check (deferred from unit CI): the recorded genohype
binary runs on a small real Hail table, then a region query and a
variant-detail query run through gnomad-browser-lite's DuckDB backend. Unit CI
mocks genohype and does not claim that criterion.