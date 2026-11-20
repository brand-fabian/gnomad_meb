"""Snakemake pipeline for genome-wide regression using regenie.

This module sets up and executes a snakemake based workflow to compute whole
genome regressions efficiently using regenie. The genome is split into a set
of chunks and processed per-chunk and per-phenotype in parallel using the 
snakemake execution system.
"""

import json
import string
from pathlib import Path
from tempfile import gettempdir

import yaml
from snakemake.cli import args_to_api, parse_args
from snakemake.exceptions import print_exception

from gnomad_meb.config import get_config
from gnomad_meb.lib.gwas.constants import REGENIE_VERSION_FILENAME
from gnomad_meb.logging import get_logger

SNAKEFILE_TEMPLATE = (Path(__file__).parent / "Snakefile.template").resolve()
SNAKEFILE_BASE = Path(__file__).parent.resolve()

def _make_config(
    bed: str,
    covariates: str,
    phenotype: str,
    annotation: str,
    variant_sets: str,
    aaf: str,
    effect_masks: str,
    n_chunks: int,
    response_kinds: dict[str, str] | None = None,
    target: str | Path | None = None,
    output_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
) -> Path:
    cfg = get_config()
    logger = get_logger()

    if target is None:
        target = Path(gettempdir()) / "config.yaml"
    elif isinstance(target, str):
        target = Path(target)

    logger.debug(f"creating snakemake pipeline configuration: target = {target}")
    with open(target, "w") as out_f:
        out_f.write(
            yaml.dump(
                {
                    "output_dir": str(output_dir or cfg.gwas.output_dir),
                    "wrapper_dir": str(cfg.gwas.wrapper_dir),
                    "log_dir": str(log_dir or cfg.gwas.log_dir),
                    "covariates": str(covariates),
                    "phenotype": str(phenotype),
                    "annotation": str(annotation),
                    "variant_sets": str(variant_sets),
                    "aaf": str(aaf),
                    "effect_masks": str(effect_masks),
                    "bed": str(bed),
                    "project": str(cfg.project),
                    "n_chunks": n_chunks,
                    "min_case_count": cfg.gwas.min_case_count,
                    "response_kinds": response_kinds or {},
                    "min_mac": cfg.gwas.min_mac,
                    "min_geno_frac": cfg.gwas.min_geno_frac,
                    "min_maf": cfg.gwas.min_maf,
                    "hwe": cfg.gwas.hwe,
                    "snps_only": "True" if cfg.gwas.snps_only else "False",
                }
            )
        )
    return target

def _make_snakefile(
    config: str | Path,
    target: str | Path | None = None,
    source: Path = SNAKEFILE_TEMPLATE,
) -> Path:
    logger = get_logger()

    if target is None:
        target = Path(gettempdir()) / "Snakefile"
    elif isinstance(target, str):
        target = Path(target)

    with open(source) as in_f:
        source_t = string.Template(in_f.read())

    logger.debug(f"creating snakefile at {target}")
    with open(target, "w") as out_f:
        out_f.write(
            source_t.safe_substitute(
                {
                    "configfile": str(config),
                    "gwas_base": SNAKEFILE_BASE,
                }
            )
        )
    return target

def _detect_regenie_version(working_directory: Path) -> dict[str, str] | None:
    """Detect the regenie version Snakemake deployed the pipeline with.

    Snakemake writes each conda rule's environment spec to
    ``{working_directory}/.snakemake/conda/<hash>.yaml`` and deploys it to the
    sibling ``<hash>/`` prefix. Find the spec that requires ``regenie`` and read
    the installed package's ``conda-meta/regenie-<version>-<build>.json`` record.

    Returns the record's ``version``/``build``/``url``, or ``None`` when no
    regenie environment was deployed.
    """
    logger = get_logger()
    conda_root = working_directory / ".snakemake" / "conda"

    for spec_path in sorted(conda_root.glob("*.yaml")):
        spec = yaml.safe_load(spec_path.read_text()) or {}
        dependencies = spec.get("dependencies", [])
        if not any(
            str(dependency).startswith("regenie") for dependency in dependencies
        ):
            continue

        records = sorted(
            (conda_root / spec_path.stem / "conda-meta").glob("regenie-*.json")
        )
        if not records:
            logger.warning(
                f"regenie conda environment without a package record: {spec_path}"
            )
            continue

        record = json.loads(records[-1].read_text())
        return {
            "version": record["version"],
            "build": record["build"],
            "url": record.get("url", ""),
        }
    return None


def regenie(
    bed: str,
    covariates: str,
    phenotype: str,
    annotation: str,
    variant_sets: str,
    aaf: str,
    effect_masks: str,
    snakemake_args: tuple[str, ...],
    n_chunks: int = 1,
    response_kinds: dict[str, str] | None = None,
    output_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
):
    cfg = get_config()
    logger = get_logger()

    working_directory = Path(cfg.scratch_dir) / "gwas"
    working_directory.mkdir(parents=True, exist_ok=True)

    cfg_p = _make_config(bed, covariates, phenotype, annotation,
                         variant_sets, aaf, effect_masks,
                         n_chunks,
                         response_kinds=response_kinds,
                         target=working_directory / "config.yaml",
                         output_dir=output_dir, log_dir=log_dir)
    smk_p = _make_snakefile(cfg_p, target=working_directory / "Snakefile")

    if "--snakefile" in snakemake_args:
        logger.warning(
            "found --snakefile arg in snakemake arguments for this workflow"
            ", this wil be overriden"
        )
        idx = snakemake_args.index("--snakefile")
        snakemake_args = (
            *snakemake_args[:idx],
            *snakemake_args[(idx + 2) :],
        )

    if "--directory" in snakemake_args:
        logger.warning(
            "found --directory arg in snakemake arguments for this workflow"
            ", this wil be overriden"
        )
        idx = snakemake_args.index("--directory")
        snakemake_args = (
            *snakemake_args[:idx],
            *snakemake_args[(idx + 2) :],
        )

    # Same as in VQSR Workflow:
    snakemake_args = (
        *snakemake_args,
        "--snakefile",
        str(smk_p),
        # Run Snakemake from the workflow's scratch working directory so its
        # relative paths, locks and logs stay out of the caller's cwd.
        "--directory",
        str(working_directory),
    )
    logger.debug(f"got snakemake argument list: {snakemake_args}")

    # Lifted from the snakemake main function with some additions
    # for the integration between the different pipelines:
    # https://github.com/snakemake/snakemake/blob/main/src/snakemake/cli.py#L2246
    try:
        logger.info(f"starting gwas workflow with genotype: bed = {bed}")
        parser, args = parse_args(snakemake_args)
        success = args_to_api(args, parser)
    except Exception as e:
        logger.error("error executing snakemake workflow")
        print_exception(e)
        return False

    if success:
        # Write the regenie provenance next to the output, so the load stage can
        # read the version back (the result files mark success).
        regenie_version = _detect_regenie_version(working_directory)
        if regenie_version is None:
            logger.warning(
                "could not detect the regenie version from the deployed conda "
                "environments; no regenie provenance will be written"
            )
        else:
            provenance = (
                Path(output_dir or cfg.gwas.output_dir) / REGENIE_VERSION_FILENAME
            )
            provenance.parent.mkdir(parents=True, exist_ok=True)
            provenance.write_text(json.dumps(regenie_version, indent=2))
            logger.debug(f"wrote regenie version provenance to {provenance}")
    return success
