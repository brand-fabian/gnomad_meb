#!/usr/bin/env python3
"""Submit the Annotation pipeline as a per-split Slurm job array.

Rationale
---------
``AnnotationPipeline.prepare`` fans the run out into one independent chain per
VDS split (``task-0`` ... ``task-N-1``); the chains share only whole-stage
resource imports and have no cross-task edges. This driver therefore emits a job
array with one task per split, each selecting that split's chain with
``gmeb run MANIFEST --select '*.task-<index>'``, followed by a small terminal
``merge`` job that depends on the array.

The shell entry point stays ``slurm/annotation.sh`` (unchanged interface) and
simply execs this driver.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from datetime import date
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "slurm" / "config.sh"

#: Config variables sourced from the bash config file and used verbatim.
CONFIG_KEYS = (
    "GNOMAD_MEB_CONDA_ENV",
    "GNOMAD_CONFIGURATION",
    "DEPENDENCY_TRACKING",
    "OUTPUT_DIR",
    "SBATCH_SCRIPT_DIR",
    "LOG_DIR",
    "SCRATCH_DIR",
    "ANNOTATION_VDS",
    "ANNOTATION_PARTITION",
    "ANNOTATION_TIME",
    "ANNOTATION_MEM_PER_CPU",
    "ANNOTATION_CPUS_PER_TASK",
    "ANNOTATION_NTASKS",
    "SPARK_SOURCE",
    "HAIL_VERSION",
    "SPARK_VERSION",
    "SPARK_MODULE",
    "CONDA_MODULE",
    "CONDA_INIT",
    "CONDA_ENV",
    "IPOIB_DOMAIN",
    "SPARK_INTERFACE",
    "MASTER_MEMORY",
    "MASTER_CORES",
)

#: Config variables that may legitimately be empty (no shell default needed).
OPTIONAL_KEYS = frozenset(
    {"MASTER_MEMORY", "MASTER_CORES", "CONDA_INIT", "IPOIB_DOMAIN", "SPARK_INTERFACE"}
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the ``annotation.sh`` command line (a strict superset of it)."""
    parser = argparse.ArgumentParser(
        prog="annotation.py",
        description=(
            "Submit the gnomad_meb annotation pipeline as a per-split job array."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"bash configuration file (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Generate the job scripts but do not submit them to Slurm.",
    )
    parser.add_argument(
        "-d",
        "--dependency",
        default="",
        help="Comma-separated job ids to depend the array on (afterany | afterok).",
    )
    parser.add_argument(
        "-j",
        "--max-jobs",
        default=100,
        help="Max jobs per slurm job array.",
        type=int,
    )
    return parser.parse_args(argv)


def load_config(path: str) -> dict[str, str]:
    """Source the bash config in a subshell and return the variables we use.

    Sourcing the file keeps the driver honest: the config is bash, so bash is
    the parser. Each requested variable is printed on its own NUL-terminated
    line as ``key<TAB>value``; a variable that is neither set nor optional is
    reported so the caller fails loudly instead of building a broken script.
    """
    # Print key<TAB>value, NUL-terminated, for every variable we consume.
    script = 'source "$1" >/dev/null 2>&1 || exit 3\n' + "".join(
        f'printf "%s\\t%s\\0" {shlex.quote(key)} "${{{key}-}}"\n' for key in CONFIG_KEYS
    )

    proc = subprocess.run(
        ["bash", "-c", script, "_", path],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"could not source configuration file: {path}")

    config: dict[str, str] = {}
    for entry in proc.stdout.split("\0"):
        if not entry:
            continue
        key, _, value = entry.partition("\t")
        config[key] = value

    missing = [
        key for key in CONFIG_KEYS if not config.get(key) and key not in OPTIONAL_KEYS
    ]
    if missing:
        raise SystemExit(
            f"configuration {path} is missing required variables: {missing}"
        )
    return config


def read_task_names(vds_metadata: str) -> list[str]:
    """The split task names from a VDS handle's metadata file, without hail.

    ``VariantDatasetParameters.tasks[i].name`` is exactly the ``task-<n>``
    suffix the annotation pipeline appends to each per-task step id.
    """
    path = Path(vds_metadata)
    if not path.is_file():
        raise SystemExit(f"annotation VDS handle not found: {path}")
    params = json.loads(path.read_text(encoding="utf-8"))
    names = [task["name"] for task in params["tasks"]]
    if not names:
        raise SystemExit(f"annotation VDS handle declares no tasks: {path}")
    return names


def generate_manifest(config: dict[str, str], manifest_path: Path) -> None:
    """Write the run manifest for the annotation invocation (``--dry-run``).

    ``--dry-run`` builds the annotation pipeline, which reads the VDS handle's
    task list in ``prepare()``, and prints the manifest with every per-task step.
    It does not initialize hail and does not read data.
    """
    cmd = [
        "uv",
        "run",
        "--project",
        str(REPO_ROOT),
        "python3",
        str(REPO_ROOT / "main.py"),
        "--config",
        str(REPO_ROOT / "config" / config["GNOMAD_CONFIGURATION"]),
        "--verbose",
        "debug",
        "--dry-run",
        "--output-format",
        "json",
        "annotation",
        "run",
        "--vds",
        config["ANNOTATION_VDS"],
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(
            "annotation --dry-run failed; the manifest could not be generated."
        )
    manifest_path.write_text(proc.stdout, encoding="utf-8")


def verify_manifest(manifest_path: Path, task_names: list[str]) -> None:
    """Fail on the login node if the manifest lacks a step for any task.

    The array index maps to ``task-<index>``; if the manifest's step ids do not
    carry ``annotate_resources.task-<i>`` for every split, the selection would
    silently run nothing and 96 jobs would be wasted. Validation here turns that
    into an immediate, cheap error.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    step_ids = {step["name"] for step in manifest.get("steps") or []}
    missing = [
        name for name in task_names if f"annotate_resources.{name}" not in step_ids
    ]
    if missing:
        raise SystemExit(
            "manifest is missing per-task steps; refusing to submit: "
            f"missing = {missing}"
        )
    if "merge" not in step_ids:
        raise SystemExit("manifest is missing the terminal 'merge' step.")


def annotation_command(n_tasks_per_job: int, n_tasks: int, manifest_path: Path) -> str:
    """The ``gmeb run`` line shared by the array and merge wrappers."""
    return dedent(f"""\
        ntasks_per_job={n_tasks_per_job}
        ntasks={n_tasks}
        start_task_id=$(( ntasks_per_job * SLURM_ARRAY_TASK_ID ))
        end_task_id=$(( start_task_id + ntasks_per_job ))
        (( end_task_id > ntasks )) && end_task_id=$ntasks

        select_opt=()
        for (( task_id = start_task_id; task_id < end_task_id; task_id++ )); do
            select_opt+=(--select "*.task-${{task_id}}")
        done

        uv run --project "{REPO_ROOT}" python3 "{REPO_ROOT / "main.py"}" --verbose debug \\
            run "{manifest_path}" "${{select_opt[@]}}"
    """)


def merge_command(manifest_path: Path) -> str:
    """The ``gmeb run`` command targeting merging of individual VDS shards."""
    return dedent(f"""\
        uv run --project "{REPO_ROOT}" python3 "{REPO_ROOT / "main.py"}" \\
        run "{manifest_path}" --select "annotation.merge"
    """)


def spark_block(config: dict[str, str], wrapper_path: Path) -> str:
    """The ``spark.sh`` invocation shared by every generated sbatch script."""
    iboption = (
        f'--ipoib-domain "{config["IPOIB_DOMAIN"]}" ' if config["IPOIB_DOMAIN"] else ""
    )
    interfaceoption = (
        f'--spark-interface "{config["SPARK_INTERFACE"]}" '
        if config["SPARK_INTERFACE"]
        else ""
    )
    master_memory = config["MASTER_MEMORY"] or "8000"
    master_cores = config["MASTER_CORES"] or config["ANNOTATION_CPUS_PER_TASK"]
    return dedent(f"""\
        bash "{config["SPARK_SOURCE"]}/spark.sh" \\
            --hail-version "{config["HAIL_VERSION"]}" \\
            {interfaceoption}{iboption}--master-mem "{master_memory}" \\
            --master-cores "{master_cores}" \\
            --spark-version "{config["SPARK_VERSION"]}" \\
            --spark-module "{config["SPARK_MODULE"]}" \\
            --conda-module "{config["CONDA_MODULE"]}" \\
            --scratch-dir "{config["SCRATCH_DIR"]}" \\
            --conda-init "{config["CONDA_INIT"]}" \\
            --conda-env "{config["CONDA_ENV"]}" \\
            --install-s3-connector \\
            -Xis "{wrapper_path}"
    """)


def write_wrapper(path: Path, config: dict[str, str], body: str) -> None:
    """Write a wrapper script: module/conda setup, then the run body."""
    path.write_text(
        dedent(f"""\
            #!/bin/bash

            # Load Java (required for hail) and the conda module,
            # then initialize the shell
            module load "{config["CONDA_MODULE"]}" Java/11
            test -n "$CONDA_INIT" && eval "$CONDA_INIT"
            conda activate "{config["GNOMAD_MEB_CONDA_ENV"]}"

            {body}
        """),
        encoding="utf-8",
    )
    path.chmod(0o755)


def write_sbatch(
    path: Path,
    config: dict[str, str],
    name: str,
    log_pattern: str,
    wrapper_path: Path,
    *,
    array: str | None = None,
) -> None:
    """Write one generated sbatch script around the shared spark block."""
    array_line = f'#SBATCH --array="{array}"\n' if array else ""
    # ``spark_block`` is dynamic and unindented; splice it in after ``dedent``
    # so it cannot skew the common prefix the header relies on.
    header = dedent(f"""\
        #!/bin/bash
        #SBATCH --partition="{config["ANNOTATION_PARTITION"]}"
        #SBATCH --time="{config["ANNOTATION_TIME"]}"
        #SBATCH --mem-per-cpu="{config["ANNOTATION_MEM_PER_CPU"]}"
        #SBATCH --cpus-per-task="{config["ANNOTATION_CPUS_PER_TASK"]}"
        #SBATCH --ntasks="{config["ANNOTATION_NTASKS"]}"
    """)
    directives = array_line + dedent(f"""\
            #SBATCH --job-name="{name}"
            #SBATCH --output="{config["LOG_DIR"]}/{log_pattern}"
            #SBATCH --error="{config["LOG_DIR"]}/{log_pattern}"
        """)
    path.write_text(
        header + directives + f"\n{spark_block(config, wrapper_path)}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def submit(sbatch_path: Path, dependency: str, dry_run: bool) -> str:
    """``sbatch`` a generated script; return the job id (empty when dry-running)."""
    if dry_run:
        print(f"[dryrun] skipping job submission: {sbatch_path.name}")
        return ""
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd.append(str(sbatch_path))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"sbatch failed for {sbatch_path.name}")
    jobid = proc.stdout.strip().split(";")[0]
    print(f"submitted {sbatch_path.name}: {jobid}")
    return jobid


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not Path(args.config).is_file():
        raise SystemExit(f"configuration file {args.config} does not exist.")

    if args.max_jobs < 1:
        raise SystemExit("can not create arrays with less than one jobs.")

    print(f"using configuration {args.config}")
    config = load_config(args.config)

    experiment = os.environ.get("EXPERIMENT_NAME", "vds")
    config["OUTPUT_DIR"] = f"{config['OUTPUT_DIR']}/{experiment}"
    config["LOG_DIR"] = f"{config['LOG_DIR']}/{experiment}"
    config["SBATCH_SCRIPT_DIR"] = f"{config['SBATCH_SCRIPT_DIR']}/{experiment}"
    for key in ("OUTPUT_DIR", "SBATCH_SCRIPT_DIR", "LOG_DIR", "SCRATCH_DIR"):
        value = config[key]
        if value.startswith("s3"):
            print(f"got s3 url: {value}")
        else:
            Path(value).mkdir(parents=True, exist_ok=True)

    task_names = read_task_names(config["ANNOTATION_VDS"])
    n_tasks = len(task_names)
    print(f"annotation run has {n_tasks} split tasks")

    expected_task_names = [f"task-{i}" for i in range(n_tasks)]
    if task_names != expected_task_names:
        raise SystemExit(
            "annotation task names do not match indexed array selection: "
            f"expected {expected_task_names}, got {task_names}"
        )

    stamp = date.today().isoformat()
    prefix = f"{experiment}.annotation.{stamp}"
    manifest_path = Path(config["SBATCH_SCRIPT_DIR"]) / f"{prefix}.manifest.json"
    generate_manifest(config, manifest_path)
    verify_manifest(manifest_path, task_names)
    print(f"wrote run manifest: {manifest_path}")

    n_tasks_per_job = math.ceil(n_tasks / args.max_jobs)
    n_array_jobs = math.ceil(n_tasks / n_tasks_per_job)
    print(
        f"generating task array: n_array_jobs = {n_array_jobs}, "
        f"n_tasks_per_job = {n_tasks_per_job}"
    )

    array_name = f"{prefix}.array"
    array_script = Path(config["SBATCH_SCRIPT_DIR"]) / f"{array_name}.sh"
    array_wrapper = Path(config["SBATCH_SCRIPT_DIR"]) / f"{array_name}.wrapper.sh"
    write_wrapper(
        array_wrapper,
        config,
        annotation_command(n_tasks_per_job, n_tasks, manifest_path),
    )
    write_sbatch(
        array_script,
        config,
        array_name,
        f"{prefix}.%A_%a.log",
        array_wrapper,
        array=f"0-{n_array_jobs - 1}",
    )

    dependency = args.dependency
    if dependency:
        dependency = f"{config['DEPENDENCY_TRACKING']}:{dependency.replace(',', ':')}"
    array_id = submit(array_script, dependency, args.dry_run)

    merge_name = f"{prefix}.merge"
    merge_script = Path(config["SBATCH_SCRIPT_DIR"]) / f"{merge_name}.sh"
    merge_wrapper = Path(config["SBATCH_SCRIPT_DIR"]) / f"{merge_name}.wrapper.sh"
    write_wrapper(merge_wrapper, config, merge_command(manifest_path))
    write_sbatch(
        merge_script,
        config,
        merge_name,
        f"{merge_name}.%j.log",
        merge_wrapper,
    )

    merge_dependency = f"afterok:{array_id}" if array_id else ""
    if not array_id and dependency:
        # Dry-run with an explicit dependency: keep it on the merge job so the
        # generated scripts still reflect the intended chain.
        merge_dependency = dependency
    merge_id = submit(merge_script, merge_dependency, args.dry_run)

    if merge_id:
        print(f"\t--> {merge_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
