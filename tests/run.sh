#!/usr/bin/env bash
#
# Integration-test entrypoint. Starts the test container and runs a pytest
# session inside it. See
# docs/adr/0002-integration-tests-run-in-a-minimal-container.md and
# docs/adr/0001-integration-tests-use-subprocess-cli-seam.md.
#
# Usage: tests/run.sh [pytest args...]
#
# With no arguments the fast development subset runs:
#     -m "integration and not vqsr"
# Pass explicit arguments to override, e.g.
#     tests/run.sh -m integration              # everything, including VQSR
#
# The suite runs serially. Reuse between runs is decided by production's
# checkpoint status, not by the harness.
#
# The repository, the fixture data, the reference-resource root and the run
# directory are bound at identical paths inside and outside the container, so
# the rendered test config is valid on both sides. Existing GMEB_-prefixed
# variables are passed through; HOME is set inside the container so no ambient
# configuration leaks in.
#
# Resume across runs: by default each invocation uses a fresh temporary run
# directory, deleted on exit, so every step is rebuilt. Export
# GMEB_TEST_RUN_DIR to pin a directory that is kept between runs; production
# then skips every *present* checkpoint it already declares (a `@checkpoint`
# step is judged by its CheckpointedCallable status), so only the changed steps
# rerun — handy for iterating on a downstream stage. The reuse signal reads the
# run directory, not the code that produced it, so delete the directory (or
# unset GMEB_TEST_RUN_DIR) after changing any upstream step.
#
# Recovery from a crashed run (operator-only): a `@declares_step` step (VQSR)
# has no completion contract, so its reuse verdict is a bare "the artifact
# exists" — an artifact truncated by a crashed run is therefore reused. To force
# a step to recompute, run the workflow by hand with `--overwrite`
# (`gnomad-workflow ... --overwrite`); the harness never passes `--overwrite`.
# See docs/adr/0022-integration-suite-uses-production-checkpoint-status.md.
#
# Filter prerequisite: the composite `gnomad-workflow` ends with the Filter
# stage, whose `annotate_constraints_table` step needs the reference root's
# constraint-metrics table, referenced as `[filter.resources].constraints` in
# the render template (see tests/integration/data/README.md).
#
# Environment:
#   GMEB_TEST_DATA_DIR  fixture dataset root     (default: <repo>/data)
#   GMEB_TEST_REF_ROOT  reference-resource root  (required)
#   GMEB_TEST_IMAGE     Singularity image        (default: tests/gnomad-meb-test.sif)
#   GMEB_TEST_RUNTIME   singularity | apptainer
#   GMEB_TEST_RUN_DIR   persistent run directory (default: fresh temp dir, removed on exit)
#   GMEB_TEST_CACHE_DIR persistent uv cache/venv (default: ${XDG_CACHE_HOME:-$HOME/.cache}/gnomad-meb-tests)
#
# Example:
#   GMEB_TEST_VEP_ROOT=/data/volumes/xcat-login/homedirs/brand/Projects/gnomad/scratch \
#       GMEB_TEST_CACHE_DIR=$HOME/.cache \
#       GMEB_TEST_RUNTIME=apptainer \
#       GMEB_TEST_REF_ROOT=/data/volumes/xcat-login/scratch/brand \
#       bash tests/run.sh
#
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(realpath "$SCRIPT_DIR/..")

fail() { echo "error: $*" >&2; exit 1; }
require_dir() { [[ -d "$1" ]] || fail "missing $2: '$1'"; }

# --- container runtime -----------------------------------------------------
RUNTIME="${GMEB_TEST_RUNTIME:-}"
if [[ -z "$RUNTIME" ]]; then
    for c in singularity apptainer; do
        if command -v "$c" >/dev/null 2>&1; then RUNTIME="$c"; break; fi
    done
fi
[[ -n "$RUNTIME" ]] || fail "no singularity/apptainer runtime found; set GMEB_TEST_RUNTIME"

# --- inputs (hard errors naming the missing path) --------------------------
DATA_DIR=$(realpath "${GMEB_TEST_DATA_DIR:-$REPO_ROOT/data}")

[[ -n "${GMEB_TEST_REF_ROOT:-}" ]] || fail "GMEB_TEST_REF_ROOT is not set (reference-resource root)"
REF_ROOT=$(realpath "$GMEB_TEST_REF_ROOT")

[[ -n "${GMEB_TEST_VEP_ROOT:-}" ]] || fail "GMEB_TEST_VEP_ROOT is not set (vep resources)"
VEP_ROOT=$(realpath "$GMEB_TEST_VEP_ROOT")

IMAGE="${GMEB_TEST_IMAGE:-$SCRIPT_DIR/gnomad-meb-test.sif}"

require_dir "$REPO_ROOT" "repository root"
require_dir "$DATA_DIR" "test data directory (GMEB_TEST_DATA_DIR)"
require_dir "$REF_ROOT" "reference-resource root (GMEB_TEST_REF_ROOT)"
[[ -f "$IMAGE" ]] || fail "missing container image: '$IMAGE' (build it with tests/build.sh or set GMEB_TEST_IMAGE)"

# --- run directory ---------------------------------------------------------
# A caller-provided GMEB_TEST_RUN_DIR is used as-is and kept across runs, so a
# previous run's completed artifacts are reused. Otherwise a fresh temporary
# directory is created and removed on exit (the hermetic default).
if [[ -n "${GMEB_TEST_RUN_DIR:-}" ]]; then
    mkdir -p "$GMEB_TEST_RUN_DIR"
    RUN_DIR=$(realpath "$GMEB_TEST_RUN_DIR")
else
    RUN_DIR=$(mktemp -d "${TMPDIR:-/tmp}/gnomad-meb-tests.XXXXXX")
    cleanup() { rm -rf "$RUN_DIR"; }
    trap cleanup EXIT
fi
mkdir -p "$RUN_DIR/home"

# --- persistent uv cache/venv ----------------------------------------------
# uv's download cache and the project venv live outside the run directory and
# are reused across runs, regardless of whether the run directory itself is
# temporary or pinned via GMEB_TEST_RUN_DIR. The path is bind-mounted at an
# identical location inside and outside the container, so the venv's recorded
# paths stay valid. uv re-syncs the venv automatically when
# pyproject.toml/uv.lock change.
CACHE_ROOT="${GMEB_TEST_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/gnomad-meb-tests}"
mkdir -p "$CACHE_ROOT/uv-cache" "$CACHE_ROOT/venv"

# --- environment -----------------------------------------------------------
# Pass every GMEB_-prefixed host variable through, then pin the test variables.
PINNED_ENV=(
    "GMEB_TEST_DATA_DIR=$DATA_DIR"
    "GMEB_TEST_REF_ROOT=$REF_ROOT"
    "GMEB_TEST_VEP_ROOT=$VEP_ROOT"
    "GMEB_TEST_RUN_DIR=$RUN_DIR"
    "UV_PROJECT_ENVIRONMENT=$CACHE_ROOT/venv"
    "UV_CACHE_DIR=$CACHE_ROOT/uv-cache"
)

ENV_ARGS=()
while IFS= read -r name; do
    [[ "$name" == GMEB_* ]] || continue
    case "$name" in
        GMEB_TEST_DATA_DIR | GMEB_TEST_REF_ROOT | GMEB_TEST_RUN_DIR | GMEB_TEST_CACHE_DIR | GMEB_TEST_VEP_ROOT) continue ;;
    esac
    ENV_ARGS+=(--env "$name=${!name}")
done < <(compgen -e)

for assignment in "${PINNED_ENV[@]}"; do
    ENV_ARGS+=(--env "$assignment")
done

BIND_ARGS=(
    --bind "$REPO_ROOT:$REPO_ROOT"
    --bind "$DATA_DIR:$DATA_DIR"
    --bind "$REF_ROOT:$REF_ROOT"
    --bind "$VEP_ROOT:$VEP_ROOT"
    --bind "$RUN_DIR:$RUN_DIR"
    --bind "$CACHE_ROOT:$CACHE_ROOT"
)

# --- pytest arguments ------------------------------------------------------
if [[ $# -eq 0 ]]; then
    PYTEST_ARGS=(-m "integration and not vqsr")
else
    PYTEST_ARGS=("$@")
fi

exec "$RUNTIME" exec \
    "${BIND_ARGS[@]}" \
    --home "$RUN_DIR/home" \
    --pwd "$REPO_ROOT" \
    "${ENV_ARGS[@]}" \
    "$IMAGE" \
    uv run --project "$REPO_ROOT" pytest "${PYTEST_ARGS[@]}"
