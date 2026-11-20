#!/usr/bin/env bash
#
# Build the integration-test container image (a Singularity .sif) from the
# committed OCI definition tests/Dockerfile. See
# docs/adr/0002-integration-tests-run-in-a-minimal-container.md.
#
# Usage: tests/build.sh [output.sif]
#
# Environment:
#   GMEB_TEST_IMAGE    output image path   (default: tests/gnomad-meb-test.sif)
#   GMEB_TEST_RUNTIME  singularity|apptainer
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

fail() { echo "error: $*" >&2; exit 1; }

IMAGE="${1:-${GMEB_TEST_IMAGE:-$SCRIPT_DIR/gnomad-meb-test.sif}}"
TAG="${GMEB_TEST_BUILD_TAG:-gnomad-meb-test:latest}"

RUNTIME="${GMEB_TEST_RUNTIME:-}"
if [[ -z "$RUNTIME" ]]; then
    for c in singularity apptainer; do
        if command -v "$c" >/dev/null 2>&1; then RUNTIME="$c"; break; fi
    done
fi
[[ -n "$RUNTIME" ]] || fail "no singularity/apptainer runtime found; set GMEB_TEST_RUNTIME"

# Apptainer/Singularity build a bundle under TMPDIR and hand it to the OCI
# packer. When TMPDIR is a `nodev` mount (e.g. the tmpfs /tmp on many hosts) the
# packer cannot build there and the build dies instantly with
# "packer failed to pack: ... exit status 125" — before any Dockerfile step
# runs. Point the runtime at a non-nodev directory; override with
# GMEB_TEST_TMPDIR.
TMPDIR_PACK="${GMEB_TEST_TMPDIR:-${APPTAINER_TMPDIR:-/var/tmp}}"
mkdir -p "$TMPDIR_PACK"
export APPTAINER_TMPDIR="$TMPDIR_PACK"
export SINGULARITY_TMPDIR="$TMPDIR_PACK"

echo "converting '$TAG' into $IMAGE with $RUNTIME"
"$RUNTIME" build "$IMAGE" "dockerfile:$SCRIPT_DIR"

echo "wrote $IMAGE"
