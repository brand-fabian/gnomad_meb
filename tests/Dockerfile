# syntax=docker/dockerfile:1
#
# Minimal integration-test image (see
# docs/adr/0002-integration-tests-run-in-a-minimal-container.md).
#
# The image provides only Python, uv, a JVM and Miniforge (conda). Python
# dependencies are *not* baked in: `uv run --project <repo>` installs them from
# pyproject.toml/uv.lock on the first invocation, so the tool's declared
# dependency set stays the single source of truth. External tools needed by VQSR
# and annotation (GATK, bcftools, picard, VEP, CADD, ...) are provisioned by
# snakemake at runtime, through its own conda environments (`--use-conda`).
#
# The JVM is required by hail/pyspark; the slurm wrappers `module load Java/11`,
# but there is no module system inside the container, so it is installed here.
# Conda is likewise provided by a module on the cluster; inside the container it
# is installed directly, since snakemake needs `conda` on PATH to build the
# per-rule environments.
#
# Build a Singularity image from this file with tests/build.sh.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

# Tools to fetch and unpack the Miniforge installer come from bookworm, before
# the unstable repo is added below. The unstable repo is only needed for
# openjdk-11 (absent from bookworm); installing the extras ahead of it keeps
# that resolve from dragging in unrelated unstable upgrades (e.g. openssl).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates bzip2 zstd \
    && rm -rf /var/lib/apt/lists/*

RUN echo "deb http://deb.debian.org/debian unstable main non-free contrib" >>/etc/apt/sources.list.d/unstable.list
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential openjdk-11-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Miniforge supplies `conda`, which snakemake's `--use-conda` VQSR/annotation
# rules need to create their per-rule environments at runtime. Pin the installer
# so the image stays reproducible; override with --build-arg if needed.
ARG MINIFORGE_VERSION=26.9.4-1
RUN ARCH="$(uname -m)" \
    && curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-${ARCH}.sh" \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda clean -afy

# Activation only needs /opt/conda/bin/activate, so no shell init is required.
ENV PATH=/opt/conda/bin:$PATH

# The gnomad_meb.annotation pipeline needs vep to function correctly, the default
# static/vep.sh looks for it in the gnomad.vep environment:
RUN conda create -y -n "gnomad.vep" -c conda-forge -c bioconda ensembl-vep

# The image is read-only at run time, so conda's package cache must live
# somewhere writable; environments themselves are created under the run
# directory, which is bind-mounted read-write.
ENV CONDA_PKGS_DIRS=/tmp/conda-pkgs

# Write-through copies rather than hardlinks so uv works across the bind mounts
# used by tests/run.sh.
ENV UV_LINK_MODE=copy

WORKDIR /opt/gnomad-meb
