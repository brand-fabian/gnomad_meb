#!/bin/bash

GMEB_ENVIRONMENT_NAME="${GMEB_ENVIRONMENT_NAME:-uv}"
GMEB_MODULE_NAME="${GMEB_MODULE_NAME:-Miniforge3}"
GMEB_MODULE_NAME_JAVA="${GMEB_MODULE_NAME_JAVA:-Java/11}"

if `type module 2>&1 | grep -q function`; then
    source ~/.bashrc && conda_init
    module load "$GMEB_MODULE_NAME" "$GMEB_MODULE_NAME_JAVA"
fi

if [ -z "$(conda env list | grep -P "^$GMEB_ENVIRONMENT_NAME\\s+/.*$GMEB_ENVIRONMENT_NAME$")" ]; then
    echo -e "creating environment: $GMEB_ENVIRONMENT_NAME"
    conda create -n "$GMEB_ENVIRONMENT_NAME" -c conda-forge "python>=3.11,<3.14" "pip" "uv"
fi

conda activate "$GMEB_ENVIRONMENT_NAME"
