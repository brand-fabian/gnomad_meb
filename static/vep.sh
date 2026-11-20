#! /bin/bash
VEP_ENV=${VEP_ENV:-"gnomad.vep"}
VEP_MODULE=${VEP_MODULE:-"lang/Miniconda3"}
if `type module 2>&1 | grep -q function`; then
    module load "$VEP_MODULE"
fi
>&2 echo "using vep from conda environment ${VEP_ENV}"
>&2 echo "got vep command line" "$@"
# Use `conda run` rather than `conda activate`: activate is a shell function that
# only exists after `conda init`, which the integration-test container never
# loads. `conda run` needs only the `conda` executable on PATH. It also sets the
# environment's PATH so VEP resolves the environment's perl (a bare
# /opt/conda/envs/${VEP_ENV}/bin/vep would pick up the system perl and lose
# PERL5LIB). `--no-capture-output` streams VEP's stdout straight to hail instead
# of buffering it until exit.
if [ "$#" -gt 1 ]; then
    exec conda run --no-capture-output -n "$VEP_ENV" vep "$@" -o STDOUT
else
    exec conda run --no-capture-output -n "$VEP_ENV" vep "$@"
fi
# id=$(shuf -i 1-1000 -n1)
# curr_date=$(date '+%Y%m%dT%H%M%S')
# if [ "$#" -gt 1 ]; then
#     vep "$@" -o /ceph01/scratch/brand/Projects/gnomad/vep/test/$curr_date.$(hostname).$id.json 2>&1 >/ceph01/scratch/brand/Projects/gnomad/vep/test/$curr_date.$(hostname).$id.log
#     cat /ceph01/scratch/brand/Projects/gnomad/vep/test/$curr_date.$(hostname).$id.json
# else
#     vep "$@"
# fi