#! /bin/bash
VEP_ENV=${VEP_ENV:-"gnomad.vep-110"}
VEP_MODULE=${VEP_MODULE:-"Miniforge3"}
source ~/.bashrc && conda_init

>&2 echo "using vep from conda environment ${VEP_ENV}"
>&2 echo "got vep command line" "$@"
conda activate "$VEP_ENV"

if [ "$#" -gt 1 ]; then
    has_no_stats_flag=1
    for arg in "$@"; do
        if [ "$arg" = "--no_stats" ]; then
            has_no_stats_flag=0
            break
        fi
    done
    
    if [ $has_no_stats_flag -eq 1 ]; then
        temp_dir=$(mktemp -d)
        >&2 echo "writing stats file to $temp_dir/stats.html"
        set -- "$@" --stats_file "$temp_dir/stats.html"
    fi

    vep "$@" -o STDOUT
else
    vep "$@"
fi
# id=$(shuf -i 1-1000 -n1)
# curr_date=$(date '+%Y%m%dT%H%M%S')
# if [ "$#" -gt 1 ]; then
#     vep "$@" -o /ceph01/scratch/brand/Projects/gnomad/vep/test/$curr_date.$(hostname).$id.json 2>&1 >/ceph01/scratch/brand/Projects/gnomad/vep/test/$curr_date.$(hostname).$id.log
#     cat /ceph01/scratch/brand/Projects/gnomad/vep/test/$curr_date.$(hostname).$id.json
# else
#     vep "$@"
# fi
