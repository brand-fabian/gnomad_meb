import json
import shlex
from pathlib import Path
from textwrap import dedent

from gnomad.utils import vep

from gnomad_meb.config import get_config
from gnomad_meb.hail import hail as hl
from gnomad_meb.logging import get_logger


def configure_vep() -> str:
    """Create a VEP command string based on the configuration in annotation.vep.
    
    Returns
    -------
    str
        VEP Command invocation. This can be added as command flag after applying
        shlex.split in the hail vep configuration.

    Note
    ----
    An example for a cluster-suitable vep executable can be found in the static
    directory of the gnomad_meb repository.
    """
    cfg = get_config()

    assert cfg.annotation.vep is not None, (
        f"missing VEP annotation configuration: cfg = {cfg.annotation}"
    )
    vep_cfg = cfg.annotation.vep

    use_cadd = vep_cfg.cadd is not None
    if use_cadd:
        cadd_plugin = (
            f"--plugin CADD,snv={vep_cfg.cadd.resolve()},"
            f"indels={vep_cfg.cadd_indel.resolve()}"
        )
    else:
        cadd_plugin = ""

    loftee = {
        **vep_cfg.loftee_files,
        "loftee_path": vep_cfg.loftee_base,
    }
    loftee_plugin = f"--plugin LoF,{','.join(f'{k}:{v}' for k, v in loftee.items())}"

    use_splice_ai = vep_cfg.splice_ai is not None
    if use_splice_ai:
        splice_ai_plugin = (
            f"--plugin SpliceAI,snv={vep_cfg.splice_ai.resolve()},"
            f"indel={vep_cfg.splice_ai_indel.resolve()}"
        )
    else:
        splice_ai_plugin = ""

    if vep_cfg.revel is not None:
        revel_plugin = f"--plugin REVEL,file={vep_cfg.revel.resolve()}"
    else:
        revel_plugin = ""

    if vep_cfg.phylop is not None:
        phylop_annotation = (
            f"--custom file={vep_cfg.phylop.resolve()},short_name=phylop,"
            "format=bigwig,num_records=all"
        )
    else:
        phylop_annotation = ""

    cmd = dedent(f"""\
        {vep_cfg.vep}
            --format vcf
            __OUTPUT_FORMAT_FLAG__
            --everything
            --allele_number
            --mane_select
            --minimal
            --no_stats
            --cache_version {vep_cfg.version}
            --cache
            --offline
            --species homo_sapiens
            --assembly GRCh38
            --dir_cache {vep_cfg.cache_dir}
            --fasta {cfg.resources.reference}
            {loftee_plugin}
            --plugin UTRAnnotator,file={vep_cfg.utr_annotator.resolve()}
            {cadd_plugin}
            {splice_ai_plugin}
            {revel_plugin}
            {phylop_annotation}
    """)
    # -o STDOUT
    return cmd


def vep_json_schema():
    """Create a VEP .json schema for the version used in the configuration."""
    cfg = get_config()

    assert cfg.annotation is not None and cfg.annotation.vep is not None, (
        f"missing vep configuration: annotation = {cfg.annotation}"
    )

    if cfg.annotation.vep.version == 110:
        use_cadd = cfg.annotation.vep.cadd is not None
        use_splice_ai = cfg.annotation.vep.splice_ai is not None
        use_revel = cfg.annotation.vep.revel is not None
        use_phylop = cfg.annotation.vep.phylop is not None

        json_schema = dedent("""\
            Struct{
                minimised:Int32,
                assembly_name:String,
                allele_string:String,
                ancestral:String,
                colocated_variants:Array[Struct{
                    allele_string:String,
                    clin_sig:Array[String],
                    clin_sig_allele:String,
                    end:Int32,
                    id:String,
                    minimised:Int32,
                    minor_allele:String,
                    minor_allele_freq:Float64,
                    phenotype_or_disease:Int32,
                    pubmed:Array[Int32],
                    seq_region_name:String,
                    somatic:Int32,
                    start:Int32,
                    strand:Int32
                }],
                context:String,
                end:Int32,
                id:String,
                input:String,
                intergenic_consequences:Array[Struct{
                    allele_num:Int32,
                    consequence_terms:Array[String],
                    impact:String,
                    minimised:Int32,
                    variant_allele:String
                }],
                most_severe_consequence:String,
                motif_feature_consequences:Array[Struct{
                    allele_num:Int32,
        """)

        if use_cadd:
            json_schema += dedent("""\
                cadd_raw:Float64,
                cadd_phred:Float64,
            """)

        json_schema += dedent("""\
                consequence_terms:Array[String],
                high_inf_pos:String,
                impact:String,
                minimised:Int32,
                motif_feature_id:String,
                motif_name:String,
                motif_pos:Int32,
                motif_score_change:Float64,
                strand:Int32,
                transcription_factors:Array[String],
                variant_allele:String
            }],
            regulatory_feature_consequences:Array[Struct{
                allele_num:Int32,
        """)

        if use_cadd:
            json_schema += dedent("""\
                cadd_phred:Float64,
                cadd_raw:Float64,
            """)

        json_schema += dedent("""\
                biotype:String,
                consequence_terms:Array[String],
                impact:String,
                minimised:Int32,
                regulatory_feature_id:String,
                variant_allele:String
            }],
            seq_region_name:String,
            start:Int32,
            strand:Int32,
            transcript_consequences:Array[Struct{
                allele_num:Int32,
                amino_acids:String,
                appris:String,
                biotype:String,
            """)

        if use_cadd:
            json_schema += dedent("""\
                cadd_raw:Float64,
                cadd_phred:Float64,""")

        json_schema += dedent("""\
            canonical:Int32,
            mane_select:String,
            mane_plus_clinical:String,
            ccds:String,
            cdna_start:Int32,
            cdna_end:Int32,
            cds_end:Int32,
            cds_start:Int32,
            codons:String,
            consequence_terms:Array[String],
            distance:Int32,
            domains:Array[Struct{
                db:String,
                name:String
            }],
            exon:String,
            gene_id:String,
            gene_pheno:Int32,
            gene_symbol:String,
            gene_symbol_source:String,
            hgnc_id:String,
            hgvsc:String,
            hgvsp:String,
            hgvs_offset:Int32,
            impact:String,
            intron:String,
            lof:String,
            lof_flags:String,
            lof_filter:String,
            lof_info:String,
            existing_inframe_oorfs:Int32,
            existing_outofframe_oorfs:Int32,
            existing_uorfs:Int32,
            five_prime_utr_consequence:String,
            five_prime_utr_annotation:Dict[
                String,
                Struct{
                    type:String,
                    KozakContext:String,
                    KozakStrength:String,
                    DistanceToCDS:String,
                    CapDistanceToStart:String,
                    DistanceToStop:String,
                    Evidence:String,
                    AltStop:String,
                    AltStopDistanceToCDS:String,
                    FrameWithCDS:String,
                    StartDistanceToCDS:String,
                    newSTOPDistanceToCDS:String,
                    alt_type:String,
                    alt_type_length:String,
                    ref_StartDistanceToCDS:String,
                    ref_type:String,
                    ref_type_length:String
                }
            ],
            minimised:Int32,
            mirna:Array[String],
            polyphen_prediction:String,
            polyphen_score:Float64,
            protein_end:Int32,
            protein_start:Int32,
            protein_id:String,
            sift_prediction:String,
            sift_score:Float64,
            strand:Int32,
            swissprot:Array[String],
            transcript_id:String,
            trembl:Array[String],
            tsl:Int32,
            uniparc:Array[String],
            uniprot_isoform:Array[String],
            variant_allele:String,
            source:String,
            flags:Array[String]""")

        if use_splice_ai:
            json_schema += dedent("""\
                ,
                spliceai:Struct{
                    DP_AG:Int32,
                    DP_DL:Int32,
                    DP_DG:Int32,
                    DP_AL:Int32,
                    DS_AL:Float64,
                    DS_DG:Float64,
                    SYMBOL:String,
                    DS_DL:Float64,
                    DS_AG:Float64
                }""")
            
        if use_revel:
            json_schema += dedent("""\
                ,
                revel:Float64
            """)

        json_schema += dedent("""\
                }],
                variant_class:String
        """)

        if use_phylop:
            json_schema += dedent("""\
                ,
                custom_annotations:Struct{
                    phylop:Array[Struct{
                        name:Float64
                    }]
                }                 
            """)

        json_schema += "}"

        return json_schema
    else:
        raise ValueError(
            f"invalid or unknown VEP version: version = {cfg.annotation.vep.version}"
        )
    

def splice_ai_ds_max_expr(
    ht: hl.Table, key: str = "vep", ds_keys: list[str] | None = None
):
    """Extract the max. ds from SpliceAI annotations.
    
    Extracts the splice ai field from `key` in the hail table and reports the
    maximum delta score annotation. By default, all splice ai delta score fields
    are considered (al, dg, dl, ag), set `ds_keys` to adjust which fields are
    selected.

    Fields are converted to float32 expressions to align the field schema with the
    gnomad v4 field `insilico_predictors.spliceai_ds_max`.

    Returns
    -------
    hl.FloatExpression
        Expression for the maximum splice ai delta score value.
    """
    if ds_keys is None:
        ds_keys = ["DS_AL", "DS_DG", "DS_DL", "DS_AG"]
    return hl.float32(hl.max(*hl.struct(**{
        f: hl.max(ht[key].transcript_consequences.spliceai[f])
        for f in ds_keys
    }).values()))


def cadd_score_expr(ht: hl.Table, key: str = "vep"):
    """Extract the max. CADD scores."""
    return hl.struct(
        phred=hl.float32(hl.max(ht[key].transcript_consequences.cadd_phred)),
        raw_score=hl.float32(hl.max(ht[key].transcript_consequences.cadd_raw)),
    )


def sift_and_polyphen_max_expr(ht: hl.Table, key: str = "vep"):
    """Extract the maximum sift and polyphen scores."""
    mane = vep.filter_vep_transcript_csqs(
        ht, synonymous=False, canonical=False, mane_select=True
    )
    canonical = vep.filter_vep_transcript_csqs(ht, synonymous=False, canonical=True)
    return hl.struct(
        sift_max=hl.or_else(
            hl.max(mane[ht.key][key].transcript_consequences.sift_score),
            hl.max(canonical[ht.key][key].transcript_consequences.sift_score),
        ),
        polyphen_max=hl.or_else(
            hl.max(mane[ht.key][key].transcript_consequences.polyphen_score),
            hl.max(canonical[ht.key][key].transcript_consequences.polyphen_score),
        ),
    )


def phylop_expr(ht: hl.Table, key: str = "vep"):
    return hl.max(ht[key].custom_annotations.phylop.name)


def revel_max_expr(ht: hl.Table, key: str = "vep"):
    return hl.max(ht[key].transcript_consequences.revel)


def insilico_predictors_expr(
    ht: hl.Table,
    key: str = "vep",
):
    """Extract the insilico_predictors fields from the table at `key`.
    
    This function extracts the insilico_predictors struct from the annotation
    field at `key`. The schema of the resulting struct is aligned with the data
    computed by the gnomad v4.0 annotation, adding empty values for variables
    that are currently unsupported by the VEP annotation (pangolin).
    """
    # The ordering of fields is important for hail equality checks between
    # struct schemas, so this must be aligned with the in_silico_predictors
    # field from the gnomad ht used in the annotation pipeline.
    return hl.struct(**{
        "cadd": cadd_score_expr(ht, key=key),
        "revel_max": revel_max_expr(ht, key=key),
        "spliceai_ds_max": splice_ai_ds_max_expr(ht, key=key),
        "pangolin_largest_ds": hl.missing(hl.tfloat64),
        "phylop": phylop_expr(ht, key=key),
        **sift_and_polyphen_max_expr(ht, key=key),
    })


def filter_gnomad_vep(
    ht: hl.Table,
    reference_ht: hl.Table,
    key: str = "vep",
    insilico_predictors_key: str = "in_silico_predictors",
    ignore_incompatible_dtypes: bool = True,
):
    """Align the vep expression to the reference annotation schema."""
    logger = get_logger()
    def _copy_array(
        field: hl.expr.ArrayExpression, reference_field: hl.expr.ArrayExpression
    ):
        assert isinstance(reference_field.dtype, hl.tarray), (
            f"invalid dtype for _copy_array: dtype = {reference_field.dtype!s}"
        )
        assert isinstance(reference_field.dtype.element_type, hl.tstruct), (
            "invalid element dtype for _copy_array: "
            f"dtype = {reference_field.dtype.element_type!s}"
        )
        return field.map(
            lambda x: hl.struct(
                **{f: x[f] for f in reference_field.dtype.element_type.fields}
            )
        )

    def _copy_struct(
        field: hl.expr.StructExpression, reference_field: hl.expr.StructExpression
    ):
        keys = reference_field.keys()
        kwargs = {}
        for key in keys:
            ref_dtype = reference_field[key].dtype
            if isinstance(ref_dtype, hl.tstruct):
                kwargs[key] = _copy_struct(field[key], reference_field[key])
            elif isinstance(ref_dtype, hl.tarray) and isinstance(
                ref_dtype.element_type, hl.tstruct
            ):
                kwargs[key] = _copy_array(field[key], reference_field[key])
            else:
                if reference_field[key].dtype == field[key].dtype:
                    kwargs[key] = field[key]
                else:
                    msg = (
                        "difference in dtypes between reference and ht fields: "
                        f"key = {key}, reference = {ref_dtype!s}, "
                        f"other = {field[key].dtype!s}"
                    )
                    if ignore_incompatible_dtypes:
                        logger.warning(msg)
                        kwargs[key] = field[key]
                    else:
                        raise ValueError(msg)
        return hl.struct(**kwargs)

    # Flatten the flags key to align it with the schema expected by the gnomAD
    # VEP annotation struct (array<str> -> str).
    updated_struct_fields = {}
    for struct_field in [
        "transcript_consequences",
        "worst_csq_by_gene",
        "worst_csq_by_gene_canonical",
        "worst_csq_for_variant",
        "worst_csq_for_variant_canonical",
    ]:
        if isinstance(ht[key][struct_field].dtype, hl.tarray):
            updated_struct_fields[struct_field] = ht[key][struct_field].map(
                lambda x: hl.struct(**{**x, "flags": hl.str(",").join(x.flags)})
            )
        elif isinstance(ht[key][struct_field].dtype, hl.tstruct):
            struct_flags = ht[key][struct_field].flags
            updated_struct_fields[struct_field] = ht[key][struct_field].annotate(
                **{
                    "flags": hl.str(",").join(struct_flags)
                    if isinstance(struct_flags.dtype, hl.tarray)
                    else struct_flags,
                }
            )

    ht = ht.annotate(**{key: hl.struct(**{**ht[key], **updated_struct_fields})})
    ht = ht.annotate(**{
        key: _copy_struct(ht[key], reference_ht[key]),
        insilico_predictors_key: insilico_predictors_expr(ht, key=key),
    })
    return ht


def vep_or_lookup_vep(
    ht: hl.Table,
    vep_config_path: str | Path,
    reference_vep_ht: hl.Table | None = None,
    reference: str | None = None,
    vep_key: str = "vep",
    insilico_predictors_key: str = "in_silico_predictors",
    include_csq: bool = False,
):
    """Apply VEP to all rows not specified in the reference table.

    This function applies VEP for all variants which have not been annotated in
    the reference table.

    This function has been sourced largely from
    https://github.com/broadinstitute/gnomad_methods (``gnomad/utils/vep.py``)
    and adheres to similar constraints, i.e. it does not check the VEP version
    before annotating and there may be subtle differences in the field schema as
    a consequence.

    Arguments
    ---------
    ht : hl.Table
        Table containing variants to annotate.
    vep_config_path : str | Path
        Path to the VEP configuration that will be written.
    reference_vep_ht : hl.Table | None
        Path to a hl.Table containing pre-annotated variants.
    reference : str | None
        Name of the hail reference to use for annotation, defaults to the
        current reference genome if not set.

    Returns
    -------
    hl.Table
        A hail table with all variants annotated.
    """
    cfg = get_config()
    logger = get_logger()

    if reference is None:
        reference = hl.default_reference().name

    logger.info(f"annotating hail table with VEP: config_path = {vep_config_path}")
    logger.debug(f"creating vep configuration: config_path = {vep_config_path}")
    vep_config = {
        "command": [s.strip() for s in shlex.split(configure_vep())],
        "env": {},
        "vep_json_schema": vep_json_schema().replace('\n', ''),
    }
    with open(vep_config_path, "w") as out_f:
        json.dump(vep_config, out_f, indent=4, ensure_ascii=False)

    if reference_vep_ht is not None:
        ht = ht.annotate(vep=reference_vep_ht[ht.key].vep)
    else:
        logger.warning("no reference vep ht supplied, annotating all variants")

    vep_ht = ht.filter(hl.is_defined(ht.vep))
    missing_vep_ht = ht.filter(hl.is_missing(ht.vep))

    defined_count, missing_count = map(
        int,
        ht.aggregate(
            (
                hl.agg.count_where(hl.is_defined(ht.vep)),
                hl.agg.count_where(hl.is_missing(ht.vep)),
            )
        ),
    )

    logger.info(f"annotating missing variation using VEP: n_missing = {missing_count}")
    missing_vep_ht = hl.vep(missing_vep_ht, vep_config_path, name=vep_key)

    if include_csq:
        logger.info(f"annotating VEP CSQ field, key = {vep_key}_csq")
        missing_vep_ht = hl.vep(
            missing_vep_ht, vep_config_path, csq=True, name=f"{vep_key}_csq"
        )

    missing_vep_ht = missing_vep_ht.checkpoint(
        hl.utils.new_temp_file("vep_missing", extension="ht"),
        overwrite=True,
    )

    missing_vep_ht = vep.process_consequences(
        missing_vep_ht, has_polyphen=False, vep_root=vep_key
    )
    missing_vep_ht = filter_gnomad_vep(
        missing_vep_ht,
        vep_ht,
        key=vep_key,
        insilico_predictors_key=insilico_predictors_key,
    )
    missing_vep_ht = missing_vep_ht.drop(missing_vep_ht.vep_proc_id)

    ht = vep_ht.union(missing_vep_ht, unify=True)
    global_annotations = {
        "vep_version": cfg.annotation.vep.version,
        "vep_help": vep.get_vep_help(vep_config_path),
        "vep_config": hl.literal(
            {k: v for k, v in vep_config.items() if v is not None and len(v) > 0}
        ),
        "defined": defined_count,
        "missing": missing_count,
        "vep_key": vep_key,
    }
    ht = ht.annotate_globals(vep_globals=hl.literal(global_annotations))
    return ht # vep_ht, missing_vep_ht
