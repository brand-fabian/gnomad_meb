import gzip
from pathlib import Path

from gnomad_meb.config import get_config
from gnomad_meb.hail import get_default_partitions
from gnomad_meb.hail import hail as hl
from gnomad_meb.logging import get_logger


def read_clinvar_vcf(path: Path | str) -> str:
    """The ``clinvar_vcf`` resource loader: return the source path.

    The clinvar VCF is not read here:
    :meth:`~gnomad_meb.lib.annotation.pipeline.AnnotationPipeline.import_clinvar_ht`
    imports it with its own VCF options and still needs the source path for
    :func:`~gnomad_meb.lib.annotation.utils.get_clinvar_filedate`, so this
    loader is path-only (ticket 31).
    """
    return str(path)


def get_cadd_ht_loader(cadd: Path | str, cadd_indel: Path | str):
    """Build the ``cadd``/``cadd_indel`` resource loader over both CADD files.

    CADD is split across a SNV and an indel file, but a
    :class:`~gnomad_meb.pipeline.Resource` declares one ``path``. Both files are
    bound explicitly here, so the loader reads the pair through
    :func:`get_cadd_ht` regardless of which resource's ``path`` the runner hands
    it — matching what ``import_cadd_ht`` consumes.
    """
    def load(path: Path | str) -> hl.Table:
        return get_cadd_ht(source=(cadd, cadd_indel))

    return load


def get_alphamissense_ht(
    source: Path | str | None = None, reference_genome: str = "GRCh38"
) -> hl.Table:
    """Create a hail table containing alpha missense scores.
    
    Note
    ----
    AlphaMissense scores can be downloaded from the google cloud storage
    bucket: https://console.cloud.google.com/storage/browser/dm_alphamissense;tab=objects?prefix=&forceOnObjectsSortingFiltering=false
    """
    if source is None:
        cfg = get_config()

        assert cfg.annotation is not None and cfg.annotation.vep is not None, (
            "missing annotation configuration"
        )
        assert cfg.annotation.vep.alpha_missense is not None, (
            f"missing alpha missense annotation: cfg = {cfg.annotation.vep}"
        )
        source = cfg.annotation.vep.alpha_missense
    else:
        source = Path(source)

    column_names = {
        "f0": "chrom",
        "f1": "pos",
        "f2": "ref",
        "f3": "alt",
        "f4": "genome",
        "f5": "uniprot_id",
        "f6": "transcript_id",
        "f7": "protein_variant",
        "f8": "am_pathogenicity",
        "f9": "am_class",
    }
    types = {
        "f0": hl.tstr,
        "f1": hl.tint32,
        "f2": hl.tstr,
        "f3": hl.tstr,
        "f4": hl.tstr,
        "f5": hl.tstr,
        "f6": hl.tstr,
        "f7": hl.tstr,
        "f8": hl.tfloat32,
        "f9": hl.tstr,
    }
    ht = hl.import_table(
        str(source.resolve()),
        delimiter="\t",
        min_partitions=get_default_partitions(),
        comment="#",
        types=types,
        no_header=True,
        force_bgz=True,
    )
    ht = ht.rename(column_names)
    ht = ht.annotate(
        locus=hl.locus(ht.chrom, ht.pos, reference_genome=reference_genome),
        alleles=hl.array([ht.ref, ht.alt]),
    )
    ht = ht.select("locus", "alleles", "am_pathogenicity", "am_class")
    ht = ht.key_by("locus", "alleles")
    return ht


def get_cadd_ht(
    source: tuple[Path | str, Path | str] | None = None,
    reference_genome: str = "GRCh38",
) -> hl.Table:
    """Load the CADD annotation files for SNVs and Indels."""
    logger = get_logger()
    if source is None:
        cfg = get_config()

        assert cfg.annotation is not None and cfg.annotation.vep is not None, (
            "missing annotation configuration"
        )
        assert (
            cfg.annotation.vep.cadd is not None
            and cfg.annotation.vep.cadd_indel is not None
        ), f"missing cadd annotation: cfg = {cfg.annotation.vep}"
        cadd, cadd_indel = cfg.annotation.vep.cadd, cfg.annotation.vep.cadd_indel
    else:
        cadd, cadd_indel = Path(source[0]), Path(source[1])

    def _get_cadd_version(p: Path | str):
        with gzip.open(p, "rt") as in_f:
            first_line = in_f.readline().strip("\n")
        prefix, ref_version, _ = first_line.split(" ", maxsplit=2)
        assert prefix == "##CADD", f"invalid cadd file header line: line = {first_line}"
        reference, version = ref_version.split("-", maxsplit=1)
        return reference, version

    def _load_cadd_snv(p: Path | str):
        logger.debug(f"loading cadd snv file to hail table: snv = {p}")
        column_names = {
            "f0": "chrom",
            "f1": "pos",
            "f2": "ref",
            "f3": "alt",
            "f4": "raw_score",
            "f5": "phred",
        }
        types = {
            "f0": hl.tstr,
            "f1": hl.tint32,
            "f2": hl.tstr,
            "f3": hl.tstr,
            "f4": hl.tfloat32,
            "f5": hl.tfloat32,
        }
        ht = hl.import_table(
            p if isinstance(p, str) else str(p.resolve()),
            delimiter="\t",
            min_partitions=get_default_partitions(),
            comment="#",
            types=types,
            no_header=True,
            force_bgz=True,
        )
        ht = ht.rename(column_names)
        chrom_expr = hl.if_else(
            ht.chrom.startswith("chr"), ht.chrom, hl.format("chr%s", ht.chrom)
        )
        ht = ht.annotate(
            locus=hl.locus(chrom_expr, ht.pos, reference_genome=reference_genome),
            alleles=hl.array([ht.ref, ht.alt]),
        )
        ht = ht.select("locus", "alleles", "raw_score", "phred")
        ht = ht.key_by("locus", "alleles")
        return ht

    def _load_cadd_indels(p: Path | str):
        """Load the default CADD indel annotation file.
        
        The reference CADD indel file contains CADD annotations for all gnomAD v4 indel
        calls with associated annotations. This function loads the RawScore and PHRED 
        score columns only from the .tsv file to a hail table.

        Note
        ----
        In order to derive field names via the types / column_names dicts, we can
        inspect the header of the .tsv file from CADD:

        # Retrieve the number of columns in the file:
        #   zcat <cadd_indel.tsv.gz> | head -n 2 | tail -n 1 | tr '\t' '\n' | wc -l

        The first four columns are identical with the SNV file and the last two columns
        denote the raw and phred scores respectively.
        """
        logger.debug(f"loading cadd indel file to hail table: indel = {p}")
        column_names = {
            "f0": "chrom",
            "f1": "pos",
            "f2": "ref",
            "f3": "alt",
            "f151": "raw_score",
            "f152": "phred",
        }
        types = {
            "f0": hl.tstr,
            "f1": hl.tint32,
            "f2": hl.tstr,
            "f3": hl.tstr,
            "f151": hl.tfloat32,
            "f152": hl.tfloat32,
        }
        ht = hl.import_table(
            p if isinstance(p, str) else str(p.resolve()),
            delimiter="\t",
            min_partitions=get_default_partitions(),
            comment="#",
            types=types,
            no_header=True,
            force_bgz=True,
            impute=True,
        )
        ht = ht.rename(column_names)
        chrom_expr = hl.if_else(
            ht.chrom.startswith("chr"), ht.chrom, hl.format("chr%s", ht.chrom)
        )
        ht = ht.annotate(
            locus=hl.locus(chrom_expr, ht.pos, reference_genome=reference_genome),
            alleles=hl.array([ht.ref, ht.alt]),
        )
        ht = ht.select("locus", "alleles", "raw_score", "phred")
        ht = ht.key_by("locus", "alleles")
        return ht

    cadd_reference, cadd_version = _get_cadd_version(cadd)
    cadd_indel_reference, cadd_indel_version = _get_cadd_version(cadd_indel)
    
    assert cadd_reference == cadd_indel_reference == reference_genome, (
        f"mismatching reference genome annotations: cadd = {cadd_reference}, "
        f"cadd_indel = {cadd_indel_reference}, request = {reference_genome}"
    )
    assert cadd_version == cadd_indel_version, (
        f"version mismatch between SNV and Indel files: cadd = {cadd_version}, "
        f"cadd_indel = {cadd_indel_version}"
    )
    logger.debug(
        f"detected cadd version and reference from table files: "
        f"cadd_version = {cadd_version}, cadd_reference = {cadd_reference}"
    )

    snv = _load_cadd_snv(cadd)
    indel = _load_cadd_indels(cadd_indel)

    ht = snv.union(indel)
    ht = ht.select(cadd=hl.struct(phred=ht.phred, raw_score=ht.raw_score))
    ht = ht.annotate_globals(cadd_version=cadd_version, cadd_reference=cadd_reference)
    return ht