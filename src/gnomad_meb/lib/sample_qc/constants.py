#: Additive, per-sample count fields of ``hl.vds.sample_qc``. The four ratio
#: fields (``r_ti_tv``, ``r_ti_tv_singleton``, ``r_het_hom_var``,
#: ``r_insertion_deletion``) are derived and recomputed from these after the
#: per-split fold. This enumerates Hail's documented struct fields; if Hail's
#: ``sample_qc`` schema changes the list must be revisited.
SAMPLE_QC_SUM_FIELDS = (
    "n_het",
    "n_hom_var",
    "n_non_ref",
    "n_singleton",
    "n_singleton_ti",
    "n_singleton_tv",
    "n_snp",
    "n_insertion",
    "n_deletion",
    "n_transition",
    "n_transversion",
    "n_star",
)
