REQUIRED_ANNOTATIONS = [
    "QUALapprox",
    "VARdp",
]
REQUIRED_ALLELE_SPECIFIC_ANNOTATIONS = [
    "AS_QUALapprox",
    "AS_SB_TABLE",
    "AS_RAW_MQ",
    "AS_RAW_ReadPosRankSum",
    "AS_RAW_MQRankSum",
    "AS_VarDP",
]

#: The GATK provenance file the VQSR step writes next to the per-split VCFs,
#: holding ``{version, build, url}`` detected from the deployed conda
#: environment. The import reads it back for the ``gatk_version`` global.
GATK_VERSION_FILENAME = "gatk.version.json"
