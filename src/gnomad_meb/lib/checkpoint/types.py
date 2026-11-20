"""The checkpoint artifact vocabulary: kinds, extensions and success-file info."""

from enum import Enum
from typing import NamedTuple


class CHECKPOINT_TYPE(Enum):
    VARIANT_DATASET = "VARIANT_DATASET"
    NATIVE_VARIANT_DATASET = "NATIVE_VARIANT_DATASET"  # <- variant dataset implementation by this module
    MATRIX_TABLE = "MATRIX_TABLE"
    HAIL_TABLE = "HAIL_TABLE"
    DATAFRAME = "DATAFRAME"
    PARQUET = "PARQUET"
    DICTIONARY = "DICTIONARY"
    PICKLE = "PICKLE"
    PLOTNINE = "PLOTNINE"
    FILE = "FILE"


CHECKPOINT_EXTENSION = {
    CHECKPOINT_TYPE.VARIANT_DATASET: "vds",
    CHECKPOINT_TYPE.NATIVE_VARIANT_DATASET: "metadata.json",
    CHECKPOINT_TYPE.MATRIX_TABLE: "mt",
    CHECKPOINT_TYPE.HAIL_TABLE: "ht",
    CHECKPOINT_TYPE.DATAFRAME: "parquet",
    CHECKPOINT_TYPE.PARQUET: "parquet",
    CHECKPOINT_TYPE.DICTIONARY: "json.gz",
    CHECKPOINT_TYPE.PICKLE: "pickle.gz",
    CHECKPOINT_TYPE.PLOTNINE: None,
    CHECKPOINT_TYPE.FILE: None,
}


class SuccessFileInfo(NamedTuple):
    path: str
    exists: bool
    mtime: str | None
    metadata: str | None
