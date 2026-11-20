"""The serializer registry: kind → serializer class.

Importing this module must not pull in hail/pandas/plotnine eagerly, because
:mod:`gnomad_meb.config` imports :mod:`gnomad_meb.lib.checkpoint.status` and the
package root re-exports these serializers. The kind classes are therefore
resolved lazily: :func:`get_serializer` imports the one module it needs, and
:data:`SERIALIZERS` is a plain kind → ``"module:Class"`` table a caller (or a
test) can read without importing the heavy modules.
"""

from importlib import import_module

from ..types import CHECKPOINT_TYPE
from .base import Serializer

#: kind → ``"submodule:ClassName"``. The class is imported on demand so the
#: registry (and the package root) stays free of hail/pandas/plotnine imports.
SERIALIZERS: dict[CHECKPOINT_TYPE, str] = {
    CHECKPOINT_TYPE.VARIANT_DATASET: "variant_dataset:VariantDatasetSerializer",
    CHECKPOINT_TYPE.NATIVE_VARIANT_DATASET: (
        "native_variant_dataset:NativeVariantDatasetSerializer"
    ),
    CHECKPOINT_TYPE.MATRIX_TABLE: "matrix_table:MatrixTableSerializer",
    CHECKPOINT_TYPE.HAIL_TABLE: "hail_table:HailTableSerializer",
    CHECKPOINT_TYPE.DATAFRAME: "dataframe:DataFrameSerializer",
    CHECKPOINT_TYPE.PARQUET: "parquet_file:ParquetFileSerializer",
    CHECKPOINT_TYPE.DICTIONARY: "dictionary:DictionarySerializer",
    CHECKPOINT_TYPE.PICKLE: "pickle:PickleSerializer",
    CHECKPOINT_TYPE.PLOTNINE: "plotnine:PlotnineSerializer",
    CHECKPOINT_TYPE.FILE: "file:FileSerializer",
}


def get_serializer(checkpoint_type: CHECKPOINT_TYPE):
    """The serializer class for ``checkpoint_type`` (importing it on demand)."""
    from gnomad_meb.logging import get_logger

    target = SERIALIZERS.get(checkpoint_type)
    if target is None:
        get_logger().error(f"unknown checkpoint type: {checkpoint_type}")
        return None
    module_name, class_name = target.split(":")
    module = import_module(f"{__name__}.{module_name}")
    return getattr(module, class_name)


__all__ = ["SERIALIZERS", "Serializer", "get_serializer"]
