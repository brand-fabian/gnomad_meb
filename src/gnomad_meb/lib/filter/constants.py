"""Filter-stage checkpoint prefixes and the manifest schema version.

Public names only: the ``{project}``-templated prefixes each filter step
declares, shared by the decorators and the manifest so the paths it records
cannot drift from the artifacts the DAG writes, plus the manifest schema
version.
"""

#: The prefix every filter-stage step is checkpointed under.
CHECKPOINT_PREFIX = "{project}/intermediate/filter"

#: The fixed high-quality (VQSR/calls/inbreeding) step: its matrix and the two
#: result dictionaries.
HIGH_QUALITY_PREFIX = f"{CHECKPOINT_PREFIX}/high_quality_gnomad"

#: The constraints-annotated matrix the configured filters build on.
CONSTRAINT_PREFIX = f"{CHECKPOINT_PREFIX}/constraint"

#: The stage's manifest: the one index of filters, output paths and results.
MANIFEST_PREFIX = f"{CHECKPOINT_PREFIX}/manifest"

#: The manifest schema version. Bump when the shape changes incompatibly.
MANIFEST_SCHEMA_VERSION = 1
