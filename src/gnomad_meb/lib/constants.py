"""Constants."""

from enum import StrEnum

from gnomad_meb.logging import get_logger

DEFAULT = "<DEFAULT>"
UNKNOWN = "<UNKNOWN>"


logger = get_logger()


class Caller(StrEnum):
    UNKNOWN = UNKNOWN
    PARABRICKS = "PARABRICKS"
    GATK = "GATK"
    DRAGEN = "DRAGEN"


VALID_PARABRICKS_VERSIONS = ["4.7.0", "4.4.0", "4.3.1"]
VALID_DRAGEN_VERSIONS = ["2.6.3"]
VALID_GATK_VERSIONS = ["4.1.4.1"]


def get_valid_caller_versions(
    clr: Caller,
    valid_versions: dict[Caller, list[str]] | None = None,
) -> set[str]:
    """Get known valid versions for a caller."""
    if valid_versions is None:
        valid_versions = {
            Caller.PARABRICKS: VALID_PARABRICKS_VERSIONS,
            Caller.DRAGEN: VALID_DRAGEN_VERSIONS,
            Caller.GATK: VALID_GATK_VERSIONS,
            Caller.UNKNOWN: [],
        }
    if clr not in valid_versions:
        logger.warning(f"unknown caller versions for {clr}")
    return set(valid_versions[clr])


DEFAULT_METADATA_SUFFIX = "gnomad_meb.md.json"


VEP_KEY = "vep"
IN_SILICO_PREDICTORS_KEY = "in_silico_predictors"
