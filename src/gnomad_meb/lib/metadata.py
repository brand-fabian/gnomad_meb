"""Create and manage metadata tables for a sequencing experiment."""

from datetime import datetime

from gnomad_meb.hail import hail as hl
from gnomad_meb.lib.constants import DEFAULT, UNKNOWN, Caller, get_valid_caller_versions
from gnomad_meb.lib.utils import get_git_revision, get_version
from gnomad_meb.logging import get_logger

logger = get_logger()


def get_metadata_expr(
    project_name: str = DEFAULT,
    caller: str = UNKNOWN,
    caller_version: str = UNKNOWN,
) -> hl.expr.Expression:
    """Get a metadata expression describing the current experiment setting."""
    caller = caller.upper()

    if caller not in Caller:
        logger.warning(f"unknown caller {caller}")

    if caller_version not in get_valid_caller_versions(caller):
        logger.warning(f"invalid caller version {caller_version} for {caller}")

    project_meta = hl.Struct(
        **{
            "project": project_name,
            "date": datetime.today().strftime("%Y-%m-%d"),
            "caller": caller.lower(),
            "caller_version": caller_version,
            "gnomad_meb": hl.Struct(
                **{"version": get_version(), "revision": get_git_revision()}
            ),
        }
    )
    return project_meta
