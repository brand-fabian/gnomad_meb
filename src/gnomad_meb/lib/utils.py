import tomllib
from collections.abc import Generator
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as im_version
from pathlib import Path
from subprocess import check_output

from gnomad_meb.lib.constants import UNKNOWN
from gnomad_meb.logging import get_logger

logger = get_logger()


@lru_cache
def get_git_revision(default: str = UNKNOWN):
    """Get current git revision of the tool."""
    try:
        git_revision = (
            check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception as e:
        logger.error(f"failed to retrieve git revision: {e}")
        return default
    return git_revision


@lru_cache
def get_version(default: str = UNKNOWN) -> str:
    version = default
    try:
        version = im_version("gnomad_meb")
    except PackageNotFoundError:
        logger.debug("could not get package version from installed packages")
    if version == default:
        pyproject_toml = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_toml.exists() and pyproject_toml.is_file():
            pyproject_data = tomllib.load(pyproject_toml)
            if "project" in pyproject_data and "version" in pyproject_data["project"]:
                version = pyproject_data["project"]["version"]
            elif (
                "tool" in pyproject_data
                and "poetry" in pyproject_data["tool"]
                and "version" in pyproject_data["tool"]["poetry"]
            ):
                version = pyproject_data["tool"]["poetry"]["version"]
        else:
            logger.warning("did not find pyproject.toml file")
    return version


@lru_cache
def get_hostname(default: str = "localhost") -> str:
    hostname = default
    try:
        hostname = check_output(["hostname", "-f"]).decode("ascii").strip()
    except Exception as e:
        logger.error(f"failed to retrieve local hostname: {e}")
        return default
    return hostname


def glob(
    roots: list[str | Path],
    patterns: list[str] = [],
    endings: list[str] = [],
) -> Generator[str, None, None]:
    assert len(patterns) + len(endings) > 0, (
        "atleast one ending or pattern must be specified"
    )
    patterns = [
        *patterns,
        *[f"*{ending}" for ending in endings],
    ]
    logger.info(
        f"searching in {len(roots)} directories, matching any "
        f"of {len(patterns)} patterns."
    )

    for root in roots:
        if isinstance(root, str):
            root = Path(root)
        assert root.exists(), f"root path {root} does not exist"
        for child in root.iterdir():
            if child.is_file() and any(child.match(p) for p in patterns):
                yield str(child.resolve())
            if child.is_dir():
                yield from glob([child], patterns=patterns)
