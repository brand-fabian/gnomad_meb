"""The ``Serializer`` ABC: read/write/metadata for one artifact kind.

Every subclass implements ``read``, ``write`` and ``metadata``. The kind's
serializer is the one place that knows how its data is written, whether it is
``is_checkpoint()``/``is_restorable()``, and how its completeness (``_SUCCESS``,
single file, S3) is probed.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar

from ..metadata import CheckpointMetadata

T = TypeVar("T")


class Serializer[T](ABC):
    """Checkpoint Serializer base class.

    The base class that defines the necessary for serializers for checkpoints
    for the population genetics pipelines.

    Every subclass should implement the `read`, `write` and `metadata` endpoints.
    Metadata is used to write a .json companion file to detail the origin of
    every checkpoint.

    Arguments
    ---------
    path : str | Path
           Path to the checkpoint data.
    data : T
           Data to write to the checkpoint, or None if the checkpoint should
           be read from.
    overwrite : bool
                If true, this permits overwriting incomplete checkpoint data.
    """

    def __init__(
        self, path: str | Path, data: T | None = None, overwrite: bool = False
    ):
        if isinstance(path, Path):
            path = str(path.resolve())
        self.path: str = path
        self.data: T = data
        self.overwrite: bool = overwrite

    @classmethod
    @abstractmethod
    def read(cls, path: str | Path, **kwargs) -> T:
        """Read the data at path into a valid instance of the serialized type."""
        pass

    @abstractmethod
    def write(self, **kwargs):
        """Serialize the current instance to the given path."""
        pass

    @abstractmethod
    def metadata(self, source: str | None = None) -> CheckpointMetadata:
        """Get a metadata of the instance at path.

        Arguments
        ---------
        source : str
                 Source lines of the function that generated the serialized data
                 instance.
        """
        pass

    def exists(self) -> bool:
        """Test, whether the current path exists."""
        return self.metadata().exists

    def is_checkpoint(self) -> bool:
        """True, if reading and writing using this serializer returns the exact same object representation.

        This flag indicates whether the exact same object structure is recreated when this serializer reads
        a file. For example, writing a pandas dataframe to csv can change dtypes or object representations,
        so should set this to false.
        """
        return True

    def is_restorable(self) -> bool:
        """True, if the data written by this serializer can be restored.

        This flag indicates, whether the data encoding represented by this serializer can be used to restore
        a similar object in memory from reading the file. This should be true in most cases, except e.g. image
        files where the object structure can not be recreated from the output file.
        """
        return True
