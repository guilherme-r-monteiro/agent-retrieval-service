import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DEVICE = "cpu"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    device: str

    @classmethod
    def from_environment(cls) -> "Settings":
        data_dir = os.environ.get("RETRIEVAL_DATA_DIR")
        if not data_dir:
            raise RuntimeError("RETRIEVAL_DATA_DIR is required")
        return cls(Path(data_dir), os.environ.get("RETRIEVAL_DEVICE", DEFAULT_DEVICE))
