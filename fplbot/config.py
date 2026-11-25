from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("FPLBOT_DATA_DIR", ROOT / "data"))
ENV_FILE = ROOT / ".env"

load_dotenv(ENV_FILE)

POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
SQUAD_SHAPE = {1: 2, 2: 5, 3: 5, 4: 3}
XI_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
XI_MAX = {1: 1, 2: 5, 3: 5, 4: 3}


@dataclass
class Config:
    raw: dict = field(default_factory=dict)

    def __getattr__(self, item):
        try:
            return self.raw[item]
        except KeyError as e:
            raise AttributeError(item) from e

    @property
    def entry_id(self) -> int | None:
        v = os.environ.get("FPL_ENTRY_ID")
        return int(v) if v else None


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:
        return Config(yaml.safe_load(f))
