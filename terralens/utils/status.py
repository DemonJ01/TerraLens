from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


class DataStatus(Enum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    MISSING = "missing"


@dataclass
class DataChecker:
    data_root: Path = Path("data")

    def __post_init__(self):
        self.data_root = Path(self.data_root)

    def _raw(self, *parts: str) -> Path:
        return self.data_root / "raw" / Path(*parts)

    def _processed(self, *parts: str) -> Path:
        return self.data_root / "processed" / Path(*parts)

    @staticmethod
    def _table_status(path: Path, required: set[str], numeric: set[str] | None = None):
        """Recognize nonempty tables with usable columns, including legacy CSVs."""
        if not path.is_file():
            return DataStatus.MISSING
        try:
            df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
            if df.empty or not required.issubset(df.columns):
                return DataStatus.PARTIAL
            valid = df[list(required)].notna().all(axis=1)
            for col in numeric or set():
                valid &= np.isfinite(pd.to_numeric(df[col], errors="coerce"))
            return DataStatus.AVAILABLE if valid.any() else DataStatus.PARTIAL
        except (OSError, ValueError, TypeError, ImportError):
            return DataStatus.PARTIAL

    @staticmethod
    def _best_status(statuses):
        statuses = list(statuses)
        if DataStatus.AVAILABLE in statuses:
            return DataStatus.AVAILABLE
        return DataStatus.PARTIAL if DataStatus.PARTIAL in statuses else DataStatus.MISSING

    def check_tracts(self, city: str, year: int = 2021) -> DataStatus:
        path = self._raw("census", f"{city}_tracts_{year}.gpkg")
        if not path.is_file():
            return DataStatus.MISSING
        try:
            # GeoPackages are SQLite databases. Check registered feature tables
            # without requiring GeoPandas just to report data readiness.
            with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
                tables = db.execute(
                    "SELECT table_name FROM gpkg_contents WHERE data_type = 'features'"
                ).fetchall()
                for (table_name,) in tables:
                    identifier = '"' + table_name.replace('"', '""') + '"'
                    columns = {row[1] for row in db.execute(f"PRAGMA table_info({identifier})")}
                    if not {"tract_id", "GEOID"}.intersection(columns):
                        continue
                    if db.execute(f"SELECT 1 FROM {identifier} LIMIT 1").fetchone():
                        return DataStatus.AVAILABLE
        except (OSError, sqlite3.Error, TypeError):
            pass
        return DataStatus.PARTIAL

    def check_census(self, city: str, year: int = 2021) -> DataStatus:
        filenames = [
            f"{city}_income_{year}.parquet",
            f"{city}_income_{year}.csv",
            f"{city}_acs_targets_{year}.parquet",
            f"{city}_targets_{year}.parquet",
        ]
        return self._best_status(
            self._table_status(
                self._raw("census", filename), {"tract_id", "median_income"}, {"median_income"}
            )
            for filename in filenames
        )

    def check_osm(self, city: str) -> DataStatus:
        features = self.data_root / "features" / f"{city}_geo_features.parquet"
        status = self._table_status(features, {"tract_id"})
        if status == DataStatus.AVAILABLE:
            df = pd.read_parquet(features)
            numeric = df.drop(columns="tract_id").select_dtypes(include="number")
            if not numeric.empty and np.isfinite(numeric.to_numpy(dtype=float)).any():
                return DataStatus.AVAILABLE
            status = DataStatus.PARTIAL
        # Downloaded extracts and raw geometries still need feature aggregation.
        raw_files = [
            self._raw("osm", f"{city}.osm.pbf"),
            self._raw("osm", f"{city}_pois.gpkg"),
            self._raw("osm", f"{city}_roads.gpkg"),
        ]
        if any(path.is_file() and path.stat().st_size > 0 for path in raw_files):
            return DataStatus.PARTIAL
        return status

    def check_safety(self, city: str) -> DataStatus:
        status = self._table_status(
            self.data_root / "features" / f"{city}_311_features.parquet", {"tract_id"}
        )
        if status == DataStatus.AVAILABLE:
            return status
        legacy = self._raw("crime", f"{city}_311_complaints.csv")
        legacy_status = self._table_status(legacy, {"latitude", "longitude", "complaint_type"})
        if legacy_status == DataStatus.AVAILABLE:
            return legacy_status
        if city == "nyc":
            raw_status = self._best_status(
                self._table_status(path, {"latitude", "longitude", "complaint_type"})
                for path in self._raw("nyc311").glob("nyc311_last_*mo.parquet")
            )
            return self._best_status([status, legacy_status, raw_status])
        return self._best_status([status, legacy_status])

    def check_dataset(self, city: str) -> DataStatus:
        path = self._processed(f"{city}_dataset.parquet")
        status = self._table_status(
            path, {"tract_id", "median_income", "lat", "lon"}, {"median_income", "lat", "lon"}
        )
        if status != DataStatus.AVAILABLE:
            return status
        df = pd.read_parquet(path)
        # Smoke/sample artifacts exercise code but are not city source data.
        synthetic_ids = (
            df["tract_id"]
            .astype(str)
            .str.contains(
                r"(?:^|[_-])(?:dummy|sample|synthetic|smoke)(?:[_-]|$)", case=False, regex=True
            )
        )
        if synthetic_ids.any():
            return DataStatus.PARTIAL
        if "city" in df and not df["city"].dropna().eq(city).all():
            return DataStatus.PARTIAL
        return status

    def check_all(self, city: str, year: int = 2021) -> Dict[str, DataStatus]:
        return {
            "tracts": self.check_tracts(city, year),
            "census": self.check_census(city, year),
            "osm": self.check_osm(city),
            "safety": self.check_safety(city),
            "dataset": self.check_dataset(city),
        }

    def is_ready_geo_only(self, city: str, year: int = 2021) -> bool:
        status = self.check_all(city, year)
        return status["dataset"] == DataStatus.AVAILABLE or all(
            status[name] == DataStatus.AVAILABLE for name in ("tracts", "census", "osm")
        )
