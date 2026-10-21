from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import requests


@dataclass
class CensusAPI:
    """
    Minimal Census ACS 5-year client.

    Notes:
    - For small requests, API key is optional.
    - NYC needs multiple county requests (Bronx, Kings, New York, Queens, Richmond).
    """

    api_key: Optional[str] = None
    timeout_s: int = 30
    sleep_s: float = 0.2  # polite rate limiting

    BASE_URL = "https://api.census.gov/data"

    def _get(self, url: str, params: Dict[str, str]) -> list:
        if self.api_key:
            params = dict(params)
            params["key"] = self.api_key

        r = requests.get(url, params=params, timeout=self.timeout_s)
        r.raise_for_status()
        time.sleep(self.sleep_s)
        return r.json()

    def acs5_tract(
        self,
        year: int,
        variables: List[str],
        state_fips: str,
        county_fips: str,
    ) -> pd.DataFrame:
        """
        Fetch ACS 5-year estimates for ALL tracts in a given county.
        """
        url = f"{self.BASE_URL}/{year}/acs/acs5"
        get_vars = ["NAME"] + variables

        params = {
            "get": ",".join(get_vars),
            "for": "tract:*",
            "in": f"state:{state_fips} county:{county_fips}",
        }

        data = self._get(url, params)
        df = pd.DataFrame(data[1:], columns=data[0])

        # Ensure columns exist
        for c in ["state", "county", "tract"]:
            if c not in df.columns:
                raise ValueError(f"Missing expected column '{c}' from Census API response")

        df["tract_id"] = (
            df["state"].astype(str) + df["county"].astype(str) + df["tract"].astype(str)
        )
        return df

    def median_income_by_counties(
        self,
        year: int,
        state_fips: str,
        counties: List[str],
    ) -> pd.DataFrame:
        """
        Variable: B19013_001E = Median household income in past 12 months
        (inflation-adjusted dollars).
        """
        frames = []
        for c in counties:
            frames.append(self.acs5_tract(year, ["B19013_001E"], state_fips, c))
        df = pd.concat(frames, ignore_index=True)

        df = df.rename(columns={"B19013_001E": "median_income"})
        df["median_income"] = pd.to_numeric(df["median_income"], errors="coerce")

        # Negative/sentinel values -> NaN
        df.loc[df["median_income"] < 0, "median_income"] = pd.NA

        # Keep useful columns only
        keep = ["tract_id", "median_income", "NAME", "state", "county", "tract"]
        df = df[keep].drop_duplicates(subset=["tract_id"]).reset_index(drop=True)
        return df
