"""Feature-set ablations use matching train and test tract identities."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from experiments.ablation_nyc import align_ablation_datasets


def ablation_inputs():
    osm = pd.DataFrame(
        {
            "tract_id": ["a", "b", "c", "d", "missing"],
            "median_income": [100, 200, 300, 400, np.nan],
            "poi_count": [1, 2, 3, 4, 0],
        }
    )
    enriched = osm.iloc[[3, 0, 2, 4]].copy()
    enriched["complaints_total_density"] = [4, 1, 3, 0]
    return osm, enriched


def test_ablation_aligns_different_row_orders_and_coverage_before_splitting():
    osm, enriched = align_ablation_datasets(*ablation_inputs())
    assert osm.tract_id.tolist() == enriched.tract_id.tolist() == ["a", "c", "d"]
    assert osm.median_income.tolist() == enriched.median_income.tolist()
    osm_train, osm_test = train_test_split(osm, test_size=0.34, random_state=42)
    enriched_train, enriched_test = train_test_split(enriched, test_size=0.34, random_state=42)
    assert osm_train.tract_id.tolist() == enriched_train.tract_id.tolist()
    assert osm_test.tract_id.tolist() == enriched_test.tract_id.tolist()
    assert enriched.complaints_total_density.tolist() == [1, 3, 4]


def test_ablation_rejects_conflicting_observations():
    osm, enriched = ablation_inputs()
    enriched.loc[enriched.tract_id == "a", "median_income"] = 999
    with pytest.raises(ValueError, match="conflicting observed"):
        align_ablation_datasets(osm, enriched)


@pytest.mark.parametrize("invalid_id", [None, "a"])
def test_ablation_rejects_missing_or_duplicate_ids(invalid_id):
    osm, enriched = ablation_inputs()
    osm.loc[1, "tract_id"] = invalid_id
    with pytest.raises(ValueError, match="tract IDs"):
        align_ablation_datasets(osm, enriched)


def test_ablation_rejects_no_shared_observations():
    osm, enriched = ablation_inputs()
    enriched["tract_id"] = ["w", "x", "y", "z"]
    with pytest.raises(ValueError, match="shared observed tracts"):
        align_ablation_datasets(osm, enriched)
