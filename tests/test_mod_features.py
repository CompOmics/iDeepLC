# Python
import pandas as pd

from ideeplc import utilities
from ideeplc.utilities import (
    MEAN_MOLLOGP,
    STD_MOLLOGP,
    build_user_mod_feature_table,
    mod_chemical_features,
)


def test_build_user_mod_feature_table(tmp_path):
    """Test building standardized modification features from a raw CSV."""
    input_csv = tmp_path / "user_mods.csv"
    output_csv = tmp_path / "user_mod_features_standardized.csv"

    pd.DataFrame(
        {
            "name": ["CustomMod"],
            "aa": ["K"],
            "smiles": ["CCO"],
        }
    ).to_csv(input_csv, index=False)

    feature_table = build_user_mod_feature_table(
        str(input_csv),
        str(output_csv),
        compute_mollogp_fn=lambda smiles: 1.0,
    )

    assert output_csv.exists()
    assert list(feature_table["name"]) == ["CustomMod#K"]
    expected = (1.0 - MEAN_MOLLOGP) / STD_MOLLOGP
    assert feature_table.iloc[0]["MolLogP_rdkit"] == expected


def test_mod_chemical_features_merges_user_table(tmp_path, monkeypatch):
    """Test that raw user mods are converted and merged automatically."""
    user_feature_csv = tmp_path / "user_mods.csv"
    pd.DataFrame(
        {
            "name": ["CustomMod"],
            "aa": ["K"],
            "smiles": ["CCO"],
        }
    ).to_csv(user_feature_csv, index=False)

    monkeypatch.setattr(utilities, "compute_mollogp", lambda smiles: 1.23)

    mod_dict = mod_chemical_features(user_mods_csv=str(user_feature_csv))

    assert "CustomMod" in mod_dict
    assert "K" in mod_dict["CustomMod"]
    expected = (1.23 - MEAN_MOLLOGP) / STD_MOLLOGP
    assert mod_dict["CustomMod"]["K"]["MolLogP_rdkit"] == expected
