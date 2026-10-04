import importlib.resources as ir
from pathlib import Path

import numpy as np
import pytest

from ml import cli, ensemble
from ml.predict import RatePredictor
from ml.prediction import predict as single_predict

SMILES = ["c1ccccc1", "C=Cc1ccccc1", "O=C(c1ccccc1)c1ccccc1"]
MODEL_FINAL = Path(str(ir.files("ml.data").joinpath("model_final.pt")))


def _output(mean, sd):
    mean, sd = np.array([[mean]]), np.array([[sd]])
    mdn = {"mu": mean[..., None], "sigma": sd[..., None], "pi": np.ones((1, 1, 1))}
    head = {"predictions": mean, "uncertainties": sd, "mdn_params": mdn}
    return {"graph": head, "node": head}


def test_one_member_is_returned_unchanged():
    out = ensemble.combine([_output(60.0, 2.0)])["graph"]
    assert out["predictions"][0, 0] == 60.0 and out["uncertainties"][0, 0] == pytest.approx(2.0)
    assert out["model_spread"][0, 0] == 0.0


def test_two_members_combine_by_the_law_of_total_variance():
    out = ensemble.combine([_output(58.0, 1.0), _output(62.0, 3.0)])["graph"]
    assert out["predictions"][0, 0] == 60.0
    assert out["model_spread"][0, 0] == 2.0
    assert out["uncertainties"][0, 0] == pytest.approx(np.sqrt((1 + 9) / 2 + 4))
    assert out["mdn_params"]["pi"].sum() == pytest.approx(1.0)
    assert out["mdn_params"]["mu"].shape[-1] == 2


def test_an_ensemble_of_model_final_alone_is_the_deployed_model():
    alone = ensemble.Ensemble([MODEL_FINAL]).predict(SMILES)
    deployed = single_predict(SMILES)
    for column in ("triplet_energy_adiabatic", "lamda_S_T", "lamda_T_S"):
        assert np.allclose(alone[column], deployed[column], atol=1e-5)
        assert np.allclose(alone[f"{column}_uncertainty"], deployed[f"{column}_uncertainty"],
                           atol=1e-5)


def test_the_default_ensemble_is_the_five_random_split_models():
    paths = ensemble.default_checkpoints()
    assert [p.parent.name for p in paths] == [
        "random_42", "random_101", "random_143", "random_404", "random_666"]
    table = ensemble.predict(SMILES)
    assert (table["n_models"] == 5).all()
    for column in ("triplet_energy_adiabatic", "lamda_S_T", "lamda_T_S"):
        assert np.isfinite(table[column]).all()
        assert (table[f"{column}_uncertainty"] >= table[f"{column}_model_spread"]).all()
    assert [len(s) for s in table["spin_map_model_spread"]] == [len(s) for s in table["spin_map"]]


def test_a_member_that_does_not_match_its_md5_is_refused(tmp_path, monkeypatch):
    members = tmp_path / "ensemble_members.csv"
    members.write_text("split,checkpoint,md5\nrandom_42,ml/results/random_42/model.pt,0\n")
    monkeypatch.setattr(ensemble, "MEMBERS", members)
    with pytest.raises(ValueError):
        ensemble.default_checkpoints()


def test_rate_predictor_can_use_the_ensemble():
    props = RatePredictor(ensemble=True).properties(["c1ccccc1"])[0]
    assert props.source == "ensemble" and props.E_T_uncertainty > 0


def test_cli_predict_with_the_ensemble(capsys):
    cli.main(["predict", "--ensemble", "--format", "csv", "c1ccccc1"])
    header, row = capsys.readouterr().out.strip().splitlines()
    assert "E_T_model_spread" in header.split(",") and row.endswith(",5")
