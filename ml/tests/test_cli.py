"""The `ent-ml` command: what it parses, what it prints, and how it fails.

Every case but the last stubs the forward pass -- the model is exercised once, at the
bottom, and the rest is argument handling and output shape.
"""

import json
import subprocess
import sys

import pytest

from ml import cli
from ml.predict import PC_COLUMNS, MolecularProperties, RatePredictor

BENZOPHENONE = "c1ccccc1C(=O)c1ccccc1"


@pytest.fixture
def stub_model(monkeypatch):
    """Answer for any structure, without loading the checkpoint."""

    def properties(self, smiles):
        for s in smiles:
            self._cache.setdefault(s, MolecularProperties(
                smiles=s, E_T=61.0, lambda_S_T=12.0, lambda_T_S=9.0,
                E_T_uncertainty=1.5, lambda_S_T_uncertainty=0.5,
                lambda_T_S_uncertainty=0.4))
        return [self._cache[s] for s in smiles]

    monkeypatch.setattr(RatePredictor, "properties", properties)


@pytest.fixture
def pc_table(tmp_path):
    path = tmp_path / "photocatalysts.csv"
    path.write_text("new_id," + ",".join(PC_COLUMNS) + "\n1,65.7,12.3,9.8\n",
                    encoding="utf-8")
    return path


# -- predict --------------------------------------------------------------------

def test_predict_prints_a_table(stub_model, capsys):
    cli.main(["predict", "c1ccccc1"])

    header, row = capsys.readouterr().out.splitlines()
    assert header.split() == ["smiles", "E_T", "E_T_uncertainty", "lambda_S_T",
                              "lambda_S_T_uncertainty", "lambda_T_S",
                              "lambda_T_S_uncertainty"]
    assert row.split() == ["c1ccccc1", "61", "1.5", "12", "0.5", "9", "0.4"]


def test_predict_csv_has_one_row_per_structure(stub_model, capsys):
    cli.main(["predict", "c1ccccc1", "CCO", "--format", "csv"])

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("smiles,E_T,")
    assert len(lines) == 3


def test_predict_json_carries_numbers(stub_model, capsys):
    cli.main(["predict", "c1ccccc1", "--format", "json"])

    rows = json.loads(capsys.readouterr().out)
    assert rows == [{"smiles": "c1ccccc1", "E_T": 61.0, "E_T_uncertainty": 1.5,
                     "lambda_S_T": 12.0, "lambda_S_T_uncertainty": 0.5,
                     "lambda_T_S": 9.0, "lambda_T_S_uncertainty": 0.4}]


def test_predict_reads_a_file_and_writes_one(stub_model, tmp_path):
    listing = tmp_path / "quenchers.smi"
    listing.write_text("c1ccccc1\n\nCCO\n", encoding="utf-8")
    out = tmp_path / "out.csv"

    cli.main(["predict", "--file", str(listing), "--format", "csv", "--out", str(out)])

    assert len(out.read_text(encoding="utf-8").splitlines()) == 3


def test_predict_refuses_an_invalid_smiles(stub_model):
    with pytest.raises(SystemExit) as exited:
        cli.main(["predict", "not-a-molecule"])

    assert "not a valid SMILES" in str(exited.value)


def test_predict_reports_a_missing_file(stub_model, tmp_path):
    with pytest.raises(SystemExit) as exited:
        cli.main(["predict", "--file", str(tmp_path / "absent.smi")])

    assert str(exited.value).startswith("ent-ml:")


def test_predict_needs_something_to_predict(stub_model):
    with pytest.raises(SystemExit) as exited:
        cli.main(["predict"])

    assert "no SMILES given" in str(exited.value)


# -- rate -----------------------------------------------------------------------

def test_rate_reports_the_pair(stub_model, pc_table, capsys):
    cli.main(["rate", "--quencher", "c1ccccc1", "--photocatalysts", str(pc_table),
              "--photocatalyst", "1", "--format", "json"])

    row, = json.loads(capsys.readouterr().out)
    assert row["photocatalyst"] == "PC1" and row["quencher"] == "c1ccccc1"
    assert row["dG_kcal"] == pytest.approx(61.0 - 65.7)
    assert row["k_ent"] > 0 and row["quenches"] is True


def test_rate_scales_the_reorganization_energies(stub_model, pc_table, capsys):
    def barrier(scale):
        cli.main(["rate", "--quencher", "c1ccccc1", "--photocatalysts", str(pc_table),
                  "--photocatalyst", "1", "--lambda-scale", scale, "--format", "json"])
        return json.loads(capsys.readouterr().out)[0]["dG_barrier_kcal"]

    assert barrier("0.4") < barrier("1.0")


def test_rate_reports_an_unknown_photocatalyst(stub_model, pc_table):
    with pytest.raises(SystemExit) as exited:
        cli.main(["rate", "--quencher", "c1ccccc1", "--photocatalysts", str(pc_table),
                  "--photocatalyst", "99"])

    assert "PC99" in str(exited.value)


# -- info, and what the command costs to start ----------------------------------

def test_info_names_the_checkpoint_and_the_vocabulary(capsys):
    cli.main(["info"])

    out = capsys.readouterr().out
    assert "model.pt" in out and "epoch" in out and "val loss" in out
    assert out.splitlines()[-1].split()[1:] == [
        "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "Sn", "I"]


def test_importing_the_cli_does_not_import_torch():
    probe = "import ml.cli, sys; print('torch' in sys.modules)"

    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)

    assert done.stdout.strip() == "False", done.stderr


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exited:
        cli.main(["--help"])

    assert exited.value.code == 0
    assert "predict" in capsys.readouterr().out


# -- one invocation that actually runs the model ---------------------------------

def test_predict_runs_the_model(capsys):
    cli.main(["predict", BENZOPHENONE, "--format", "json"])

    row, = json.loads(capsys.readouterr().out)
    assert row["smiles"] == BENZOPHENONE
    assert 30.0 < row["E_T"] < 75.0
    assert row["E_T_uncertainty"] > 0
    assert row["lambda_S_T"] > 0 and row["lambda_T_S"] > 0
