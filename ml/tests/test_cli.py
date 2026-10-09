"""The `ent-ml` command: what it parses, what it prints, and how it fails.

Every case but the last stubs the forward pass -- the model is exercised once, at the
bottom, and the rest is argument handling and output shape.
"""

import json
import math
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


def test_rate_takes_the_reorganization_energies_as_predicted(stub_model, pc_table, capsys):
    from ml.energy.marcus_ent import calculate_driving_force_and_barrier

    cli.main(["rate", "--quencher", "c1ccccc1", "--photocatalysts", str(pc_table),
              "--photocatalyst", "1", "--format", "json"])

    row, = json.loads(capsys.readouterr().out)
    _, barrier, _ = calculate_driving_force_and_barrier(65.7, 12.3, 9.8, 61.0, 12.0, 9.0)
    assert row["dG_barrier_kcal"] == pytest.approx(barrier)


def test_rate_takes_the_prefactor_from_the_encounter_constant_at_7_a(stub_model, pc_table,
                                                                     capsys):
    from ml.energy.two_gate import K_BOLTZMANN, PLANCK, encounter_constant

    cli.main(["rate", "--quencher", "c1ccccc1", "--photocatalysts", str(pc_table),
              "--photocatalyst", "1", "--format", "json"])

    row, = json.loads(capsys.readouterr().out)
    rt = 1.987204259e-3 * 298.15
    escape_over_nu = 2.0e10 / encounter_constant(7.0) / (K_BOLTZMANN * 298.15 / PLANCK)
    expected = 2.0e10 / (1.0 + math.exp(row["dG_kcal"] / rt)
                         + escape_over_nu * math.exp(row["dG_barrier_kcal"] / rt))
    assert row["k_ent"] == pytest.approx(expected)


def test_rate_without_tau0_reports_the_barrier_only(stub_model, pc_table, capsys):
    cli.main(["rate", "--quencher", "c1ccccc1", "--photocatalysts", str(pc_table),
              "--photocatalyst", "1", "--format", "json"])

    row, = json.loads(capsys.readouterr().out)
    assert "two_gate" not in row and "t_pc_kcal" not in row


def test_rate_with_tau0_adds_the_two_gate_verdict(stub_model, pc_table, capsys):
    from ml.energy.two_gate import lifetime_barrier

    cli.main(["rate", "--quencher", "c1ccccc1", "--photocatalysts", str(pc_table),
              "--photocatalyst", "1", "--tau0", "1000", "--format", "json"])

    row, = json.loads(capsys.readouterr().out)
    assert row["t_pc_kcal"] == pytest.approx(lifetime_barrier(1000.0))
    assert row["two_gate"] is (row["dG_kcal"] <= 0
                               and row["dG_barrier_kcal"] <= row["t_pc_kcal"])


def test_rate_takes_tau0_from_the_table(stub_model, tmp_path, capsys):
    from ml.predict import PC_TAU0

    path = tmp_path / "photocatalysts.csv"
    path.write_text("new_id," + ",".join(PC_COLUMNS) + f",{PC_TAU0}\n1,65.7,12.3,9.8,1000\n",
                    encoding="utf-8")

    cli.main(["rate", "--quencher", "c1ccccc1", "--photocatalysts", str(path),
              "--format", "json"])

    row, = json.loads(capsys.readouterr().out)
    assert "two_gate" in row


def test_two_gate_verdict_follows_the_lifetime(stub_model):
    from ml.predict import photocatalyst_properties

    pc = {"new_id": 1, PC_COLUMNS[0]: 65.7, PC_COLUMNS[1]: 12.3, PC_COLUMNS[2]: 9.8}
    short = photocatalyst_properties(dict(pc, **{"tau0 / ns": 0.4}))
    long = photocatalyst_properties(dict(pc, **{"tau0 / ns": 1e6}))
    unknown = photocatalyst_properties(pc)

    predictor = RatePredictor()
    assert predictor.predict(short, "c1ccccc1").two_gate is False
    assert predictor.predict(long, "c1ccccc1").two_gate is True
    assert predictor.predict(unknown, "c1ccccc1").two_gate is None


def test_rate_refuses_a_lifetime_that_is_not_positive(stub_model, pc_table):
    with pytest.raises(SystemExit) as exited:
        cli.main(["rate", "--quencher", "c1ccccc1", "--photocatalysts", str(pc_table),
                  "--photocatalyst", "1", "--tau0", "0"])

    assert "tau0" in str(exited.value)


def test_rate_reports_an_unknown_photocatalyst(stub_model, pc_table):
    with pytest.raises(SystemExit) as exited:
        cli.main(["rate", "--quencher", "c1ccccc1", "--photocatalysts", str(pc_table),
                  "--photocatalyst", "99"])

    assert "PC99" in str(exited.value)


# -- info, and what the command costs to start ----------------------------------

def test_info_names_the_checkpoint_and_the_vocabulary(capsys):
    cli.main(["info"])

    out = capsys.readouterr().out
    assert "model_final.pt" in out and "split" in out and "epoch" in out
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
