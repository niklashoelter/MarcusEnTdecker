"""The photocatalyst table comes from the caller; the package ships none."""

import pytest

from ml.predict import PC_COLUMNS, PC_TAU0, photocatalyst_properties

ROW = {"new_id": 1, PC_COLUMNS[0]: 65.7, PC_COLUMNS[1]: 12.3, PC_COLUMNS[2]: 9.8}


def write_csv(path, rows=(ROW,)):
    columns = ["new_id", *PC_COLUMNS]
    lines = [",".join(columns)]
    lines += [",".join(str(row[c]) for c in columns) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_no_source_says_what_is_needed():
    with pytest.raises(ValueError) as raised:
        photocatalyst_properties()

    message = str(raised.value)
    assert "new_id" in message and PC_COLUMNS[0] in message
    # It must not send the reader to a script or a directory of this repository.
    assert "computational_data" not in message and "run `uv run" not in message


def test_reads_a_csv_path(tmp_path):
    properties = photocatalyst_properties(write_csv(tmp_path / "pc.csv"), 1)

    assert properties.smiles == "PC1"
    assert properties.E_T == 65.7
    assert properties.lambda_S_T == 12.3
    assert properties.lambda_T_S == 9.8
    assert properties.source == "dft"
    assert properties.E_T_uncertainty is None


def test_reads_a_dataframe():
    pandas = pytest.importorskip("pandas")

    properties = photocatalyst_properties(pandas.DataFrame([ROW]), 1)

    assert properties.E_T == 65.7


def test_reads_a_single_row_mapping_without_an_id():
    assert photocatalyst_properties(ROW).smiles == "PC1"


def test_reads_a_mapping_of_id_to_row():
    table = {7: {c: ROW[c] for c in PC_COLUMNS}}

    assert photocatalyst_properties(table, 7).smiles == "PC7"


def test_unknown_id_names_the_photocatalyst(tmp_path):
    with pytest.raises(KeyError, match="PC9"):
        photocatalyst_properties(write_csv(tmp_path / "pc.csv"), 9)


def test_missing_file_is_reported_as_such(tmp_path):
    with pytest.raises(FileNotFoundError):
        photocatalyst_properties(tmp_path / "absent.csv", 1)


def test_incomplete_row_refuses_to_predict():
    row = dict(ROW, **{PC_COLUMNS[2]: None})

    with pytest.raises(ValueError, match="incomplete"):
        photocatalyst_properties(row, 1)


def test_reads_the_lifetime_when_the_table_has_it(tmp_path):
    path = tmp_path / "pc.csv"
    columns = ["new_id", *PC_COLUMNS, PC_TAU0]
    path.write_text(",".join(columns) + "\n1,65.7,12.3,9.8,2387.8\n2,59.4,8.8,8.5,\n",
                    encoding="utf-8")

    assert photocatalyst_properties(path, 1).tau0_ns == 2387.8
    assert photocatalyst_properties(path, 2).tau0_ns is None


def test_a_blank_lifetime_in_a_dataframe_is_none():
    pandas = pytest.importorskip("pandas")

    table = pandas.DataFrame([dict(ROW, **{PC_TAU0: float("nan")})])

    assert photocatalyst_properties(table, 1).tau0_ns is None


def test_without_a_lifetime_column_tau0_is_none(tmp_path):
    assert photocatalyst_properties(write_csv(tmp_path / "pc.csv"), 1).tau0_ns is None


def test_the_module_does_not_name_the_repository_around_it():
    import ml.predict

    assert "../data" not in ml.predict.__doc__
    assert "benchmark" not in ml.predict.__doc__
