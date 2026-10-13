# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1", "openpyxl==3.1.5", "rdkit==2025.9.4"]
# ///
"""
Two-gate classification with machine-learned instead of DFT quencher parameters.

Reads   ../../data/exp_dataset_ent.csv (also through pairs.py)
        ../../data/compound_identification.xlsx          (quencher SMILES)
        ../../ml/data/training_dataset_full.csv, ../../ml/data/splits/*.csv
        results/quencher_classes/quencher_classes.csv
Writes  results/ml_vs_dft/training_overlap.csv  per quencher (all 48): identical structure in the
                                                training data, its split in every data split and
                                                its training targets; largest Tanimoto similarity
                                                to a training molecule
        results/ml_vs_dft/quenchers.csv         per quencher: ML - DFT parameters, summed ML
                                                uncertainty, pairs classified correctly with DFT
                                                and with ML, pairs rescued and lost, and the
                                                change in the within-PC AUC when only this
                                                quencher gets DFT parameters
        results/ml_vs_dft/parameter_swaps.csv   two-gate ML with E_T(Q) or both lambda(Q) from DFT
        results/ml_vs_dft/classes.csv           predictions per quencher class
        results/ml_vs_dft/uncertainty.csv       DFT fallback for the quenchers with the largest
                                                summed ML uncertainty or the largest effect
        results/ml_vs_dft/summary.csv           counts and correlations over pairs and quenchers

    uv run benchmark/further_analysis/ml_vs_dft.py

The criterion is two-gate with r from DFT sizes; the photocatalyst parameters are DFT in
both cases. "ML" uses E_T(Q), lambda_S0-T1(Q) and lambda_T1-S0(Q) from the model, "DFT"
the DFT values; a swap takes the named quencher parameters (for all pairs, or for the
pairs of the named quenchers) from DFT and the others from ML. A pair is rescued when ML
classifies it correctly and DFT does not, and lost in the opposite case. The summed ML
uncertainty is the sum of the three predictive standard deviations of a quencher.

Uncertainty fallback: among the scored quenchers, the FLAG_FRACTION with the largest
summed uncertainty, or with the largest within-PC AUC gain when they alone get DFT
parameters (an upper bound that uses the outcomes), get DFT parameters.

Training overlap: structures are compared as canonical SMILES, then without
stereochemistry, then as canonical tautomers; the Tanimoto similarity uses Morgan
fingerprints (radius 2, 2048 bits).
"""

import sys
from functools import cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import classification  # noqa: E402
import pairs  # noqa: E402
import scoring  # noqa: E402

ROOT = HERE.parent.parent
IDENTIFICATION = ROOT / "data" / "compound_identification.xlsx"
TRAINING = ROOT / "ml" / "data" / "training_dataset_full.csv"
SPLITS = ROOT / "ml" / "data" / "splits"
CLASSES = HERE / "results" / "quencher_classes" / "quencher_classes.csv"
RESULTS = HERE / "results" / "ml_vs_dft"

TWO_GATE = classification.candidates("two_gate", "r from DFT sizes")[0]
PARAMETERS = {"E_T": "q_E_T", "lambda_S0_T1": "q_lambda_S0_T1", "lambda_T1_S0": "q_lambda_T1_S0"}
FLAG_FRACTION = 0.25
SWAPS = {"DFT": ("E_T", "lambda_S0_T1", "lambda_T1_S0"), "ML": (),
         "ML, E_T from DFT": ("E_T",), "ML, both lambda from DFT": ("lambda_S0_T1", "lambda_T1_S0")}
TRAINING_TARGETS = {"E_T": "adiabatic_energy", "lambda_S0_T1": "lambda_acceptor_singlet",
                    "lambda_T1_S0": "lambda_donor_triplet"}


# ----------------------------------------------------------------------------- training overlap
@cache
def _rdkit():
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    return (Chem, DataStructs, rdMolStandardize.TautomerEnumerator(),
            rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048))


def structure_keys(smiles):
    """(canonical, no stereo, canonical tautomer, Morgan fingerprint) of one SMILES."""
    Chem, _, tautomers, morgan = _rdkit()
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None, None, None, None
    canonical = Chem.MolToSmiles(m)
    no_stereo = Chem.MolToSmiles(m, isomericSmiles=False)
    try:
        tautomer = Chem.MolToSmiles(tautomers.Canonicalize(Chem.MolFromSmiles(no_stereo)), isomericSmiles=False)
    except Exception:
        tautomer = no_stereo
    return canonical, no_stereo, tautomer, morgan.GetFingerprint(m)


def training_overlap(quenchers, training, splits, q):
    """quenchers: quencher_id, smiles; training: idx, smiles and targets; splits: name -> split per row."""
    DataStructs = _rdkit()[1]
    keys = [structure_keys(s) for s in training.smiles]
    index = [{} for _ in range(3)]
    for i, k in enumerate(keys):
        for j in range(3):
            index[j].setdefault(k[j], []).append(i)
    usable = [i for i, k in enumerate(keys) if k[3] is not None]
    fingerprints = [keys[i][3] for i in usable]
    rows = []
    for _, r in quenchers.iterrows():
        k = structure_keys(r.smiles)
        hit = index[0].get(k[0]) or index[1].get(k[1]) or index[2].get(k[2]) or []
        match = ("exact" if index[0].get(k[0]) else "no stereo" if index[1].get(k[1])
                 else "tautomer" if index[2].get(k[2]) else "")
        sims = np.array(DataStructs.BulkTanimotoSimilarity(k[3], fingerprints))
        j = int(np.argmax(sims))
        name = f"Q{r.quencher_id}"
        row = {"quencher": name, "smiles": r.smiles, "match": match, "n_matches": len(hit),
               "max_tanimoto": sims[j], "nearest_training_smiles": training.smiles.iloc[usable[j]]}
        for split, labels in splits.items():
            row[f"split {split}"] = "/".join(sorted(set(labels[hit]))) if hit else ""
        for p, column in TRAINING_TARGETS.items():
            row[f"training {p} / kcal mol-1"] = training[column].iloc[hit[0]] if hit else np.nan
            row[f"dft {p} / kcal mol-1"] = q.loc[name, f"{PARAMETERS[p]}_dft / kcal mol-1"] if hit else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def load_training():
    training = pd.read_csv(TRAINING, usecols=["idx", "smiles", *TRAINING_TARGETS.values()])
    splits = {}
    for path in sorted(SPLITS.glob("*.csv")):
        s = pd.read_csv(path)
        assert np.array_equal(s.idx.to_numpy(), training.idx.to_numpy()), path.name
        splits[path.stem] = s.split.to_numpy()
    return training, splits


# ----------------------------------------------------------------------------- classification
def with_dft(df, parameters, rows=None):
    """df whose ML quencher `parameters` are the DFT ones (on `rows` only, if given)."""
    d = df.copy()
    rows = np.ones(len(df), bool) if rows is None else rows
    for p in parameters:
        ml, dft = ((pairs.dg_ent(s) if p == "E_T" else f"{PARAMETERS[p]}_{s} / kcal mol-1") for s in ("ml", "dft"))
        d[ml] = np.where(rows, df[dft], df[ml])
    return d


def two_gate(df):
    return classification.predict(df, TWO_GATE, "ml")


def pooled_within(y, score, groups):
    return scoring.auc_within(y, score, groups)["pooled"]


def quencher_table(df, q, classes, overlap):
    y, g = df[pairs.QUENCHES].to_numpy(bool), df[pairs.PC_NAME].to_numpy()
    p_dft, _ = classification.predict(df, TWO_GATE, "dft")
    p_ml, s_ml = classification.predict(df, TWO_GATE, "ml")
    base = pooled_within(y, s_ml, g)
    names = df[pairs.Q_NAME].to_numpy()
    rows = []
    for name in q.index:
        r = {"quencher": name, "class": classes.get(name)}
        for p, col in PARAMETERS.items():
            r[f"{p} ML - DFT / kcal mol-1"] = q.loc[name, f"{col}_ml / kcal mol-1"] - q.loc[name, f"{col}_dft / kcal mol-1"]
        r["ML uncertainty sum / kcal mol-1"] = sum(q.loc[name, pairs.q_ml_uncertainty(p)] for p in PARAMETERS)
        r["max_tanimoto"] = overlap.loc[name, "max_tanimoto"]
        r["in_training_data"] = overlap.loc[name, "match"] != ""
        m = names == name
        r["n"] = int(m.sum())
        if m.any():
            yy, pd_, pm = y[m], p_dft[m], p_ml[m]
            _, s_x = two_gate(with_dft(df, PARAMETERS, m))
            r |= {"n_quench": int(yy.sum()), "correct_dft": int((pd_ == yy).sum()), "correct_ml": int((pm == yy).sum()),
                  "TP_dft": int((pd_ & yy).sum()), "TP_ml": int((pm & yy).sum()),
                  "FP_dft": int((pd_ & ~yy).sum()), "FP_ml": int((pm & ~yy).sum()),
                  "rescued_quench": int(((pm == yy) & (pd_ != yy) & yy).sum()),
                  "rescued_no_quench": int(((pm == yy) & (pd_ != yy) & ~yy).sum()),
                  "lost_quench": int(((pm != yy) & (pd_ == yy) & yy).sum()),
                  "lost_no_quench": int(((pm != yy) & (pd_ == yy) & ~yy).sum()),
                  "dAUC_within_PC_pooled_if_dft": pooled_within(y, s_x, g) - base}
            r["dcorrect_ml_minus_dft"] = r["correct_ml"] - r["correct_dft"]
        rows.append(r)
    t = pd.DataFrame(rows)
    order = ["quencher", "class", *[c for c in t.columns if "ML - DFT" in c], "ML uncertainty sum / kcal mol-1",
             "max_tanimoto", "in_training_data", "n", "n_quench", "correct_dft", "correct_ml", "dcorrect_ml_minus_dft",
             "TP_dft", "TP_ml", "FP_dft", "FP_ml", "rescued_quench", "rescued_no_quench", "lost_quench",
             "lost_no_quench", "dAUC_within_PC_pooled_if_dft"]
    t = t[order]
    for c in order[order.index("n_quench"):-1]:
        t[c] = t[c].astype("Int64")
    return t


def parameter_swaps(df):
    y, g = df[pairs.QUENCHES].to_numpy(bool), df[pairs.PC_NAME].to_numpy()
    rows = []
    for name, from_dft in SWAPS.items():
        predicted, score = two_gate(with_dft(df, from_dft))
        rows.append({"quencher_parameters": name, **scoring.metrics(y, predicted),
                     "AUC_within_PC_pooled": pooled_within(y, score, g)})
    t = pd.DataFrame(rows)
    assert np.isclose(t.MCC.iloc[0], scoring.metrics(y, classification.predict(df, TWO_GATE, "dft")[0])["MCC"])
    return t


def class_table(df, classes):
    y = df[pairs.QUENCHES].to_numpy(bool)
    p_dft = classification.predict(df, TWO_GATE, "dft")[0]
    p_ml = classification.predict(df, TWO_GATE, "ml")[0]
    cls = df[pairs.Q_NAME].map(classes).to_numpy()
    rows = []
    for c in ("rigid", "flexible", "dissociative"):
        m = cls == c
        rows.append({"class": c, "n": int(m.sum()), "n_quench": int(y[m].sum()),
                     "predicted_dft": int(p_dft[m].sum()), "predicted_ml": int(p_ml[m].sum()),
                     "TP_dft": int((p_dft & y)[m].sum()), "TP_ml": int((p_ml & y)[m].sum()),
                     "correct_dft": int((p_dft == y)[m].sum()), "correct_ml": int((p_ml == y)[m].sum())})
    return pd.DataFrame(rows)


def uncertainty(df, quenchers):
    """DFT fallback for the most uncertain and for the most effective quenchers."""
    y, g = df[pairs.QUENCHES].to_numpy(bool), df[pairs.PC_NAME].to_numpy()
    scored = quenchers[quenchers.n > 0].set_index("quencher")
    k = int(round(FLAG_FRACTION * len(scored)))
    flagged = {
        "largest ML uncertainty sum": list(scored["ML uncertainty sum / kcal mol-1"]
                                           .sort_values(ascending=False, kind="stable").index[:k]),
        "largest effect (uses the outcomes)": list(scored["dAUC_within_PC_pooled_if_dft"]
                                                   .sort_values(ascending=False, kind="stable").index[:k])}
    p_ml, s_ml = two_gate(df)
    rows = [{"fallback": "none", "n_quenchers": 0, "quenchers": "", "MCC": scoring.metrics(y, p_ml)["MCC"],
             "AUC_within_PC_pooled": pooled_within(y, s_ml, g)}]
    for name, qs in flagged.items():
        predicted, score = two_gate(with_dft(df, PARAMETERS, df[pairs.Q_NAME].isin(qs).to_numpy()))
        rows.append({"fallback": name, "n_quenchers": len(qs), "quenchers": " ".join(qs),
                     "MCC": scoring.metrics(y, predicted)["MCC"],
                     "AUC_within_PC_pooled": pooled_within(y, score, g)})
    both = [q for q in flagged["largest ML uncertainty sum"] if q in flagged["largest effect (uses the outcomes)"]]
    rows.append({"fallback": "in both", "n_quenchers": len(both),
                 "quenchers": " ".join(sorted(both, key=lambda s: int(s[1:])))})
    return pd.DataFrame(rows)


def summary(df, quenchers, overlap, q):
    y = df[pairs.QUENCHES].to_numpy(bool)
    p_dft = classification.predict(df, TWO_GATE, "dft")[0]
    p_ml = classification.predict(df, TWO_GATE, "ml")[0]
    rescued, lost = (p_ml == y) & (p_dft != y), (p_ml != y) & (p_dft == y)
    scored = quenchers[quenchers.n > 0].set_index("quencher")
    dc = scored.dcorrect_ml_minus_dft.astype(int)
    losses = dc[dc < 0].sort_values(kind="stable")
    other = overlap[overlap.match == ""]
    abs_dev = sum((q[f"{c}_ml / kcal mol-1"] - q[f"{c}_dft / kcal mol-1"]).abs() for c in PARAMETERS.values())
    sigma = sum(q[pairs.q_ml_uncertainty(p)] for p in PARAMETERS)
    rows = [
        ("pairs", len(y), ""),
        ("correct, DFT", int((p_dft == y).sum()), ""), ("correct, ML", int((p_ml == y).sum()), ""),
        ("rescued by ML", int(rescued.sum()), ""), ("rescued by ML, quenching", int((rescued & y).sum()), ""),
        ("rescued by ML, non-quenching", int((rescued & ~y).sum()), ""),
        ("quenchers with rescued pairs", int((scored.rescued_quench + scored.rescued_no_quench > 0).sum()), ""),
        ("lost by ML", int(lost.sum()), ""), ("lost by ML, quenching", int((lost & y).sum()), ""),
        ("lost by ML, non-quenching", int((lost & ~y).sum()), ""),
        ("quenchers with lost pairs", int((scored.lost_quench + scored.lost_no_quench > 0).sum()), ""),
        ("net change in correct pairs, quenchers losing", int(-losses.sum()), ""),
        ("quenchers with a net loss", len(losses), ""),
        ("net change in correct pairs, quenchers gaining", int(dc[dc > 0].sum()), ""),
        ("quenchers with a net gain", int((dc > 0).sum()), ""),
        ("net loss of the five quenchers losing most", int(-losses.iloc[:5].sum()), " ".join(losses.index[:5])),
        ("quenchers in the training data", int((overlap.match != "").sum()),
         " ".join(overlap.loc[overlap.match != "", "quencher"])),
        ("median max_tanimoto, quenchers not in the training data", float(other.max_tanimoto.median()),
         f"n = {len(other)}"),
        ("Spearman rho, ML uncertainty sum vs within-PC AUC gain if DFT, scored quenchers",
         float(stats.spearmanr(scored["ML uncertainty sum / kcal mol-1"], scored.dAUC_within_PC_pooled_if_dft)[0]),
         f"n = {len(scored)}"),
        ("Spearman rho, ML uncertainty sum vs summed |ML - DFT|, all quenchers",
         float(stats.spearmanr(sigma, abs_dev)[0]), f"n = {len(q)}"),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value", "detail"])


def rounded(table):
    """kcal/mol and Tanimoto similarities to 2 decimals, everything else to 3; in
    quantity/value tables by row."""
    table = table.copy()
    if {"quantity", "value"} <= set(table.columns):
        table["value"] = [round(float(v), 2 if "kcal" in qq else 3) for qq, v in zip(table.quantity, table.value)]
    for c in table.select_dtypes(float).columns:
        if c != "value":
            table[c] = table[c].round(2 if "kcal" in c or "tanimoto" in c else 3)
    return table


def main():
    q = pd.read_csv(pairs.DATASET).groupby(pairs.Q_NAME).first()
    q = q.sort_index(key=lambda s: s.str[1:].astype(int))
    classes = pd.read_csv(CLASSES).set_index("id")["proposed_class"]
    ids = pd.read_excel(IDENTIFICATION, sheet_name="Quenchers")
    ids = ids[ids.quencher_id.isin(q.quencher_id)].sort_values("quencher_id").reset_index(drop=True)
    training, splits = load_training()
    overlap = training_overlap(ids, training, splits, q)
    df = pairs.scored(pairs.load())
    quenchers = quencher_table(df, q, classes, overlap.set_index("quencher"))
    tables = {"training_overlap": overlap, "quenchers": quenchers, "parameter_swaps": parameter_swaps(df),
              "classes": class_table(df, classes), "uncertainty": uncertainty(df, quenchers),
              "summary": summary(df, quenchers, overlap, q)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        rounded(table).to_csv(RESULTS / f"{name}.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 250)
    for name in ("parameter_swaps", "classes", "uncertainty", "summary"):
        print(rounded(tables[name]).to_string(index=False), "\n")
    print(f"-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
