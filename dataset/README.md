# dataset — the pair-level table

This is where processed measurements and processed calculations become one row per
photocatalyst–quencher pair. Everything downstream — benchmarks, models, figures —
reads this table and nothing else.

```
data/experimental_data_processed/   ┐
                                    ├──> build_dataset.py ──> pairs.csv   (1296 rows)
data/computational_data_processed/  ┘
```

Nothing is read from outside `data/`. In particular, no XLSX is consulted at runtime
and no file is copied in as an import side effect.

## The central quantity

```
k_q = K_SV / tau_0
```

`K_SV` per pair from `stern_volmer_fits.csv`, `tau_0` per photocatalyst from
`lifetimes_tau0.csv`. Both are measured, both trace back to instrument files.

## Two things this table must not smooth over

**Half the dataset is left-censored.** 662 of 1296 pairs (51.1 %) fall below the
detection floor `K_SV >= 25 M-1 and r2 >= 0.90`, and 379 of those have a negative
fitted slope. They are one-sided observations, not missing values, and they belong in
any analysis as non-quenchers. No redox value and no DFT energy can turn a
non-quencher into missing data. In rate terms the floor is photocatalyst-specific,
3.9e6 to 3.4e8 M-1 s-1, because it scales as `25 / tau_0`.

**Label on k_q, not K_SV.** `tau_0` spans 86x across the screen, so a fixed K_SV
cutoff would demand an 86x higher rate constant from the short-lived photocatalysts.
Marcus and Rehm-Weller predict k_q.

## Status

`build_dataset.py` is not written yet. It needs the literature redox potentials and
extinction coefficients for the quenchers, which are the last input still outside this
repository. Everything else it needs is already in `data/`.

The photocatalyst DFT columns will stay empty until those job files arrive; the table
is built without them rather than waiting.

## Column naming

| suffix | meaning |
|---|---|
| `_exp` | measured here, or from the literature with a source recorded |
| `_dft` | computed by DFT (four-point scheme) |
| `_md` | from Milo molecular-dynamics sampling (DvTE only) |
| `_ml` | predicted by a model |

Columns with no suffix are identifiers, instrument settings, or arithmetic
derivations. The unit follows the name after a slash, as `k_q / M-1 s-1`.

## Sign conventions

All quencher-centric, negative = exergonic, `F = 23.0605 kcal mol-1 V-1`:

```
dG_EnT     = E_T(Q) - E_T(PC)
dG_SET_ox  = F * (E_Ox(Q) - E_ox_ES(PC))      Q oxidized, PC* is the oxidant
dG_SET_red = F * (E_red_ES(PC) - E_Red(Q))    Q reduced,  PC* is the reductant
dG_SET     = min(dG_SET_ox, dG_SET_red)
```

Potentials are named by their couple, not by direction, because the naming is
quencher-centric and inverts easily: `E_ox_ES` is the potential at which PC\*
*oxidizes a substrate*, i.e. the reduction potential of the PC\*/PC- couple.
