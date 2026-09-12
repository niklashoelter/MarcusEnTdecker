from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Dict, Any
import pandas as pd
import importlib.resources as ir


@lru_cache(maxsize=1)
def load_photocatalysts_df() -> pd.DataFrame:
    """Load the packaged photocatalysts.csv into a DataFrame (cached)."""
    # Use importlib.resources to access data inside the installed package
    with ir.files("webapp.data").joinpath("photocatalysts.csv").open("rb") as f:
        df = pd.read_csv(f)
    # Normalize expected columns
    if "photocatalyst_name" not in df.columns:
        raise ValueError("photocatalysts.csv must contain a 'photocatalyst_name' column")
    return df


def list_photocatalyst_names() -> List[str]:
    """Return available photocatalyst names for selection."""
    df = load_photocatalysts_df()
    names = df["photocatalyst_name"].astype(str).fillna("").tolist()
    # Filter empties, keep order
    return [n for n in names if n.strip()]


def get_photocatalyst_row(name: str) -> Optional[Dict[str, Any]]:
    """Return a row dict for the given photocatalyst name, or None if not found."""
    if not name:
        return None
    df = load_photocatalysts_df()
    mask = df["photocatalyst_name"].astype(str) == str(name)
    if not bool(mask.any()):
        return None
    row = df.loc[mask].iloc[0]
    return row.to_dict()
