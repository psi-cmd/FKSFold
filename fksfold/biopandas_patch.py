"""Utility to monkey-patch BioPandas so that missing mmCIF columns
are auto-filled. This avoids directly editing the site-packages code and keeps
our patch within the project repo.

Call `apply_patch()` once during package import to replace
`PandasMmcif._construct_df` with a version that back-fills missing `label_*/auth_*`
columns before Biopandas tries to cast dtypes.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd


def apply_patch() -> None:  # noqa: D401 – simple description
    """Apply monkey-patch to BioPandas if it is available.

    If BioPandas is not installed, the function is a no-op so that importing
    this module never crashes the surrounding code.
    """
    try:
        # Local import so that the whole project still works even if Biopandas
        # is missing from the environment in some contexts (e.g. doc builds).
        from biopandas.mmcif import pandas_mmcif as _pm
        from biopandas.mmcif.engines import (
            ANISOU_DF_COLUMNS,
            mmcif_col_types,
        )
        from biopandas.mmcif.mmcif_parser import load_cif_data
    except ModuleNotFoundError:
        # Biopandas not available – silently skip
        return

    # Prevent double-patching (idempotent)
    if getattr(_pm.PandasMmcif, "_fksfold_patch_applied", False):
        return

    def _construct_df_patched(self: _pm.PandasMmcif, text: str):  # type: ignore[name-defined]
        """Replacement for PandasMmcif._construct_df with column back-fill."""
        data = load_cif_data(text)
        data = data[list(data.keys())[0]]
        self.data = data

        full_df = (
            pd.DataFrame.from_dict(data["atom_site"], orient="index").transpose()
        )

        # Fill missing auth_/label_ columns expected by Biopandas
        for col, dtype in mmcif_col_types.items():
            # Handle label_* columns
            if col.startswith("label_"):
                if col not in full_df.columns or full_df[col].isna().all():
                    counterpart = col.replace("label_", "auth_")
                    if counterpart in full_df.columns:
                        full_df[col] = full_df[counterpart]
                    else:
                        full_df[col] = "" if dtype == str else pd.NA
            # Handle auth_* columns
            if col.startswith("auth_"):
                if col not in full_df.columns or full_df[col].isna().all():
                    counterpart = col.replace("auth_", "label_")
                    if counterpart in full_df.columns:
                        full_df[col] = full_df[counterpart]
                    else:
                        full_df[col] = "" if dtype == str else pd.NA

        full_df = full_df.astype(mmcif_col_types, errors="ignore")

        df: Dict[str, pd.DataFrame] = {}
        df["ATOM"] = full_df[full_df.group_PDB == "ATOM"].copy()
        df["HETATM"] = full_df[full_df.group_PDB == "HETATM"].copy()
        try:
            df["ANISOU"] = pd.DataFrame(data["atom_site_anisotrop"])
        except KeyError:
            df["ANISOU"] = pd.DataFrame(columns=ANISOU_DF_COLUMNS)
        return df

    # Monkey-patch and mark as patched
    _pm.PandasMmcif._construct_df = _construct_df_patched  # type: ignore[assignment]
    _pm.PandasMmcif._fksfold_patch_applied = True 