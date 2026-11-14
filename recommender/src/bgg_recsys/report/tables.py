"""How tables are shown: number formats in one place, chosen by name.

`show(df, caption, formats={"column": "share"})`; columns without an explicit format get one from their dtype
(integers as counts, floats as metrics).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from IPython.display import display
from pandas.api.types import is_float_dtype, is_integer_dtype

FORMATS = {
    "count": "{:,.0f}",
    "share": "{:.1%}",
    "small share": "{:.3%}",
    "signed share": "{:+.1%}",
    "percent": "{:.0%}",
    "signed percent": "{:+.0%}",
    "metric": "{:.3f}",
    "rmse": "{:.4f}",
    "difference": "{:+.4f}",
    "rating": "{:.2f}",
    "signed rating": "{:+.2f}",
    "year": "{:.0f}",
    "scientific": "{:.1e}",
}


def show(table: pd.DataFrame | pd.Series, caption: str | None = None, formats: Mapping | None = None, default: str | None = None) -> None:
    """Display a table with named number formats; `default` applies to every numeric column not in `formats`."""
    df = table.to_frame() if isinstance(table, pd.Series) else table
    chosen = {}
    for col in df.columns:
        numeric = is_integer_dtype(df[col]) or is_float_dtype(df[col])
        name = (formats or {}).get(col, default if numeric else None)
        if name is None and is_integer_dtype(df[col]):
            name = "count"
        elif name is None and is_float_dtype(df[col]):
            name = "metric"
        if name is not None:
            chosen[col] = FORMATS[name]
    styler = df.style.set_caption(caption) if caption else df.style
    display(styler.format(chosen, na_rep="—"))


def facts(values: Mapping[str, object] | pd.Series, caption: str | None = None, formats: Mapping[str, str] | None = None,
          default: str | None = None) -> None:
    """Display named values as a two-column table, formatting each by name (otherwise `default`, else integers as counts, floats as metrics)."""
    rows = {}
    for key, value in dict(values).items():
        name = (formats or {}).get(key, default)
        if name is None and isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            name = "count"
        elif name is None and isinstance(value, (float, np.floating)):
            name = "metric"
        rows[key] = FORMATS[name].format(value) if name else value
    table = pd.DataFrame({"value": rows})
    display(table.style.set_caption(caption) if caption else table)
