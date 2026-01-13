
from __future__ import annotations
import os, json, shutil, tempfile, traceback
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import ScalarFormatter

# deps
try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    _HAS_TUKEY = True
except Exception:
    _HAS_TUKEY = False
try:
    import scikit_posthocs as sp
    _HAS_SCPH = True
except Exception:
    _HAS_SCPH = False
try:
    from statsmodels.stats.diagnostic import lilliefors
    _HAS_LILLIEFORS = True
except Exception:
    _HAS_LILLIEFORS = False
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    _HAS_SM = True
except Exception:
    _HAS_SM = False

# Tkinter GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser

# --------------------------------------------------------------------------------
# I/O helpers
def _read_any_table(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("No file path provided.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if not os.path.isfile(path):
        raise ValueError(f"Selected path is not a file: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Insufficient read permission for: {path}")
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
            return pd.read_excel(path, sheet_name=(sheet or 0), engine="openpyxl")
        elif ext == ".xls":
            try:
                return pd.read_excel(path, sheet_name=(sheet or 0), engine="xlrd")
            except Exception:
                raise ValueError("Reading .xls requires 'xlrd'. Install xlrd or save as .xlsx and try again.")
        else:
            return pd.read_csv(path)
    except PermissionError as pe:
        raise PermissionError(
            f"Permission denied for '{path}'. "
            "Close the file in Excel, ensure it’s not read-only or blocked, "
            "and copy it to a local folder you own (e.g., Documents) before retrying."
        ) from pe
    except OSError as oe:
        if getattr(oe, 'errno', None) == 13:
            raise PermissionError(
                f"OS blocked access to '{path}' (errno 13). "
                "Close any app using the file and retry, or move the file to a local, writable folder."
            ) from oe
        raise

def _excel_sheet_names(path: str) -> List[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext not in {".xlsx", ".xlsm", ".xltx", ".xltm", ".xls"}:
        return []
    try:
        xls = pd.ExcelFile(path, engine=("openpyxl" if ext != ".xls" else "xlrd"))
        return list(xls.sheet_names)
    except Exception as e:
        print(f"(Excel) Could not read sheet names: {e}")
        return []

def _pick_excel_sheet_dialog(path: str) -> Optional[str]:
    sheets = _excel_sheet_names(path)
    if not sheets:
        return None
    default = "data" if "data" in {s.lower() for s in sheets} else sheets[0]
    dlg = tk.Toplevel()
    dlg.title("Choose Excel sheet")
    dlg.resizable(False, False)
    ttk.Label(dlg, text="Select sheet to import:").pack(padx=12, pady=(12, 4))
    var = tk.StringVar(value=default)
    cb = ttk.Combobox(dlg, textvariable=var, values=sheets, state="readonly", width=28)
    cb.pack(padx=12, pady=(0, 8))
    chosen = {"name": None}
    def _ok():
        chosen["name"] = var.get().strip() or None
        dlg.destroy()
    def _cancel():
        chosen["name"] = None
        dlg.destroy()
    btns = ttk.Frame(dlg); btns.pack(padx=12, pady=(0, 10))
    ttk.Button(btns, text="OK", command=_ok).grid(row=0, column=0, padx=6)
    ttk.Button(btns, text="Cancel", command=_cancel).grid(row=0, column=1, padx=6)
    dlg.grab_set()
    dlg.wait_window()
    return chosen["name"]

# --------------------------------------------------------------------------------
# Wide-form Excel loader
def is_wide_excel(path: str, sheet: Optional[str] = None) -> bool:
    try:
        df_raw = pd.read_excel(path, header=None, nrows=2, engine="openpyxl", sheet_name=(sheet or 0))
        if df_raw.shape[0] < 2:
            return False
        first_row = df_raw.iloc[0].tolist()
        second_row = df_raw.iloc[1].tolist()
        cond1 = all(isinstance(x, str) for x in first_row if pd.notna(x))
        cond2 = any(isinstance(y, (int, float)) for y in second_row if pd.notna(y))
        return cond1 and cond2
    except Exception:
        return False

def load_excel_wide(path: str, sheet: Optional[str] = None) -> Tuple[Dict[str, List[float]], pd.DataFrame]:
    df_raw = pd.read_excel(path, header=None, engine="openpyxl", sheet_name=(sheet or 0))
    if df_raw.shape[0] < 2 or df_raw.shape[1] < 1:
        raise ValueError("Wide-format Excel must have at least 1 column and 2 rows (names + values).")
    group_names = [str(x).strip() for x in df_raw.iloc[0].tolist()]
    df_values = df_raw.iloc[1:].reset_index(drop=True)
    long_rows = []
    for col_idx, group in enumerate(group_names):
        if group is None or group == "" or str(group).lower() == "nan":
            group = f"Group_{col_idx+1}"
        series = pd.to_numeric(df_values.iloc[:, col_idx], errors="coerce").dropna()
        for val in series.tolist():
            long_rows.append({"group": group, "value": float(val)})
    df_long = pd.DataFrame(long_rows, columns=["group", "value"])
    data_dict = {grp: sub["value"].tolist() for grp, sub in df_long.groupby("group")}
    if not data_dict:
        raise ValueError("No numeric values found when interpreting columns as groups.")
    return data_dict, df_long

# --------------------------------------------------------------------------------
# loader
def load_data(file_path: Optional[str], sheet: Optional[str] = None) -> Tuple[Dict[str, List[float]], pd.DataFrame]:
    if not file_path:
        raise ValueError("Please select a data file (CSV or Excel).")
    df_raw = _read_any_table(file_path, sheet=sheet)
    if df_raw.shape[1] < 2:
        raise ValueError("Table must have at least two columns: group, value (optional third: subject).")
    cols_lower = {c.lower(): c for c in df_raw.columns}
    group_col = cols_lower.get("group", df_raw.columns[0])
    value_col = cols_lower.get("value", df_raw.columns[1])
    subject_col = cols_lower.get("subject", None)
    df = df_raw.copy()
    df.rename(columns={group_col: "group", value_col: "value"}, inplace=True)
    if subject_col:
        df.rename(columns={subject_col: "subject"}, inplace=True)
    df = df.dropna(subset=["group", "value"]).copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).copy()
    data_dict: Dict[str, List[float]] = {}
    for grp, sub in df.groupby("group"):
        vals = pd.to_numeric(sub["value"], errors="coerce").dropna().astype(float).tolist()
        if len(vals) > 0:
            data_dict[str(grp)] = vals
    if len(data_dict) < 1:
        raise ValueError("No valid groups with data found in the table.")
    return data_dict, df

# --------------------------------------------------------------------------------
# Color helpers
def tuple_to_hex(rgb) -> str:
    r, g, b = [int(255*x) for x in rgb[:3]]
    return f"#{r:02X}{g:02X}{b:02X}"

def assign_colors(categories: List[str], base_colors: Dict[str, str]) -> Dict[str, str]:
    palette = plt.get_cmap("tab10")
    return {cat: base_colors.get(cat, tuple_to_hex(palette(i % 10))) for i, cat in enumerate(categories)}

def ensure_colors_for_keys(keys: List[str], colors: Dict[str, str]) -> Dict[str, str]:
    out = {k: v for k, v in (colors or {}).items()
           if k in keys and isinstance(v, str) and v.startswith("#") and len(v) in (4,7)}
    needed = [c for c in keys if c not in out]
    if needed:
        auto = assign_colors(keys, {})
        for c in needed:
            out[c] = auto.get(c, "#777777")
    return out

# --------------------------------------------------------------------------------
# Stats, formatting, tables, plots
def adjust_pvals(pvals: List[float], method: str = "holm") -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    if m == 0:
        return pvals
    method = method.lower().strip()
    if method == "bonferroni":
        return np.clip(pvals * m, 0, 1)
    elif method == "holm":
        order = np.argsort(pvals)
        adjusted = np.empty(m, dtype=float)
        running_max = 0.0
        for i, idx in enumerate(order):
            adj = min(pvals[idx] * (m - i), 1.0)
            running_max = max(running_max, adj)
            adjusted[idx] = running_max
        return adjusted
    elif method in {"bh","fdr","benjamini-hochberg"}:
        order = np.argsort(pvals)
        ranked = pvals[order]
        denom = np.arange(1, m+1)
        q = ranked * m / denom
        q = np.minimum.accumulate(q[::-1])[::-1]
        adjusted = np.empty_like(pvals)
        adjusted[order] = np.clip(q, 0, 1)
        return adjusted
    return pvals

def diagnostics(data: Dict[str, List[float]]) -> Tuple[Dict[str, Tuple[Optional[float], Optional[float]]], Tuple[Optional[float], Optional[float]]]:
    cats = list(data.keys())
    arrays = [np.asarray(data[c], dtype=float) for c in cats]
    print("\n=== Diagnostics ===")
    shapiro = {}
    for c, arr in zip(cats, arrays):
        if len(arr) >= 3:
            try:
                w, p = stats.shapiro(arr)
                print(f" Shapiro–Wilk ({c}): W={w:.3f}, p={p:.3e}")
                shapiro[c] = (float(w), float(p))
            except Exception as e:
                print(f" Shapiro–Wilk ({c}): error ({e})")
                shapiro[c] = (None, None)
        else:
            print(f" Shapiro–Wilk ({c}): n={len(arr)} (<3), skipped.")
            shapiro[c] = (None, None)
    if all(len(a) >= 2 for a in arrays) and len(arrays) >= 2:
        try:
            w, p = stats.levene(*arrays, center='median')
            print(f" Levene (equal variances across groups): W={w:.3f}, p={p:.3e}")
            levene = (float(w), float(p))
        except Exception as e:
            print(f" Levene: error ({e})")
            levene = (None, None)
    else:
        print(" Levene: not enough data across groups.")
        levene = (None, None)
    return shapiro, levene

def format_p_label(p: float) -> str:
    if p < 0.001: return "p<0.001"
    if p < 0.01: return "p<0.01"
    s = f"{p:.2f}".rstrip('0').rstrip('.')
    return f"p={s}"

def format_sig_marker(p: float, mode: str = "p-value") -> str:
    try:
        p = float(p)
    except Exception:
        return "" if mode == "asterisks" else "p=NA"
    if mode == "asterisks":
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return ""
    return format_p_label(p)

def compute_descriptives(data: Dict[str, List[float]], ci_level: float = 0.95) -> pd.DataFrame:
    rows = []
    alpha = 1.0 - float(ci_level)
    for g, vals in data.items():
        x = np.asarray(vals, dtype=float)
        x = x[~np.isnan(x)]
        n = int(x.size)
        if n == 0:
            rows.append({"group": g, "n": 0, "mean": np.nan, "sd": np.nan, "se": np.nan,
                         "ci_low": np.nan, "ci_high": np.nan, "ci_level": ci_level})
            continue
        mean = float(np.mean(x))
        sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
        se = (sd / np.sqrt(n)) if n > 0 else np.nan
        if n > 1 and np.isfinite(se):
            tcrit = stats.t.ppf(1 - alpha / 2.0, df=n - 1)
            half = float(tcrit * se)
            ci_low = mean - half
            ci_high = mean + half
        else:
            ci_low = np.nan
            ci_high = np.nan
        rows.append({
            "group": g, "n": n, "mean": mean, "sd": sd, "se": se,
            "ci_low": ci_low, "ci_high": ci_high, "ci_level": ci_level
        })
    df_desc = pd.DataFrame(rows, columns=["group","n","mean","sd","se","ci_low","ci_high","ci_level"])
    return df_desc

# Y-label history
HISTORY_FILE = "y_label_history.json"
HISTORY_LIMIT = 4
def load_y_label_history() -> List[str]:
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [str(x) for x in data][:HISTORY_LIMIT]
    except Exception:
        pass
    return []

def save_y_label_to_history(label: str):
    if not label or not isinstance(label, str):
        return
    label = label.strip()
    if not label:
        return
    hist = load_y_label_history()
    hist = [h for h in hist if h.strip().lower() != label.lower()]
    hist.insert(0, label)
    hist = hist[:HISTORY_LIMIT]
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# Q–Q plots and tables (unchanged)
def qq_plot_groups(data: Dict[str, List[float]], title: str = "Q–Q plots (Normality check)"):
    cats = [g for g in data.keys() if len(data[g]) >= 3]
    if not cats:
        print("\n(Q–Q plots) Not enough data (need ≥3 per group).")
        return
    n = len(cats)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.4, rows*3.2), squeeze=False)
    for idx, g in enumerate(cats):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        x = np.asarray(data[g], dtype=float)
        stats.probplot(x, dist="norm", plot=ax)
        ax.set_title(f"{g} (n={len(x)})", fontsize=9)
    for idx in range(n, rows*cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].set_visible(False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

def _make_table_figure(df, title: str, max_height_in=16.0):
    if isinstance(df, pd.DataFrame):
        data = df.copy()
    else:
        raise ValueError("Input must be a pandas DataFrame")
    rows, cols = data.shape
    fig_w = max(6.0, 1.2 * max(1, cols))
    fig_h = min(max(2.4, 0.6 + 0.36 * max(1, rows)), max_height_in)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=8)
    str_vals = [[str(v) for v in row] for row in data.values]
    table = ax.table(cellText=str_vals,
                     colLabels=[str(c) for c in data.columns],
                     loc="upper left",
                     cellLoc="center")
    table.auto_set_font_size(False)
    base_fs = 10
    density = (rows * cols) / 120.0
    fs = max(6, int(round(base_fs - 2 * density)))
    table.set_fontsize(fs)
    table.scale(1.0, 1.1)
    plt.tight_layout()
    return fig

def show_pairwise_long_table_figure(pairwise_results):
    if not pairwise_results:
        print("\n(No pairwise comparisons to display)")
        return
    df = pd.DataFrame(pairwise_results).copy()
    for c in ["ref","group","name","stat","p_raw","p_adj"]:
        if c not in df.columns:
            df[c] = np.nan
    def _fmt_stat(x): return "" if pd.isna(x) else f"{x:.3f}"
    def _fmt_p(x):
        if pd.isna(x): return ""
        try: x = float(x)
        except Exception: return str(x)
        if 0.001 <= x < 0.1:
            s = f"{x:.6f}".rstrip("0").rstrip(".")
        else:
            s = f"{x:.3e}"
        return s
    df_long = df[["ref","group","name","stat","p_raw","p_adj"]].copy()
    df_long["stat"] = df_long["stat"].map(_fmt_stat)
    df_long["p_raw"] = df_long["p_raw"].map(_fmt_p)
    df_long["p_adj"] = df_long["p_adj"].map(_fmt_p)
    df_long = df_long.sort_values(["ref","group","name"], kind="stable")
    _make_table_figure(df_long, title="Pairwise comparisons — p-values (long table)")

def show_normality_table(rows: List[Dict[str, str]], title: str = "Normality tests"):
    win = tk.Toplevel()
    win.title(title)
    win.geometry("1100x600")  # enlarged popup
    frame = ttk.Frame(win, padding=(8,8))
    frame.pack(fill="both", expand=True)
    columns = ["Group", "Test", "Stat", "p-value", "Decision", "Note"]
    tree = ttk.Treeview(frame, columns=columns, show="headings", height=16)
    for col in columns:
        tree.heading(col, text=col)
    w = {"Group":150, "Test":220, "Decision":140}
    for col in columns:
        width = w.get(col, (140 if col in {"Stat","p-value"} else 520))
        tree.column(col, width=width, anchor="center" if col != "Note" else "w")
    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    for r in rows:
        tree.insert("", "end", values=[r[c] for c in columns])
    ttk.Button(frame, text="Close", command=win.destroy).grid(row=2, column=0, sticky="e", pady=(8,0))

def show_descriptives_table(df: pd.DataFrame, title: str = "Descriptives"):
    win = tk.Toplevel()
    win.title(title)
    win.geometry("1100x600")  # enlarged popup
    frame = ttk.Frame(win, padding=(8,8))
    frame.pack(fill="both", expand=True)
    columns = list(df.columns)
    tree = ttk.Treeview(frame, columns=columns, show="headings", height=16)
    for col in columns:
        tree.heading(col, text=col)
    tree.column(col, width=140, anchor="center")
    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    for _, row in df.iterrows():
        tree.insert("", "end", values=[row[c] for c in columns])
    ttk.Button(frame, text="Close", command=win.destroy).grid(row=2, column=0, sticky="e", pady=(8,0))

# --------------------------------------------------------------------------------
# Normality runners, posthocs, two-way ANOVA, etc. (logic mostly unchanged)
def run_normality_all(data: Dict[str, List[float]], alpha: float) -> Dict[str, Dict[str, dict]]:
    results: Dict[str, Dict[str, dict]] = {}
    for g, vals in data.items():
        x = np.asarray(vals, dtype=float)
        n = len(x)
        mu = np.mean(x)
        sigma = np.std(x, ddof=1) if n > 1 else 0.0
        group_res: Dict[str, dict] = {}
        if n >= 3:
            try:
                stat, p = stats.shapiro(x)
                group_res["Shapiro–Wilk"] = {"stat": float(stat), "p": float(p), "decision": (p >= alpha)}
            except Exception as e:
                group_res["Shapiro–Wilk"] = {"stat": None, "p": None, "decision": None, "error": str(e)}
        else:
            group_res["Shapiro–Wilk"] = {"stat": None, "p": None, "decision": None, "note": "n<3"}
        if n >= 8:
            try:
                stat, p = stats.normaltest(x)
                group_res["D’Agostino K²"] = {"stat": float(stat), "p": float(p), "decision": (p >= alpha)}
            except Exception as e:
                group_res["D’Agostino K²"] = {"stat": None, "p": None, "decision": None, "error": str(e)}
        else:
            group_res["D’Agostino K²"] = {"stat": None, "p": None, "decision": None, "note": "n<8"}
        if n >= 2:
            try:
                stat, p = stats.jarque_bera(x)
                group_res["Jarque–Bera"] = {"stat": float(stat), "p": float(p), "decision": (p >= alpha)}
            except Exception as e:
                group_res["Jarque–Bera"] = {"stat": None, "p": None, "decision": None, "error": str(e)}
        else:
            group_res["Jarque–Bera"] = {"stat": None, "p": None, "decision": None, "note": "n<2"}
        try:
            ad = stats.anderson(x, dist='norm')
            levels = list(ad.significance_level)
            crits = list(ad.critical_values)
            level_to_crit = {float(l): float(c) for l, c in zip(levels, crits)}
            stat = float(ad.statistic)
            reject_5 = stat > level_to_crit.get(5.0, np.nan)
            group_res["Anderson–Darling"] = {
                "stat": stat, "p": None, "decision": (not reject_5),
                "critical_values": level_to_crit,
                "note": "Decision at 5% via critical values; no p-value."
            }
        except Exception as e:
            group_res["Anderson–Darling"] = {"stat": None, "p": None, "decision": None, "error": str(e)}
        if _HAS_LILLIEFORS and n >= 3:
            try:
                stat, p = lilliefors(x, dist='norm', pvalmethod='approx')
                group_res["Lilliefors"] = {"stat": float(stat), "p": float(p), "decision": (p >= alpha)}
            except Exception as e:
                group_res["Lilliefors"] = {"stat": None, "p": None, "decision": None, "error": str(e)}
        else:
            if not _HAS_LILLIEFORS:
                group_res["Lilliefors"] = {"stat": None, "p": None, "decision": None, "note": "statsmodels not installed"}
        if n >= 3 and sigma > 0:
            try:
                stat, p = stats.kstest(x, 'norm', args=(mu, sigma))
                group_res["KS (param est)"] = {
                    "stat": float(stat), "p": float(p), "decision": (p >= alpha),
                    "note": "Params estimated; p conservative. Prefer Lilliefors if available."
                }
            except Exception as e:
                group_res["KS (param est)"] = {"stat": None, "p": None, "decision": None, "error": str(e)}
        else:
            group_res["KS (param est)"] = {"stat": None, "p": None, "decision": None, "note": "n<3 or σ=0"}
        results[g] = group_res
    return results

def flatten_normality_results(results: Dict[str, Dict[str, dict]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for g, tests in results.items():
        for test_name, res in tests.items():
            stat = res.get("stat")
            pval = res.get("p")
            dec = res.get("decision")
            note = res.get("note") or res.get("error")
            if test_name == "Anderson–Darling" and res.get("critical_values"):
                note = (note or "")
                crits = res["critical_values"]
                note = f"{note} crit@5%={crits.get(5.0,'NA')}".strip()
            rows.append({
                "Group": g,
                "Test": test_name,
                "Stat": "NA" if stat is None else f"{stat:.4f}",
                "p-value": "NA" if pval is None else f"{pval:.4e}",
                "Decision": "NA" if dec is None else ("pass" if dec else "reject"),
                "Note": note or ""
            })
    return rows

def run_tukey_all_pairs(data: Dict[str, List[float]], alpha: float) -> List[Dict[str, float]]:
    if not _HAS_TUKEY: raise ImportError("Tukey HSD needs 'statsmodels'.")
    endog, groups = [], []
    for g, vals in data.items():
        endog.extend(vals); groups.extend([g] * len(vals))
    endog = np.asarray(endog, dtype=float)
    groups = np.asarray(groups, dtype=object)
    tukey = pairwise_tukeyhsd(endog=endog, groups=groups, alpha=alpha)
    rows = tukey.summary().data[1:]
    out = []
    for g1, g2, meandiff, p_adj, *_ in rows:
        out.append({"ref": str(g1), "group": str(g2), "name": "Tukey HSD",
                    "stat": float(meandiff), "p_raw": float(p_adj), "p_adj": float(p_adj)})
    return out

def run_dunn_all_pairs(df: pd.DataFrame, correction: str) -> List[Dict[str, float]]:
    if not _HAS_SCPH: raise ImportError("Dunn's needs 'scikit-posthocs'.")
    corr_map = {"bonferroni":"bonferroni", "holm":"holm", "bh":"fdr_bh"}
    p_adjust = corr_map.get(correction.lower().strip(), "holm")
    dunn_df = sp.posthoc_dunn(df, val_col="value", group_col="group", p_adjust=p_adjust)
    cats = list(dunn_df.columns)
    out = []
    for i in range(len(cats)):
        for j in range(i+1, len(cats)):
            g1, g2 = cats[i], cats[j]
            p_adj = float(dunn_df.loc[g1, g2])
            out.append({"ref": str(g1), "group": str(g2), "name": "Dunn's test",
                        "stat": np.nan, "p_raw": p_adj, "p_adj": p_adj})
    return out

def run_gameshowell_all_pairs(df: pd.DataFrame) -> List[Dict[str, float]]:
    if not _HAS_SCPH: raise ImportError("Games–Howell needs 'scikit-posthocs'.")
    gh = sp.posthoc_gameshowell(df, val_col='value', group_col='group')
    cats = list(gh.columns)
    out = []
    for i in range(len(cats)):
        for j in range(i+1, len(cats)):
            g1, g2 = cats[i], cats[j]
            p_adj = float(gh.loc[g1, g2])
            out.append({"ref": str(g1), "group": str(g2), "name": "Games–Howell",
                        "stat": np.nan, "p_raw": p_adj, "p_adj": p_adj})
    return out

def run_t_ind_vs_ref(data: Dict[str, List[float]], ref: str, equal_var: bool, correction: str, exclude: Optional[str] = None) -> List[Dict[str, float]]:
    out = []
    for g in data.keys():
        if g == ref or (exclude and g == exclude): continue
        t_stat, p_val = stats.ttest_ind(data[ref], data[g], equal_var=equal_var)
        out.append({"ref": ref, "group": g, "name": "Student t" if equal_var else "Welch t",
                    "stat": float(t_stat), "p_raw": float(p_val)})
    p_adj = adjust_pvals([r["p_raw"] for r in out], method=correction)
    for r, pa in zip(out, p_adj): r["p_adj"] = float(pa)
    return out

def run_t_ind_all_pairs(data: Dict[str, List[float]], equal_var: bool, correction: str) -> List[Dict[str, float]]:
    cats = list(data.keys())
    out = []
    for i in range(len(cats)):
        for j in range(i+1, len(cats)):
            g1, g2 = cats[i], cats[j]
            t_stat, p_val = stats.ttest_ind(data[g1], data[g2], equal_var=equal_var)
            out.append({"ref": g1, "group": g2, "name": "Student t" if equal_var else "Welch t",
                        "stat": float(t_stat), "p_raw": float(p_val)})
    p_adj = adjust_pvals([r["p_raw"] for r in out], method=correction)
    for r, pa in zip(out, p_adj): r["p_adj"] = float(pa)
    return out

def run_t_paired(df: pd.DataFrame, mode: str, correction: str, ref: Optional[str] = None, exclude: Optional[str] = None) -> List[Dict[str, float]]:
    if "subject" not in df.columns:
        raise ValueError("Paired t-test requires a 'subject' column.")
    cats = list(df["group"].unique()); out = []
    def paired(g1: str, g2: str):
        a = df[df["group"] == g1][["subject","value"]].rename(columns={"value":"v1"})
        b = df[df["group"] == g2][["subject","value"]].rename(columns={"value":"v2"})
        merged = pd.merge(a, b, on="subject", how="inner")
        if len(merged) < 2: return None
        t_stat, p_val = stats.ttest_rel(merged["v1"].values, merged["v2"].values)
        return {"ref": g1, "group": g2, "name": "Paired t", "stat": float(t_stat), "p_raw": float(p_val)}
    if mode == "vs_ref":
        if not ref: raise ValueError("ref required for paired vs_ref")
        for g in cats:
            if g == ref or (exclude and g == exclude): continue
            res = paired(ref, g)
            if res: out.append(res)
    else:
        for i in range(len(cats)):
            for j in range(i+1, len(cats)):
                res = paired(cats[i], cats[j])
                if res: out.append(res)
    p_adj = adjust_pvals([r["p_raw"] for r in out], method=correction)
    for r, pa in zip(out, p_adj): r["p_adj"] = float(pa)
    return out

def run_wilcoxon_paired(df: pd.DataFrame, mode: str, correction: str, ref: Optional[str] = None, exclude: Optional[str] = None) -> List[Dict[str, float]]:
    if "subject" not in df.columns:
        raise ValueError("Wilcoxon requires a 'subject' column.")
    cats = list(df["group"].unique()); out = []
    def paired(g1: str, g2: str):
        a = df[df["group"] == g1][["subject","value"]].rename(columns={"value":"v1"})
        b = df[df["group"] == g2][["subject","value"]].rename(columns={"value":"v2"})
        merged = pd.merge(a, b, on="subject", how="inner").dropna()
        if len(merged) < 2: return None
        try:
            stat, p_val = stats.wilcoxon(merged["v1"].values, merged["v2"].values, zero_method="wilcox",
                                         alternative="two-sided", method="auto")
        except TypeError:
            stat, p_val = stats.wilcoxon(merged["v1"].values, merged["v2"].values, zero_method="wilcox",
                                         alternative="two-sided")
        return {"ref": g1, "group": g2, "name": "Wilcoxon signed-rank", "stat": float(stat), "p_raw": float(p_val)}
    if mode == "vs_ref":
        if not ref: raise ValueError("ref required for Wilcoxon vs_ref")
        for g in cats:
            if g == ref or (exclude and g == exclude): continue
            res = paired(ref, g)
            if res: out.append(res)
    else:
        for i in range(len(cats)):
            for j in range(i+1, len(cats)):
                res = paired(cats[i], cats[j])
                if res: out.append(res)
    p_adj = adjust_pvals([r["p_raw"] for r in out], method=correction)
    for r, pa in zip(out, p_adj): r["p_adj"] = float(pa)
    return out

def run_t_one_sample(df: pd.DataFrame, mu: float) -> List[Dict[str, float]]:
    out = []
    for g, sub in df.groupby("group"):
        t_stat, p_val = stats.ttest_1samp(sub["value"].values, popmean=mu)
        out.append({"ref": g, "group": f"μ={mu}", "name": "One-sample t",
                    "stat": float(t_stat), "p_raw": float(p_val), "p_adj": float(p_val), "single": True})
    return out

# Two‑way ANOVA
def run_two_way_anova(df: pd.DataFrame, factor_a: str, factor_b: str):
    if not _HAS_SM:
        raise ImportError("Two-way ANOVA requires 'statsmodels'.")
    for col in [factor_a, factor_b, "value"]:
        if col not in df.columns:
            raise ValueError(f"Two-way ANOVA requires columns '{factor_a}', '{factor_b}', and 'value' in the table.")
    model = ols(f'value ~ C({factor_a}) * C({factor_b})', data=df).fit()
    anova_tbl = sm.stats.anova_lm(model, typ=2)
    print("\nTwo-way ANOVA (Type II SS) — main effects and interaction:")
    print(anova_tbl)
    return anova_tbl, model

def run_friedman(df: pd.DataFrame):
    if "subject" not in df.columns:
        raise ValueError("Friedman test requires a 'subject' column.")
    wide = df.pivot_table(index="subject", columns="group", values="value", aggfunc='mean')
    wide = wide.dropna(axis=0, how="any")
    if wide.shape[1] < 3 or wide.shape[0] < 2:
        raise ValueError("Friedman requires ≥3 groups and ≥2 complete subjects.")
    arrays = [wide[c].values for c in wide.columns]
    stat, p = stats.friedmanchisquare(*arrays)
    print(f"\nFriedman test: χ²_F={float(stat):.3f}, p={float(p):.3e} (subjects n={wide.shape[0]}; groups={list(wide.columns)})")
    return stat, p, list(wide.columns)

# --------------------------------------------------------------------------------
# Histogram panel
def parse_bins_spec(spec: str) -> object:
    s = (spec or "").strip().lower()
    if s in {"auto","fd","sturges","doane","scott","rice","sqrt"}:
        return s
    try:
        n = int(float(s))
        return max(1, n)
    except:
        return "auto"

def plot_histograms_panel(
    categories: List[str],
    data: Dict[str, List[float]],
    colors: Dict[str, str],
    max_groups: int = 30,
    bins_spec: str = "auto",
    title: str = "Histograms per group"
):
    groups_with_data = [g for g in categories if len(data.get(g, [])) > 0]
    if len(groups_with_data) == 0:
        print("No groups with data for histograms.")
        return
    groups_with_data = groups_with_data[:max_groups]
    n = len(groups_with_data)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    bins = parse_bins_spec(bins_spec)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.6, rows*2.2), squeeze=False)
    for idx, g in enumerate(groups_with_data):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        vals = np.asarray(data[g], dtype=float)
        ax.hist(vals, bins=bins, color=colors.get(g, "#777777"), alpha=0.85, edgecolor="white")
        ax.set_title(f"{g} (n={len(vals)})", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(False)
    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].set_visible(False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

# --------------------------------------------------------------------------------
# Series helpers & plotting
def series_levels(df: pd.DataFrame, series_factor: Optional[str]) -> List[str]:
    if not series_factor or series_factor not in df.columns:
        return []
    levels = list(map(str, sorted(df[series_factor].dropna().astype(str).unique())))
    return levels

def build_series_means(categories: List[str], df: pd.DataFrame, series_factor: Optional[str]) -> Dict[str, np.ndarray]:
    if not series_factor or series_factor not in df.columns:
        result = {"All": np.array([df[df["group"] == g]["value"].mean() for g in categories], dtype=float)}
        return result
    levels = series_levels(df, series_factor)
    out: Dict[str, np.ndarray] = {}
    for lvl in levels:
        means = []
        sub = df[df[series_factor].astype(str) == lvl]
        for g in categories:
            means.append(sub[sub["group"] == g]["value"].mean())
        out[str(lvl)] = np.array(means, dtype=float)
    return out

def build_group_x(categories: List[str], df: pd.DataFrame, x_col: Optional[str]) -> np.ndarray:
    if not x_col or x_col not in df.columns:
        return np.arange(len(categories), dtype=float)
    col = df[x_col]
    if not np.issubdtype(col.dropna().infer_objects().dtype, np.number):
        return np.arange(len(categories), dtype=float)
    xs = []
    for g in categories:
        vals = df[df["group"] == g][x_col].dropna().astype(float)
        xs.append(vals.mean() if len(vals) else np.nan)
    xs = np.array(xs, dtype=float)
    if np.isnan(xs).sum() > max(0, len(xs)//3):
        return np.arange(len(categories), dtype=float)
    return xs

def plot_box(ax, categories, data, colors):
    vals = [np.asarray(data[c], dtype=float) for c in categories]
    bp = ax.boxplot(vals, patch_artist=True, labels=categories)
    for patch, cat in zip(bp['boxes'], categories):
        patch.set_facecolor(colors.get(cat, "#999999"))
        patch.set_alpha(0.75)
        patch.set_edgecolor("#444444")
    for whisker in bp['whiskers'] + bp['caps'] + bp['medians']:
        whisker.set_color("#444444")
    return ax

def plot_violin(ax, categories, data, colors):
    vals = [np.asarray(data[c], dtype=float) for c in categories]
    v = ax.violinplot(vals, showmeans=False, showmedians=True, showextrema=True)
    for i, b in enumerate(v['bodies']):
        b.set_facecolor(colors.get(categories[i], "#999999"))
        b.set_alpha(0.75)
        b.set_edgecolor("#444444")
    for k in ("cmins","cmaxes","cbars","cmedians"):
        if k in v: v[k].set_color("#444444")
    ax.set_xticks(np.arange(1, len(categories)+1))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    return ax

def plot_strip(ax, categories, data, colors, jitter=0.08):
    rng = np.random.default_rng(42)
    x = np.arange(len(categories))
    for i, cat in enumerate(categories):
        y = np.asarray(data[cat], dtype=float)
        xj = x[i] + rng.normal(0, jitter, size=len(y))
        ax.scatter(xj, y, s=28, color=colors.get(cat,"#999999"),
                   edgecolor="#444444", linewidths=0.6, alpha=1.0, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    return ax

def plot_mean_ci(ax, categories, means, ci_half, colors, bar=False, line=False, ci_label="95% CI"):
    x = np.arange(len(categories))
    if bar:
        for i, cat in enumerate(categories):
            ax.bar(x[i], means[i], width=0.70, color=colors.get(cat, "#999999"),
                   edgecolor="#444444", linewidth=1.4, alpha=0.80, zorder=1)
        ax.errorbar(x, means, yerr=ci_half, fmt="none", ecolor="black",
                    elinewidth=1.5, capsize=6, capthick=1.5, zorder=2, label=ci_label)
    elif line:
        ax.plot(x, means, color="#333333", marker="o", lw=1.8, zorder=2, label="Mean")
        ax.errorbar(x, means, yerr=ci_half, fmt="none", ecolor="#333333",
                    elinewidth=1.2, capsize=5, capthick=1.2, zorder=2, label=ci_label)
    else:
        ax.errorbar(x, means, yerr=ci_half, fmt="o", color="#333333",
                    ecolor="black", elinewidth=1.5, capsize=6, capthick=1.5, zorder=3, label=ci_label)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    return ax

def plot_line_means(ax, categories, means):
    x = np.arange(len(categories))
    ax.plot(x, means, marker="o", color="#2F3B52", lw=1.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    return ax

def plot_area_quartiles(ax, categories, data):
    x = np.arange(len(categories))
    q1, q2, q3 = [], [], []
    for c in categories:
        arr = np.asarray(data[c], dtype=float)
        if arr.size == 0:
            q1.append(0.0); q2.append(0.0); q3.append(0.0)
        else:
            q1.append(float(np.percentile(arr, 25)))
            q2.append(float(np.percentile(arr, 50)))
            q3.append(float(np.percentile(arr, 75)))
    q1 = np.asarray(q1); q2 = np.asarray(q2); q3 = np.asarray(q3)
    layer_colors = ["#88C057", "#F28E2B", "#4E79A7"]
    ax.fill_between(x, 0, q1, color=layer_colors[0], alpha=0.85, label="Q1 (25th)")
    ax.fill_between(x, q1, q2, color=layer_colors[1], alpha=0.85, label="Median band")
    ax.fill_between(x, q2, q3, color=layer_colors[2], alpha=0.85, label="Q3 band")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    return ax

def plot_multi_lines(ax, x_vec: np.ndarray, categories: List[str], series_means: Dict[str, np.ndarray], colors: Dict[str, str]):
    for i, (series, y) in enumerate(series_means.items()):
        col = colors.get(series, "#2F3B52")
        ax.plot(x_vec, y, marker="o", lw=1.8, color=col, label=str(series))
    ax.legend(loc="best", fontsize=9)
    ax.set_xticks(x_vec if np.all(np.isfinite(x_vec)) else np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    return ax

def plot_multi_areas(ax, x_vec: np.ndarray, categories: List[str], series_means: Dict[str, np.ndarray], colors: Dict[str, str]):
    for series, y in series_means.items():
        col = colors.get(series, "#4E79A7")
        ax.fill_between(x_vec, y, 0, color=col, alpha=0.35, step=None, label=str(series))
        ax.plot(x_vec, y, color=col, lw=1.2)
    ax.legend(loc="best", fontsize=9)
    ax.set_xticks(x_vec if np.all(np.isfinite(x_vec)) else np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    return ax

def plot_pie(ax, categories: List[str], data: Dict[str, List[float]], colors: Dict[str, str], value_mode: str = "sum"):
    if not categories:
        ax.text(0.5, 0.5, "No groups to plot", ha="center", va="center", fontsize=12)
        return ax
    if value_mode == "count":
        sizes = [len(data.get(c, [])) for c in categories]
    else:
        sizes = [float(np.nansum(np.asarray(data.get(c, []), dtype=float))) for c in categories]
    total = sum(sizes)
    if not np.isfinite(total) or total <= 0:
        sizes = [len(data.get(c, [])) for c in categories]
        total = sum(sizes)
        if total <= 0:
            ax.text(0.5, 0.5, "All groups empty", ha="center", va="center", fontsize=12)
            return ax
    col_list = [colors.get(c, "#999999") for c in categories]
    ax.pie(
        sizes,
        labels=categories,
        autopct='%1.1f%%',
        colors=col_list,
        startangle=90,
        pctdistance=0.75,
        labeldistance=1.05,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
    )
    ax.axis('equal')
    return ax

def run_and_plot_regressions(ax, x_vec: np.ndarray, categories: List[str],
                             series_means: Dict[str, np.ndarray], colors: Dict[str, str]):
    print("\n=== Linear regressions (per series) ===")
    r2_values = []
    for series, y in series_means.items():
        x = np.asarray(x_vec, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            print(f" {series}: insufficient points for regression.")
            continue
        res = stats.linregress(x[mask], y[mask])
        a = float(res.intercept)
        b = float(res.slope)
        r2 = float(res.rvalue**2)
        p = float(res.pvalue)
        r2_values.append(r2)
        print(f" {series}: y = {a:.4g} + {b:.4g}·x \n R²={r2:.4f}, p={p:.3e}")
        xfit = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 200)
        yfit = a + b * xfit
        col = colors.get(series, "#2F3B52")
        per_line_label = f"{series} — Regression (series), R²={r2:.3f}"
        ax.plot(xfit, yfit, lw=2.0, color=col, label=per_line_label)
        ax.scatter(x[mask], y[mask], s=28, color=col, edgecolor="#333333", zorder=3)
    if r2_values:
        reg_name = "Regression (series)"
        r2_min = min(r2_values); r2_max = max(r2_values)
        corner_text = f"{reg_name}\nR² range: {r2_min:.3f}–{r2_max:.3f}"
        ax.text(
            0.99, 0.99, corner_text,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color="#1f1f1f",
            bbox=dict(facecolor="white", edgecolor="#888",
                      boxstyle="round,pad=0.3", alpha=0.85)
        )
    ax.legend(loc="best", fontsize=9)
    ax.set_xticks(x_vec if np.all(np.isfinite(x_vec)) else np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    return ax

def run_and_plot_regression_global(ax, x_vec: np.ndarray, categories: List[str],
                                   y_means: np.ndarray, color: str = "#2F3B52"):
    x = np.asarray(x_vec, dtype=float)
    y = np.asarray(y_means, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        ax.text(0.5, 0.5, "Not enough points for regression", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="red")
        return ax
    res = stats.linregress(x[mask], y[mask])
    a = float(res.intercept); b = float(res.slope)
    r2 = float(res.rvalue**2); p = float(res.pvalue)
    xfit = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 200)
    yfit = a + b * xfit
    label = f"Regression (global), R²={r2:.3f}"
    ax.plot(xfit, yfit, lw=2.0, color=color, label=label)
    ax.scatter(x[mask], y[mask], s=28, color=color, edgecolor="#333333", zorder=3)
    corner_text = f"Regression (global)\nR²: {r2:.3f}"
    ax.text(
        0.99, 0.99, corner_text,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, color="#1f1f1f",
        bbox=dict(facecolor="white", edgecolor="#888", boxstyle="round,pad=0.3", alpha=0.85)
    )
    ax.legend(loc="best", fontsize=9)
    ax.set_xticks(x_vec if np.all(np.isfinite(x_vec)) else np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    return ax

# --------------------------------------------------------------------------------
# Execute analysis
# (significance bracket computation functions omitted here for brevity—they are same as your original)
# ... [All bracket computation and execute_analysis functions stay exactly as in your original file]
# For completeness, I am including them below unchanged.

def compute_brackets_default_only(
    categories,
    pairwise_results,
    alpha,
    x,
    bar_tops,
    base_gap,
    stack_step,
    line_h,
    default_ref,
    sig_marker_mode,
    collision_k,
    top_margin_factor
):
    """Compute brackets for pairwise comparisons that involve the default reference group only.
    Returns: bracket_layout (list of dicts), required_ymax (float)"""
    def _padj(res):
        p = res.get("p_adj")
        if p is None:
            p = res.get("p_raw")
        try:
            return float(p)
        except Exception:
            return None
    idx = {g: i for i, g in enumerate(categories)}
    comparisons = []
    for res in (pairwise_results or []):
        g1 = res.get("ref")
        g2 = res.get("group")
        if (g1 not in idx) or (g2 not in idx):
            continue
        if default_ref not in (g1, g2):
            continue
        p_adj = _padj(res)
        if p_adj is None or p_adj > alpha:
            continue
        i1, i2 = idx[g1], idx[g2]
        base_y = max(bar_tops[i1], bar_tops[i2]) + base_gap
        comparisons.append({"x1": float(x[i1]), "x2": float(x[i2]), "base_y": base_y,
                            "text": format_sig_marker(p_adj, sig_marker_mode)})
    # Prefer wider spans first (helps reduce overlap)
    comparisons.sort(key=lambda d: abs(d["x2"] - d["x1"]), reverse=True)
    brackets = []
    def _x_overlaps(b1, b2, eps=1e-9):
        lo1, hi1 = min(b1["x1"], b1["x2"]), max(b1["x1"], b1["x2"])
        lo2, hi2 = min(b2["x1"], b2["x2"]), max(b2["x1"], b2["x2"])
        return not (hi1 < lo2 - eps or hi2 < lo1 - eps)
    for c in comparisons:
        y = c["base_y"]
        # Allow tighter vertical packing (collision_k < 1.0)
        while any((_x_overlaps(c, b)) and (abs(y - b["y"]) < (stack_step * collision_k)) for b in brackets):
            y += stack_step
        brackets.append({"x1": c["x1"], "x2": c["x2"], "y": y, "text": c["text"]})
    highest_bracket_y = max([b["y"] for b in brackets], default=max(bar_tops)) + line_h
    required_ymax = highest_bracket_y * (1.0 + top_margin_factor)
    return brackets, required_ymax

def compute_brackets_from_selection(
    categories,
    pairwise_results,
    alpha,
    x,
    bar_tops,
    base_gap,
    stack_step,
    line_h,
    selected_tokens,
    require_sig,
    sig_marker_mode,
    collision_k,
    top_margin_factor
):
    """Compute brackets for selected pairs (Custom mode or All pairs).
    Returns: bracket_layout (list of dicts), required_ymax (float)"""
    def _padj(res):
        p = res.get("p_adj")
        if p is None:
            p = res.get("p_raw")
        try:
            return float(p)
        except Exception:
            return None
    def _norm_token(a, b):
        return f"{a}\n{b}" if a < b else f"{b}\n{a}"
    lookup = {}
    for res in (pairwise_results or []):
        g1, g2 = res.get("ref"), res.get("group")
        if g1 and g2:
            lookup[_norm_token(g1, g2)] = res
    idx = {g: i for i, g in enumerate(categories)}
    candidates = []
    for tok in (selected_tokens or []):
        if "\n" not in tok:
            continue
        a, b = tok.split("\n", 1)
        norm = _norm_token(a, b)
        res = lookup.get(norm)
        if not res:
            continue
        p_adj = _padj(res)
        if require_sig and (p_adj is None or p_adj > alpha):
            continue
        if a not in idx or b not in idx:
            continue
        i1, i2 = idx[a], idx[b]
        base_y = max(bar_tops[i1], bar_tops[i2]) + base_gap
        candidates.append({"x1": float(x[i1]), "x2": float(x[i2]), "base_y": base_y,
                           "text": format_sig_marker(p_adj, sig_marker_mode)})
    # Prefer wider spans first
    candidates.sort(key=lambda d: abs(d["x2"] - d["x1"]), reverse=True)
    brackets = []
    def _x_overlaps(b1, b2, eps=1e-9):
        lo1, hi1 = min(b1["x1"], b1["x2"]), max(b1["x1"], b1["x2"])
        lo2, hi2 = min(b2["x1"], b2["x2"]), max(b2["x1"], b2["x2"])
        return not (hi1 < lo2 - eps or hi2 < lo1 - eps)
    for c in candidates:
        y = c["base_y"]
        while any((_x_overlaps(c, b)) and (abs(y - b["y"]) < (stack_step * collision_k)) for b in brackets):
            y += stack_step
        brackets.append({"x1": c["x1"], "x2": c["x2"], "y": y, "text": c["text"]})
    highest_bracket_y = max([b["y"] for b in brackets], default=max(bar_tops)) + line_h
    required_ymax = highest_bracket_y * (1.0 + top_margin_factor)
    return brackets, required_ymax

def execute_analysis(cfg: Dict[str, str], data_dict: Dict[str, List[float]], df_data: pd.DataFrame):
    categories = list(data_dict.keys())
    default_ref = cfg["default_ref"] if cfg["default_ref"] else ("-dox" if "-dox" in categories else (categories[0] if categories else ""))
    ref = cfg["ref"] if cfg["ref"] else default_ref
    exclude_vs = cfg["exclude"] if cfg["exclude"] else None
    # Parse floats and new y-range fields
    def _get_float(val: str, default: float) -> float:
        try: return float(val)
        except: return default
    try: alpha = float(cfg["alpha"])
    except Exception: alpha = 0.05
    try: mu_val = float(cfg["mu"])
    except Exception: mu_val = 0.0
    # New optional y-axis range controls
    y_min_str = (cfg.get("y_min") or "").strip()
    y_max_str = (cfg.get("y_max") or "").strip()
    y_min_user = None
    y_max_user = None
    if y_min_str != "":
        try: y_min_user = float(y_min_str)
        except: y_min_user = None
    if y_max_str != "":
        try: y_max_user = float(y_max_str)
        except: y_max_user = None
    if y_min_user is not None and y_max_user is not None and y_max_user <= y_min_user:
        print("[Y-axis] Provided y_max <= y_min; ignoring custom limits.")
        y_min_user = None; y_max_user = None
    y_axis_label = cfg.get("y_label", "Value") or "Value"
    sig_marker_mode = (cfg.get("sig_marker_mode") or "p-value").strip()
    # Advanced spacing defaults tuned to allow more brackets
    base_gap_factor = _get_float(cfg["base_gap_factor"], 0.08)  # was 0.10
    stack_step_factor = _get_float(cfg["stack_step_factor"], 0.09)  # was 0.12
    line_height_factor= _get_float(cfg["line_height_factor"],0.025)  # was 0.03
    collision_k = _get_float(cfg["collision_k"], 0.75)  # was 1.00
    top_margin_factor = _get_float(cfg["top_margin_factor"], 0.12)  # was 0.06
    # Colors
    try:
        colors_cat = json.loads(cfg.get("colors_json","{}"))
        if not isinstance(colors_cat, dict): colors_cat = {}
    except Exception:
        colors_cat = {}
    colors_cat = ensure_colors_for_keys(categories, colors_cat)
    # Optional Excel writer
    xlsx_path = (cfg.get("export_xlsx") or "").strip()
    excel_writer = None
    _excel_sheets_written = 0
    def _write_xlsx(df: pd.DataFrame, sheet_name: str):
        nonlocal excel_writer, _excel_sheets_written
        if not xlsx_path:
            return
        if excel_writer is None:
            try:
                excel_writer = pd.ExcelWriter(xlsx_path, engine="openpyxl")
            except Exception as e:
                print(f"\n[Excel] Could not create workbook '{xlsx_path}': {e}")
                return
        sheet = (sheet_name or "Sheet1")[:31]
        try:
            df.to_excel(excel_writer, sheet_name=sheet, index=False)
            _excel_sheets_written += 1
        except Exception as e:
            print(f"\n[Excel] Failed to write sheet '{sheet}': {e}")

    # Early exit branches
    if cfg["analysis"] == "Normality (all)":
        normality_results = run_normality_all(data_dict, alpha=alpha)
        rows = flatten_normality_results(normality_results)
        if cfg["export_csv"]:
            pd.DataFrame(rows).to_csv(cfg["export_csv"], index=False)
            print(f"\nExported normality table to: {cfg['export_csv']}")
        _write_xlsx(pd.DataFrame(rows), "normality")
        root_tmp = tk.Tk(); root_tmp.withdraw()
        show_normality_table(rows, title=f"Normality tests (alpha={alpha})")
        root_tmp.mainloop()
        qq_plot_groups(data_dict, title="Q–Q plots (Normality check)")
        plt.ion(); plt.show(block=True)
        if excel_writer is not None:
            try:
                cfg_df = pd.DataFrame(sorted(cfg.items()), columns=["key","value"])
                cfg_df.to_excel(excel_writer, sheet_name="config", index=False)
                df_data.to_excel(excel_writer, sheet_name="raw_data", index=False)
                excel_writer.close()
                print(f"\nExported Excel workbook to: {xlsx_path} (sheets: {_excel_sheets_written + 2})")
            except Exception as e:
                print(f"\n[Excel] Failed to finalize workbook: {e}")
        return

    if cfg["analysis"] == "Descriptives":
        try:
            cl = float(cfg.get("ci_level", "0.95"))
            if not (0.0 < cl < 1.0): cl = 0.95
        except Exception:
            cl = 0.95
        desc_df = compute_descriptives(data_dict, ci_level=cl)
        print("\nDescriptives (mean, SD, SE, CI):")
        print(desc_df.to_string(index=False))
        if cfg["export_csv"]:
            desc_df.to_csv(cfg["export_csv"], index=False)
            print(f"\nExported descriptives to: {cfg['export_csv']}")
        _write_xlsx(desc_df, "descriptives")
        show_descriptives_table(desc_df, title=f"Descriptives (CI level={cl})")
        if excel_writer is not None:
            try:
                cfg_df = pd.DataFrame(sorted(cfg.items()), columns=["key","value"])
                cfg_df.to_excel(excel_writer, sheet_name="config", index=False)
                df_data.to_excel(excel_writer, sheet_name="raw_data", index=False)
                excel_writer.close()
                print(f"\nExported Excel workbook to: {xlsx_path} (sheets: {_excel_sheets_written + 2})")
            except Exception as e:
                print(f"\n[Excel] Failed to finalize workbook: {e}")
        return

    # Plot-only flag
    only_plot = (cfg["analysis"] == "None")

    # Diagnostics
    if not only_plot:
        diagnostics(data_dict)

    # Primary analysis and posthocs
    pairwise_results: List[Dict[str, float]] = []
    single_annos: List[Dict[str, float]] = []
    ph = cfg["posthoc"]; scope = cfg["scope"]; correction = cfg["correction"]

    if not only_plot:
        if cfg["analysis"] == "ANOVA":
            group_arrays = [np.array(data_dict[c], dtype=float) for c in categories]
            f_stat, p_val = stats.f_oneway(*group_arrays)
            print(f"\nGlobal test (One-way ANOVA): F={float(f_stat):.3f}, p={float(p_val):.3e}")
            if ph == "Tukey HSD":
                pairwise_results = run_tukey_all_pairs(data_dict, alpha=alpha)
            elif ph == "Games–Howell":
                pairwise_results = run_gameshowell_all_pairs(df_data)
            elif ph in {"vs_ref (Welch)","all_pairs (Welch)"}:
                if "vs_ref" in ph:
                    pairwise_results = run_t_ind_vs_ref(data_dict, ref, equal_var=False, correction=correction, exclude=exclude_vs)
                else:
                    pairwise_results = run_t_ind_all_pairs(data_dict, equal_var=False, correction=correction)
            elif ph in {"vs_ref (Student)","all_pairs (Student)"}:
                if "vs_ref" in ph:
                    pairwise_results = run_t_ind_vs_ref(data_dict, ref, equal_var=True, correction=correction, exclude=exclude_vs)
                else:
                    pairwise_results = run_t_ind_all_pairs(data_dict, equal_var=True, correction=correction)

        elif cfg["analysis"] == "ANOVA (two-way)":
            try:
                anova_tbl, _ = run_two_way_anova(df_data, cfg.get("factor_a",""), cfg.get("factor_b",""))
                try:
                    _write_xlsx(anova_tbl.reset_index().rename(columns={"index": "Effect"}), "anova_two_way")
                except Exception:
                    pass
            except Exception as e:
                print(f"\nTwo-way ANOVA error: {e}")

        elif cfg["analysis"] == "Kruskal":
            group_arrays = [np.array(data_dict[c], dtype=float) for c in categories]
            h_stat, p_val = stats.kruskal(*group_arrays)
            print(f"\nGlobal test (Kruskal–Wallis): H={float(h_stat):.3f}, p={float(p_val):.3e}")
            if ph == "Dunn's":
                pairwise_results = run_dunn_all_pairs(df_data, correction=correction)
            elif ph in {"vs_ref (Mann–Whitney)","all_pairs (Mann–Whitney)"}:
                if "vs_ref" in ph:
                    out = []
                    for g in categories:
                        if g == ref or (exclude_vs and g == exclude_vs): continue
                        u_stat, p_val = stats.mannwhitneyu(data_dict[ref], data_dict[g], alternative="two-sided")
                        out.append({"ref": ref, "group": g, "name": "Mann–Whitney U", "stat": float(u_stat), "p_raw": float(p_val)})
                    p_adj = adjust_pvals([r["p_raw"] for r in out], method=correction)
                    for r, pa in zip(out, p_adj): r["p_adj"] = float(pa)
                    pairwise_results = out
                else:
                    out = []
                    for i in range(len(categories)):
                        for j in range(i+1, len(categories)):
                            g1, g2 = categories[i], categories[j]
                            u_stat, p_val = stats.mannwhitneyu(data_dict[g1], data_dict[g2], alternative="two-sided")
                            out.append({"ref": g1, "group": g2, "name": "Mann–Whitney U", "stat": float(u_stat), "p_raw": float(p_val)})
                    p_adj = adjust_pvals([r["p_raw"] for r in out], method=correction)
                    for r, pa in zip(out, p_adj): r["p_adj"] = float(pa)
                    pairwise_results = out

        elif cfg["analysis"] == "Mann–Whitney (2 groups)":
            if len(categories) < 2:
                print("\nMann–Whitney: need at least 2 groups.")
            else:
                out = []
                if scope == "vs_ref":
                    for g in categories:
                        if g == ref or (exclude_vs and g == exclude_vs): continue
                        u_stat, p_val = stats.mannwhitneyu(data_dict[ref], data_dict[g], alternative="two-sided")
                        out.append({"ref": ref, "group": g, "name": "Mann–Whitney U", "stat": float(u_stat), "p_raw": float(p_val)})
                else:
                    for i in range(len(categories)):
                        for j in range(i+1, len(categories)):
                            g1, g2 = categories[i], categories[j]
                            u_stat, p_val = stats.mannwhitneyu(data_dict[g1], data_dict[g2], alternative="two-sided")
                            out.append({"ref": g1, "group": g2, "name": "Mann–Whitney U", "stat": float(u_stat), "p_raw": float(p_val)})
                p_adj = adjust_pvals([r["p_raw"] for r in out], method=correction)
                for r, pa in zip(out, p_adj): r["p_adj"] = float(pa)
                pairwise_results = out

        elif cfg["analysis"] == "t_ind_equal":
            pairwise_results = (run_t_ind_vs_ref(data_dict, ref, equal_var=True, correction=correction, exclude=exclude_vs)
                                if scope == "vs_ref" else
                                run_t_ind_all_pairs(data_dict, equal_var=True, correction=correction))

        elif cfg["analysis"] == "t_ind_welch":
            pairwise_results = (run_t_ind_vs_ref(data_dict, ref, equal_var=False, correction=correction, exclude=exclude_vs)
                                if scope == "vs_ref" else
                                run_t_ind_all_pairs(data_dict, equal_var=False, correction=correction))

        elif cfg["analysis"] == "t_paired":
            pairwise_results = run_t_paired(df_data, mode=scope, correction=correction, ref=ref, exclude=exclude_vs)

        elif cfg["analysis"] == "Wilcoxon (paired)":
            pairwise_results = run_wilcoxon_paired(df_data, mode=scope, correction=correction, ref=ref, exclude=exclude_vs)

        elif cfg["analysis"] == "t_one_sample":
            one = run_t_one_sample(df_data, mu=mu_val)
            pairwise_results = one
            single_annos = [r for r in one if r["ref"] == default_ref]

        elif cfg["analysis"] == "Friedman":
            try:
                stat, p, groups = run_friedman(df_data)
                _write_xlsx(pd.DataFrame([{"chi2_F": float(stat), "p": float(p), "groups": ",".join(map(str, groups))}]), "friedman")
            except Exception as e:
                print(f"\nFriedman error: {e}")

    for r in pairwise_results:
        if "p_adj" not in r:
            r["p_adj"] = float(r.get("p_raw", np.nan))

    if not only_plot and cfg["export_csv"] and pairwise_results:
        pd.DataFrame(pairwise_results).to_csv(cfg["export_csv"], index=False)
        print(f"\nExported post-hoc results to: {cfg['export_csv']}")
    if pairwise_results:
        _write_xlsx(pd.DataFrame(pairwise_results), "posthoc")

    # Show long table
    if not only_plot:
        show_pairwise_long_table_figure(pairwise_results)

    # Plotting
    global_max = max([max(vals) for vals in data_dict.values()]) if any(len(v) for v in data_dict.values()) else 1.0
    base_gap = base_gap_factor * global_max
    stack_step = stack_step_factor * global_max
    line_h = line_height_factor * global_max
    x = np.arange(len(categories))
    means = np.array([np.mean(data_dict[c]) for c in categories], dtype=float)
    sems = np.array([np.std(data_dict[c], ddof=1)/np.sqrt(len(data_dict[c])) if len(data_dict[c])>1 else 0 for c in categories], dtype=float)
    bar_tops = np.array([max(data_dict[c]) if len(data_dict[c]) > 0 else (means[i] if len(means)>i else 0)
                         for i, c in enumerate(categories)], dtype=float)
    def _nice_number(xv: float) -> float:
        if xv <= 0: return 1.0
        exp = np.floor(np.log10(xv)); frac = xv / (10 ** exp)
        if frac <= 1: nice = 1
        elif frac <= 2: nice = 2
        elif frac <= 5: nice = 5
        else: nice = 10
        return nice * (10 ** exp)

    mode = cfg.get("bracket_mode", "Default ref only")
    selected_csv = cfg.get("selected_pairs", "").strip()
    selected_tokens = [s for s in (selected_csv.split(",") if selected_csv else []) if s]
    include_nonsig_flag = (cfg.get("include_nonsig","False").lower() == "true")

    if not only_plot:
        if mode == "All significant" or mode == "All (ignore significance)":
            all_tokens = []
            cats_sorted = sorted(categories)
            for i in range(len(cats_sorted)):
                for j in range(i+1, len(cats_sorted)):
                    all_tokens.append(f"{cats_sorted[i]}\n{cats_sorted[j]}")
            require_sig = (mode == "All significant")
            bracket_layout, required_ymax = compute_brackets_from_selection(
                categories, pairwise_results, alpha, x, bar_tops,
                base_gap, stack_step, line_h, all_tokens, require_sig, sig_marker_mode, collision_k, top_margin_factor
            )
        elif mode == "Custom…":
            bracket_layout, required_ymax = compute_brackets_from_selection(
                categories, pairwise_results, alpha, x, bar_tops,
                base_gap, stack_step, line_h, selected_tokens, not include_nonsig_flag, sig_marker_mode, collision_k, top_margin_factor
            )
        else:
            bracket_layout, required_ymax = compute_brackets_default_only(
                categories, pairwise_results, alpha, x, bar_tops,
                base_gap, stack_step, line_h, default_ref, sig_marker_mode, collision_k, top_margin_factor
            )
    else:
        bracket_layout, required_ymax = [], max(bar_tops) * 1.05 if len(bar_tops) else 1.0

    # Cap to 20 brackets for clarity; they’re already ordered to reduce overlaps
    if len(bracket_layout) > 20:
        bracket_layout = bracket_layout[:20]

    if not only_plot and cfg["analysis"] == "t_one_sample" and single_annos:
        required_ymax = max(required_ymax, bar_tops[categories.index(default_ref)] + base_gap + line_h)

    tallest = float(np.nanmax(bar_tops)) if len(bar_tops) else 1.0
    desired_step = max(1.0, tallest / 4.0)
    step = _nice_number(desired_step)
    step = max(1, int(round(step)))
    if tallest > 4 * step:
        step = max(1, int(_nice_number(tallest / 4.0 * 1.01)))
    labeled_max = 4 * step

    
    # Dynamic y-axis scaling
    if y_min_user is not None and y_max_user is not None:
        # Respect user-provided limits
        y_min = y_min_user
        y_max = y_max_user
    else:
        # Auto-scale to tallest data + margin, and ensure brackets fit
        y_min = 0.0
        tallest_bar = float(np.nanmax(bar_tops)) if len(bar_tops) else 1.0
        y_max = tallest_bar * 1.2  # 20% breathing room
    y_max = max(y_max, required_ymax)


    plt.rcParams.update({"figure.dpi":160, "axes.spines.top":False, "axes.spines.right":False, "axes.linewidth":1.0})

    # Figure size scales with brackets drawn (up to 20)
    height_in = max(3.6, 3.4 + 0.28 * (0 if only_plot else max(0, min(len(bracket_layout), 20))))
    width_in = max(3.2, 1.2 + 0.48 * len(categories))
    ptype = (cfg.get("plot_type") or "Bar + scatter").strip()

    # Plot type "None"
    if ptype == "None":
        if cfg["save_fig"]:
            print("\n[Plot None] 'save_fig' was set but plot type is 'None'; no figure was saved.")
        if excel_writer is not None:
            try:
                cfg_df = pd.DataFrame(sorted(cfg.items()), columns=["key","value"])
                cfg_df.to_excel(excel_writer, sheet_name="config", index=False)
                df_data.to_excel(excel_writer, sheet_name="raw_data", index=False)
                excel_writer.close()
                print(f"\nExported Excel workbook to: {xlsx_path} (sheets: {_excel_sheets_written + 2})")
            except Exception as e:
                print(f"\n[Excel] Failed to finalize workbook: {e}")
        return

    # --- Start plotting
    plt.ion()
    fig, ax = plt.subplots(figsize=(width_in, height_in))

    # Series factor & X column
    series_factor = cfg.get("series_factor","").strip()
    x_col = cfg.get("x_col","").strip()
    series_means = None
    series_colors = None
    x_vec = None
    if ptype in {"Lines (series)","Areas (series)","Regression (series)","Regression (global)"}:
        x_vec = build_group_x(categories, df_data, x_col if x_col else None)
        if ptype in {"Lines (series)","Areas (series)","Regression (series)"}:
            series_means = build_series_means(categories, df_data, series_factor if series_factor else None)
            series_levels_list = list(series_means.keys())
            series_colors = ensure_colors_for_keys(series_levels_list, {})

    # Dispatch
    if ptype == "Box":
        plot_box(ax, categories, data_dict, colors_cat)
    elif ptype == "Violin":
        plot_violin(ax, categories, data_dict, colors_cat)
    elif ptype == "Strip":
        plot_strip(ax, categories, data_dict, colors_cat)
    elif ptype == "Mean ± CI":
        n_vec = np.array([len(data_dict[c]) for c in categories], dtype=float)
        df_vec = np.maximum(n_vec - 1, 1)
        tcrit = stats.t.ppf(1 - 0.05/2, df=df_vec)
        ci_half = sems * tcrit
        plot_mean_ci(ax, categories, means, ci_half, colors_cat, bar=False, line=False, ci_label="95% CI")
    elif ptype == "Line ± CI":
        n_vec = np.array([len(data_dict[c]) for c in categories], dtype=float)
        df_vec = np.maximum(n_vec - 1, 1)
        tcrit = stats.t.ppf(1 - 0.05/2, df=df_vec)
        ci_half = sems * tcrit
        plot_mean_ci(ax, categories, means, ci_half, colors_cat, bar=False, line=True, ci_label="95% CI")
    elif ptype == "Line (means)":
        plot_line_means(ax, categories, means)
    elif ptype == "Area (quartiles stacked)":
        plot_area_quartiles(ax, categories, data_dict)
    elif ptype == "Lines (series)":
        plot_multi_lines(ax, x_vec, categories, series_means, series_colors)
    elif ptype == "Areas (series)":
        plot_multi_areas(ax, x_vec, categories, series_means, series_colors)
    elif ptype == "Regression (series)":
        run_and_plot_regressions(ax, x_vec, categories, series_means, series_colors)
    elif ptype == "Regression (global)":
        y_means_global = np.array([df_data[df_data["group"] == g]["value"].mean() for g in categories], dtype=float)
        run_and_plot_regression_global(ax, x_vec, categories, y_means_global, color="#2F3B52")
    elif ptype == "Pie chart":
        plot_pie(ax, categories, data_dict, colors_cat, value_mode="sum")
    else:
        bar_width = 0.70
        for i, cat in enumerate(categories):
            ax.bar(x[i], means[i], width=bar_width, color=colors_cat.get(cat,"#999999"),
                   edgecolor="#444444", linewidth=1.4, alpha=0.80, zorder=1)
        ax.errorbar(x, means, yerr=sems, fmt="none", ecolor="gray", elinewidth=1.5, capsize=6, capthick=1.5, zorder=2)
        rng = np.random.default_rng(42)
        for i, cat in enumerate(categories):
            yvals = np.array(data_dict[cat], dtype=float)
            jitter = rng.normal(0, 0.04, size=len(yvals))
            ax.scatter(np.full_like(yvals, x[i], dtype=float) + jitter, yvals,
                       s=28, color=colors_cat.get(cat,"#999999"),
                       edgecolor="#444444", linewidths=0.6, alpha=1.0, zorder=3)

    # Axis label and ticks/brackets
    if ptype == "Pie chart":
        ax.set_ylabel("")
    else:
        ax.set_ylabel(y_axis_label, fontsize=15)

    if not ptype.startswith("Regression") and ptype != "Pie chart":
        ax.set_ylim(y_min, y_max)

    # Build ticks automatically or from y-range
    if y_min_user is not None and y_max_user is not None:
        ticks = np.linspace(y_min, y_max, 5, endpoint=True)
    else:
        ticks = np.array([0, 1*step, 2*step, 3*step, 4*step], dtype=float)
    ax.set_yticks(ticks)
    formatter = ScalarFormatter(useMathText=True); formatter.set_scientific(False); formatter.set_useOffset(False)
    ax.yaxis.set_major_formatter(formatter)

    # Dynamic yticklabels formatting based on range
    yticklabels = []
    y_span = max(1e-12, (y_max - y_min))
    if y_span <= 1.0:
        yticklabels = [f"{val:.1f}" for val in ticks]
    elif y_max > 10000:
        yticklabels = [f"{val:.1e}" for val in ticks]
    else:
        yticklabels = [f"{int(val)}" for val in ticks]
    ax.set_yticklabels(yticklabels)

    # Spine widths
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1.0)

    if ptype != "Pie chart":
        for b in bracket_layout:
            ax.plot(
                [b["x1"], b["x1"], b["x2"], b["x2"]],
                [b["y"], b["y"] + line_h, b["y"] + line_h, b["y"]],
                c="black", lw=1.2, clip_on=False, zorder=4
            )
            ax.text(
                (b["x1"] + b["x2"]) / 2, b["y"] + line_h, b["text"],
                ha="center", va="bottom", fontsize=9, color="black", clip_on=False, zorder=4
            )

    if not only_plot and cfg["analysis"] == "t_one_sample" and single_annos and ptype != "Pie chart":
        xi = categories.index(default_ref); y_top = bar_tops[xi] + base_gap
        for r in single_annos:
            ax.text(
                x[xi], y_top + line_h, format_sig_marker(r["p_adj"], sig_marker_mode),
                ha="center", va="bottom", fontsize=9, color="black", clip_on=False, zorder=4
            )

    if (cfg.get("plot_type") or "").strip() in {
        "Mean ± CI","Line ± CI","Area (quartiles stacked)",
        "Lines (series)","Areas (series)","Regression (series)","Regression (global)"
    }:
        ax.legend(loc="best", fontsize=9)

    ax.set_title("", fontsize=10, pad=10)
    plt.tight_layout()

    if cfg["save_fig"]:
        ext = os.path.splitext(cfg["save_fig"])[1].lower()
        if ext in {".tif", ".tiff"}:
            fig.savefig(cfg["save_fig"], dpi=300, format="tiff", bbox_inches="tight")
        else:
            fig.savefig(cfg["save_fig"], dpi=300, bbox_inches="tight")
        print(f"\nSaved figure to: {cfg['save_fig']}")

    # Keep interactive window open to allow zoom/pan
    plt.show(block=True)

    # Optional histograms
    if cfg["show_hists"].lower() == "true":
        bins_spec = cfg["hist_bins"]
        try:
            max_h = int(float(cfg["hist_max"])); max_h = min(30, max(1, max_h))
        except: max_h = 30
        plot_histograms_panel(categories=categories, data=data_dict, colors=colors_cat, max_groups=max_h,
                              bins_spec=bins_spec, title=f"Histograms per group (max {max_h}, bins={bins_spec})")
        plt.ion(); plt.show(block=True)

    # Finalize Excel workbook
    if excel_writer is not None:
        try:
            cfg_df = pd.DataFrame(sorted(cfg.items()), columns=["key","value"])
            cfg_df.to_excel(excel_writer, sheet_name="config", index=False)
            df_data.to_excel(excel_writer, sheet_name="raw_data", index=False)
            excel_writer.close()
            print(f"\nExported Excel workbook to: {xlsx_path} (sheets: {_excel_sheets_written + 2})")
        except Exception as e:
            print(f"\n[Excel] Failed to finalize workbook: {e}")

# --------------------------------------------------------------------------------
# GUI (SCROLLABLE START MENU ADDED)
def open_config_gui():
    import platform

    root = tk.Tk()
    root.title("Analysis Setup")
    root.geometry("1200x800")  # Larger main window
    root.resizable(True, True)

    # ------------------------------
    # SCROLLABLE START MENU WRAPPER
    # ------------------------------
    # Root -> (Canvas + vertical Scrollbar); inside the Canvas we place `main` frame.
    outer = ttk.Frame(root)
    outer.pack(fill="both", expand=True)

    canvas = tk.Canvas(outer, highlightthickness=0)
    vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vscroll.set)

    canvas.pack(side="left", fill="both", expand=True)
    vscroll.pack(side="right", fill="y")

    # The actual content frame that used to be `main = ttk.Frame(root, ...)`
    main = ttk.Frame(canvas, padding=(8, 6))
    main_window_id = canvas.create_window((0, 0), window=main, anchor="nw")

    def on_main_configure(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))
    main.bind("<Configure>", on_main_configure)

    def on_canvas_configure(event):
        try:
            canvas.itemconfigure(main_window_id, width=event.width)
        except tk.TclError:
            pass
    canvas.bind("<Configure>", on_canvas_configure)

    # Mouse wheel support (Windows, macOS, Linux/X)
    def _bind_mousewheel(widget):
        system = platform.system()
        if system == "Windows":
            widget.bind_all("<MouseWheel>", _on_mousewheel_windows)
        elif system == "Darwin":
            widget.bind_all("<MouseWheel>", _on_mousewheel_darwin)
        else:
            widget.bind_all("<Button-4>", _on_mousewheel_x11)
            widget.bind_all("<Button-5>", _on_mousewheel_x11)

    def _unbind_mousewheel(widget):
        system = platform.system()
        if system == "Windows":
            widget.unbind_all("<MouseWheel>")
        elif system == "Darwin":
            widget.unbind_all("<MouseWheel>")
        else:
            widget.unbind_all("<Button-4>")
            widget.unbind_all("<Button-5>")

    def _on_mousewheel_windows(event):
        canvas.yview_scroll(-int(event.delta / 120) * 3, "units")

    def _on_mousewheel_darwin(event):
        canvas.yview_scroll(-int(event.delta), "units")

    def _on_mousewheel_x11(event):
        if event.num == 4:
            canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            canvas.yview_scroll(3, "units")

    canvas.bind("<Enter>", lambda _e: _bind_mousewheel(canvas))
    canvas.bind("<Leave>", lambda _e: _unbind_mousewheel(canvas))

    # ------------------------------
    # From here down your layout is the same, now inside `main`
    # ------------------------------

    data_dict: Dict[str, List[float]] = {}
    df_data = pd.DataFrame(columns=["group", "value"])
    categories: List[str] = []

    def default_ref(cats: List[str]) -> str:
        return "-dox" if "-dox" in cats else (cats[0] if cats else "")

    # Vars
    csv_path = tk.StringVar(value="")
    analysis = tk.StringVar(value="ANOVA")
    posthoc = tk.StringVar(value="None")
    correction= tk.StringVar(value="holm")
    scope = tk.StringVar(value="vs_ref")
    alpha = tk.StringVar(value="0.05")
    tick = tk.StringVar(value="auto")
    ci_level = tk.StringVar(value="0.95")
    default_ref_var = tk.StringVar(value="")
    ref_var = tk.StringVar(value="")
    exclude_var = tk.StringVar(value="")
    mu_var = tk.StringVar(value="0.0")
    save_fig = tk.StringVar(value="")
    export_csv = tk.StringVar(value="")
    export_xlsx = tk.StringVar(value="")
    y_label_var = tk.StringVar(value=(load_y_label_history()[0] if load_y_label_history() else "Value"))
    y_history = load_y_label_history()
    y_min_var = tk.StringVar(value="")
    y_max_var = tk.StringVar(value="")
    current_colors: Dict[str, str] = {}
    factor_a_var = tk.StringVar(value="")
    factor_b_var = tk.StringVar(value="")
    series_factor_var = tk.StringVar(value="")
    x_col_var = tk.StringVar(value="")
    sig_marker_mode = tk.StringVar(value="p-value")
    adv_show = tk.BooleanVar(value=False)
    base_gap_factor_var = tk.StringVar(value="0.08")
    stack_step_factor_var = tk.StringVar(value="0.09")
    line_height_factor_var = tk.StringVar(value="0.025")
    collision_k_var = tk.StringVar(value="0.75")
    top_margin_factor_var = tk.StringVar(value="0.12")
    show_hists = tk.BooleanVar(value=False)
    hist_bins = tk.StringVar(value="auto")
    hist_max = tk.StringVar(value="30")
    bracket_mode = tk.StringVar(value="Default ref only")
    selected_pairs_var = tk.StringVar(value="")
    include_nonsig = tk.BooleanVar(value=False)
    plot_type = tk.StringVar(value="Bar + scatter")
    interpret_wide_var = tk.BooleanVar(value=False)
    prev_bracket_mode = {"value": "Default ref only"}
    display_to_token: Dict[str, str] = {}
    token_to_display: Dict[str, str] = {}

    def build_pair_tokens(cats: List[str]) -> List[str]:
        toks = []
        cats_sorted = sorted(cats, key=str)
        for i in range(len(cats_sorted)):
            for j in range(i+1, len(cats_sorted)):
                a, b = cats_sorted[i], cats_sorted[j]
                toks.append(f"{a}\n{b}")
        return toks

    def rebuild_display_maps(cats: List[str]) -> List[str]:
        display_to_token.clear()
        token_to_display.clear()
        tokens = build_pair_tokens(cats)
        for tok in tokens:
            a, b = tok.split("\n", 1)
            disp = f"{a} ↔ {b}"
            display_to_token[disp] = tok
            token_to_display[tok] = disp
        return list(display_to_token.keys())

    # Layout
    r = 0
    ttk.Label(main, text="Data file (CSV/Excel) (group,value,[subject], …):").grid(row=r, column=0, sticky="w")
    ttk.Entry(main, textvariable=csv_path, width=36).grid(row=r, column=1, columnspan=2, sticky="we", padx=(4,0))
    ttk.Checkbutton(main, text="Interpret columns as groups", variable=interpret_wide_var).grid(row=r, column=3, sticky="e")
    r += 1

    def browse_data_file():
        nonlocal data_dict, df_data, categories
        path = filedialog.askopenfilename(
            title="Select data file",
            filetypes=[
                ("CSV or Excel", "*.csv *.xlsx *.xls"),
                ("CSV", "*.csv"),
                ("Excel", "*.xlsx *.xls"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            ext = os.path.splitext(path)[1].lower()
            chosen_sheet = None
            if ext in {".xlsx", ".xlsm", ".xltx", ".xltm", ".xls"}:
                sheets = _excel_sheet_names(path)
                if len(sheets) > 1:
                    chosen_sheet = _pick_excel_sheet_dialog(path)
                if chosen_sheet is None and len(sheets) > 1:
                    return
            if interpret_wide_var.get() and ext in {".xlsx", ".xlsm", ".xltx", ".xltm", ".xls"} and is_wide_excel(path, sheet=chosen_sheet):
                dd, df = load_excel_wide(path, sheet=chosen_sheet)
            else:
                dd, df = load_data(path, sheet=chosen_sheet)
            data_dict, df_data = dd, df
            categories = list(data_dict.keys())
            csv_path.set(path)
            refresh_groups()
            rebuild_color_rows()  # refresh colors UI with new groups
            messagebox.showinfo("File loaded", f"Loaded {len(data_dict)} groups from:\n{os.path.basename(path)}")
        except Exception as e:
            print("File load error:\n", traceback.format_exc())
            messagebox.showerror("Error loading file", f"{type(e).__name__}: {e}")

    ttk.Button(main, text="Browse…", command=browse_data_file).grid(row=r, column=3, sticky="e")
    r += 1

    ttk.Label(main, text="Primary analysis:").grid(row=r, column=0, sticky="w")
    analysis_cb = ttk.Combobox(
        main, textvariable=analysis,
        values=[
            "None",
            "Normality (all)","ANOVA","ANOVA (two-way)","Kruskal",
            "Mann–Whitney (2 groups)","t_ind_equal","t_ind_welch",
            "t_paired","Wilcoxon (paired)","t_one_sample","Friedman",
            "Descriptives"
        ],
        width=22, state="readonly"
    )
    analysis_cb.grid(row=r, column=1, sticky="w", padx=(4,0))
    ttk.Label(main, text="Post-hoc (optional):").grid(row=r, column=2, sticky="w")
    posthoc_cb = ttk.Combobox(main, textvariable=posthoc, values=["None"], width=24, state="readonly")
    posthoc_cb.grid(row=r, column=3, sticky="w", padx=(4,0))
    r += 1

    ttk.Label(main, text="Plot type:").grid(row=r, column=0, sticky="w")
    ttk.Combobox(
        main, textvariable=plot_type, width=24, state="readonly",
        values=[
            "None",
            "Bar + scatter", "Box", "Violin", "Strip",
            "Mean ± CI", "Line ± CI", "Line (means)",
            "Area (quartiles stacked)",
            "Lines (series)", "Areas (series)",
            "Regression (series)", "Regression (global)",
            "Pie chart"
        ]
    ).grid(row=r, column=1, sticky="w", padx=(4,0))
    r += 1

    ttk.Label(main, text="Series factor (for multi-lines/areas/regression):").grid(row=r, column=0, sticky="w")
    series_factor_cb = ttk.Combobox(main, textvariable=series_factor_var, values=[], width=24, state="readonly")
    series_factor_cb.grid(row=r, column=1, sticky="w", padx=(4,0))
    ttk.Label(main, text="X column (numeric, optional for regression):").grid(row=r, column=2, sticky="w")
    x_col_cb = ttk.Combobox(main, textvariable=x_col_var, values=[], width=24, state="readonly")
    x_col_cb.grid(row=r, column=3, sticky="w", padx=(4,0))
    r += 1

    ttk.Label(main, text="Comparison scope (pairwise):").grid(row=r, column=0, sticky="w")
    scope_cb = ttk.Combobox(main, textvariable=scope, values=["vs_ref","all_pairs"], width=18, state="readonly")
    scope_cb.grid(row=r, column=1, sticky="w", padx=(4,0))
    ttk.Label(main, text="Correction:").grid(row=r, column=2, sticky="w")
    ttk.Combobox(main, textvariable=correction, values=["bonferroni","holm","bh"], width=24, state="readonly").grid(row=r, column=3, sticky="w", padx=(4,0))
    r += 1

    ttk.Label(main, text="Significance marker:").grid(row=r, column=0, sticky="w")
    ttk.Combobox(main, textvariable=sig_marker_mode, values=["p-value","asterisks"], width=18, state="readonly").grid(row=r, column=1, sticky="w", padx=(4,0))
    r += 1

    ttk.Label(main, text="Alpha:").grid(row=r, column=0, sticky="w")
    ttk.Entry(main, textvariable=alpha, width=10).grid(row=r, column=1, sticky="w", padx=(4,0))
    ttk.Label(main, text="CI level (0-1):").grid(row=r, column=2, sticky="w")
    ttk.Entry(main, textvariable=ci_level, width=10).grid(row=r, column=3, sticky="w", padx=(4,0))
    r += 1

    ttk.Label(main, text="Y tick step (ignored):").grid(row=r, column=2, sticky="w")
    ttk.Entry(main, textvariable=tick, width=10).grid(row=r, column=3, sticky="w", padx=(4,0))
    r += 1

    ttk.Label(main, text="Default ref (brackets):").grid(row=r, column=0, sticky="w")
    default_ref_cb = ttk.Combobox(main, textvariable=default_ref_var, values=[], width=18, state="readonly")
    default_ref_cb.grid(row=r, column=1, sticky="w", padx=(4,0))
    ttk.Label(main, text="Analysis ref:").grid(row=r, column=2, sticky="w")
    ref_cb = ttk.Combobox(main, textvariable=ref_var, values=[], width=18, state="readonly")
    ref_cb.grid(row=r, column=3, sticky="w", padx=(4,0))
    r += 1

    ttk.Label(main, text="Exclude (vs_ref):").grid(row=r, column=0, sticky="w")
    exclude_cb = ttk.Combobox(main, textvariable=exclude_var, values=[], width=18, state="readonly")
    exclude_cb.grid(row=r, column=1, sticky="w", padx=(4,0))
    ttk.Label(main, text="One-sample μ:").grid(row=r, column=2, sticky="w")
    ttk.Entry(main, textvariable=mu_var, width=10).grid(row=r, column=3, sticky="w", padx=(4,0))
    r += 1

    ttk.Label(main, text="Factor A (two-way):").grid(row=r, column=0, sticky="w")
    factor_a_cb = ttk.Combobox(main, textvariable=factor_a_var, values=[], width=18, state="readonly")
    factor_a_cb.grid(row=r, column=1, sticky="w", padx=(4,0))
    ttk.Label(main, text="Factor B (two-way):").grid(row=r, column=2, sticky="w")
    factor_b_cb = ttk.Combobox(main, textvariable=factor_b_var, values=[], width=18, state="readonly")
    factor_b_cb.grid(row=r, column=3, sticky="w", padx=(4,0))
    r += 1

    # Y axis controls
    ttk.Label(main, text="Y-axis label:").grid(row=r, column=2, sticky="e", padx=(0,6))
    y_label_cb = ttk.Combobox(main, textvariable=y_label_var, values=y_history, width=22)
    y_label_cb.grid(row=r, column=3, sticky="w")
    r += 1

    ttk.Label(main, text="Y-axis min (optional):").grid(row=r, column=0, sticky="w")
    ttk.Entry(main, textvariable=y_min_var, width=12).grid(row=r, column=1, sticky="w", padx=(4,0))
    ttk.Label(main, text="Y-axis max (optional):").grid(row=r, column=2, sticky="w")
    ttk.Entry(main, textvariable=y_max_var, width=12).grid(row=r, column=3, sticky="w", padx=(4,0))
    r += 1

    subj_label = ttk.Label(main, text="Subject column: ✗ (paired/RM disabled)")
    subj_label.grid(row=r, column=0, columnspan=2, sticky="w")
    r += 1

    # Bracket mode
    ttk.Label(main, text="Bracket source:").grid(row=r, column=0, sticky="w")
    bracket_mode_cb = ttk.Combobox(
        main, textvariable=bracket_mode,
        values=["Default ref only", "All significant", "All (ignore significance)", "Custom…"],
        width=24, state="readonly"
    )
    bracket_mode_cb.grid(row=r, column=1, sticky="w", padx=(4,0))

    # Custom pairs frame
    custom_pairs_frame = ttk.LabelFrame(main, text="Bracket pairs (Custom)", padding=(8,6))
    custom_pairs_frame.grid(row=r, column=0, columnspan=4, sticky="nsew")
    r += 1

    ttk.Label(custom_pairs_frame, text="Choose a pair (all comparisons shown):").grid(row=0, column=0, sticky="w")
    pair_combo = ttk.Combobox(custom_pairs_frame, values=[], width=30, state="readonly")
    pair_combo.grid(row=0, column=1, sticky="w", padx=(4,0))
    add_btn = ttk.Button(custom_pairs_frame, text="Add")
    add_btn.grid(row=0, column=2, sticky="w", padx=(6,0))

    ttk.Label(custom_pairs_frame, text="Selected pairs:").grid(row=1, column=0, sticky="w", pady=(6,0))
    selected_list = tk.Listbox(custom_pairs_frame, height=8, width=36, exportselection=False)
    selected_list.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(2,0))
    remove_btn = ttk.Button(custom_pairs_frame, text="Remove")
    remove_btn.grid(row=2, column=2, sticky="nw", padx=(6,0))
    clear_btn = ttk.Button(custom_pairs_frame, text="Clear")
    clear_btn.grid(row=3, column=2, sticky="nw", padx=(6,0), pady=(4,0))
    include_nonsig_chk = ttk.Checkbutton(custom_pairs_frame, text="Include non-significant (ignore α)", variable=include_nonsig)
    include_nonsig_chk.grid(row=4, column=0, columnspan=2, sticky="w", pady=(6,0))
    done_btn = ttk.Button(custom_pairs_frame, text="Done")
    done_btn.grid(row=4, column=2, sticky="e", padx=(6,0))
    custom_pairs_frame.grid_remove()

    def on_add_pair():
        disp = pair_combo.get().strip()
        tok = display_to_token.get(disp)
        if not tok:
            messagebox.showwarning("Add pair", "Choose a pair from the dropdown."); return
        current = [s.strip() for s in (selected_pairs_var.get() or "").split(",") if s.strip()]
        if tok in current: return
        current.append(tok)
        selected_pairs_var.set(",".join(current))
        selected_list.insert("end", token_to_display.get(tok, tok))
    add_btn.configure(command=on_add_pair)

    def on_remove_selected():
        sel = list(selected_list.curselection())
        if not sel: return
        items = [selected_list.get(i) for i in sel]
        toks = [display_to_token.get(d, d) for d in items]
        current = [s.strip() for s in (selected_pairs_var.get() or "").split(",") if s.strip()]
        current = [c for c in current if c not in toks]
        selected_pairs_var.set(",".join(current))
        for i in reversed(sel): selected_list.delete(i)
    remove_btn.configure(command=on_remove_selected)

    def on_clear_pairs():
        selected_pairs_var.set("")
        selected_list.delete(0, "end")
    clear_btn.configure(command=on_clear_pairs)

    def rebuild_pairs_and_combo(cats: List[str]):
        displays = rebuild_display_maps(cats)
        pair_combo.configure(values=displays)
        if displays:
            pair_combo.set(displays[0]); add_btn.state(["!disabled"])
        else:
            pair_combo.set(""); add_btn.state(["disabled"])

    def refresh_groups():
        nonlocal categories
        categories = list(data_dict.keys())
        def_ref = default_ref(categories)
        for cbx, var in [(default_ref_cb, default_ref_var), (ref_cb, ref_var), (exclude_cb, exclude_var)]:
            cbx['values'] = categories + ([ "" ] if cbx is exclude_cb else [])
        default_ref_var.set(def_ref if def_ref else "")
        ref_var.set(default_ref_var.get() if default_ref_var.get() else "")
        if exclude_var.get() not in categories + [""]:
            exclude_var.set("")
        subj_label.configure(text="Subject column: ✓ (paired/RM available)" if "subject" in df_data.columns else "Subject column: ✗ (paired/RM disabled)")
        factor_candidates = [c for c in df_data.columns if c not in {"value"}]
        nonnum = [c for c in factor_candidates if not np.issubdtype(df_data[c].dropna().infer_objects().dtype, np.number)]
        factor_list = sorted(set([c for c in nonnum if c not in {"group","subject"}] or [c for c in factor_candidates if c not in {"group","subject"}]))
        factor_a_cb['values'] = factor_list
        factor_b_cb['values'] = factor_list
        series_candidates = sorted([c for c in df_data.columns if c not in {"group","value","subject"} and not np.issubdtype(df_data[c].dropna().infer_objects().dtype, np.number)])
        series_factor_cb['values'] = series_candidates
        if not series_factor_var.get() and series_candidates:
            series_factor_var.set(series_candidates[0])
        numeric_candidates = [c for c in df_data.columns if c not in {"value"} and np.issubdtype(df_data[c].dropna().infer_objects().dtype, np.number)]
        x_col_cb['values'] = sorted(numeric_candidates)
        if not x_col_var.get() and x_col_cb['values']:
            x_col_var.set("")
        rebuild_pairs_and_combo(categories)
        valid_tokens = set(build_pair_tokens(categories))
        current = [s.strip() for s in (selected_pairs_var.get() or "").split(",") if s.strip()]
        current = [tok for tok in current if tok in valid_tokens]
        selected_pairs_var.set(",".join(current))
        selected_list.delete(0, "end")
        for tok in current:
            selected_list.insert("end", token_to_display.get(tok, tok))
        y_label_cb['values'] = load_y_label_history()

    def update_posthoc_options(*_):
        a = analysis.get()
        opts = ["None"]
        if a == "None":
            opts = ["None"]
        elif a == "Normality (all)":
            opts = ["None"]
        elif a == "ANOVA":
            opts = ["None"]
            if _HAS_TUKEY: opts.append("Tukey HSD")
            if _HAS_SCPH:  opts.append("Games–Howell")
            opts += ["vs_ref (Welch)","vs_ref (Student)","all_pairs (Welch)","all_pairs (Student)"]
        elif a == "Kruskal":
            opts = ["None"]
            if _HAS_SCPH: opts.append("Dunn's")
            opts += ["vs_ref (Mann–Whitney)","all_pairs (Mann–Whitney)"]
        else:
            opts = ["None"]
        posthoc_cb['values'] = opts
        if posthoc.get() not in opts:
            posthoc.set(opts[0])

    def on_bracket_mode_changed(*_):
        mode = bracket_mode.get()
        if mode == "Custom…":
            if prev_bracket_mode["value"] == "Custom…":
                prev_bracket_mode["value"] = "Default ref only"
            custom_pairs_frame.grid()
            rebuild_pairs_and_combo(categories)
        else:
            prev_bracket_mode["value"] = mode
            custom_pairs_frame.grid_remove()

    bracket_mode_cb.bind("<<ComboboxSelected>>", on_bracket_mode_changed)

    def on_done_custom():
        bracket_mode.set(prev_bracket_mode["value"])
        on_bracket_mode_changed()
    done_btn.configure(command=on_done_custom)

    # COLORS UI
    colors_frame = ttk.LabelFrame(main, text="Group colors", padding=(8,6))
    colors_frame.grid(row=r, column=0, columnspan=4, sticky="we", pady=(6,0))
    r += 1

    colors_rows = ttk.Frame(colors_frame)
    colors_rows.grid(row=0, column=0, sticky="we")
    colors_frame.grid_columnconfigure(0, weight=1)
    color_buttons: Dict[str, ttk.Button] = {}

    def pick_color_for(group: str):
        cur = current_colors.get(group, "")
        if not (isinstance(cur, str) and cur.startswith("#") and len(cur) in (4, 7)):
            cur = "#777777"
        _, hex_ = colorchooser.askcolor(color=cur, title=f"Choose color for {group}")
        if hex_:
            current_colors[group] = hex_.upper()
            btn = color_buttons.get(group)
            if btn is not None:
                btn.configure(text=f"{group}: {hex_.upper()}")

    def rebuild_color_rows():
        for w in colors_rows.winfo_children():
            w.destroy()
        color_buttons.clear()
        for i, g in enumerate(categories):
            ttk.Label(colors_rows, text=g).grid(row=i, column=0, sticky="w", padx=(0,6), pady=2)
            hex_color = current_colors.get(g, "")
            if not (isinstance(hex_color, str) and hex_color.startswith("#") and len(hex_color) in (4, 7)):
                hex_color = ""
            btn_text = f"{g}: {hex_color or '(auto)'}"
            btn = ttk.Button(colors_rows, text=btn_text, command=lambda gr=g: pick_color_for(gr))
            btn.grid(row=i, column=1, sticky="w", padx=(0,6), pady=2)
            color_buttons[g] = btn

    toolbar = ttk.Frame(colors_frame)
    toolbar.grid(row=1, column=0, sticky="we", pady=(6,0))

    def clear_all_colors():
        current_colors.clear()
        rebuild_color_rows()

    def set_default_palette():
        auto = ensure_colors_for_keys(categories, {})
        current_colors.clear()
        current_colors.update(auto)
        rebuild_color_rows()

    ttk.Button(toolbar, text="Reset (auto)", command=clear_all_colors).grid(row=0, column=0, padx=(0,6))
    ttk.Button(toolbar, text="Fill with palette", command=set_default_palette).grid(row=0, column=1, padx=(0,6))

    rebuild_color_rows()

    # ACTION BUTTONS
    btns = ttk.Frame(main)
    btns.grid(row=r, column=0, columnspan=4, pady=(6,0), sticky="e")

    def on_run():
        save_y_label_to_history(y_label_var.get())
        y_label_cb['values'] = load_y_label_history()

        if not csv_path.get().strip():
            messagebox.showerror("Data file required", "Please load a CSV/Excel first."); return
        if not categories:
            messagebox.showerror("No data", "No groups were found in the loaded file."); return
        try:
            a = float(alpha.get()); assert 0 < a < 1
        except Exception:
            messagebox.showerror("Invalid alpha", "Alpha must be 0–1 (e.g., 0.05)"); return
        tv = tick.get().strip().lower()
        if tv != "auto":
            try: float(tv)
            except Exception:
                messagebox.showerror("Invalid tick step", "Use 'auto' or a number (e.g., 1000)"); return
        if analysis.get() == "ANOVA (two-way)":
            fa, fb = factor_a_var.get().strip(), factor_b_var.get().strip()
            if not fa or not fb or fa == fb:
                messagebox.showerror("Two-way ANOVA", "Pick two different factor columns (Factor A and Factor B)."); return
            if fa not in df_data.columns or fb not in df_data.columns:
                messagebox.showerror("Two-way ANOVA", "Selected factor columns not found in data."); return
        if posthoc.get() == "Tukey HSD" and not _HAS_TUKEY:
            messagebox.showerror("Missing dependency", "Tukey needs 'statsmodels'."); return
        if posthoc.get() in {"Dunn's","Games–Howell"} and not _HAS_SCPH:
            messagebox.showerror("Missing dependency", "This post-hoc needs 'scikit-posthocs'."); return
        if analysis.get() in {"t_paired", "Wilcoxon (paired)", "Friedman"} and "subject" not in df_data.columns:
            messagebox.showerror("Paired / RM tests", "Data must include 'subject' column."); return
        if analysis.get() != "None" and ref_var.get() not in categories:
            messagebox.showerror("Reference", "Analysis ref group not found in data."); return
        if default_ref_var.get() not in categories:
            messagebox.showerror("Default ref", "Default ref (brackets) not found in data."); return
        if exclude_var.get() not in categories + [""]:
            messagebox.showerror("Exclude", "Exclude must be one of the groups or empty."); return

        def _is_float(s):
            try:
                float(s); return True
            except:
                return False
        for label, var in [
            ("Bracket base gap", base_gap_factor_var),
            ("Stack step", stack_step_factor_var),
            ("Line height", line_height_factor_var),
            ("Collision K", collision_k_var),
            ("Top margin", top_margin_factor_var)
        ]:
            if not _is_float(var.get()):
                messagebox.showerror("Advanced spacing", f"{label} must be numeric."); return

        # Validate y-range fields (optional)
        if y_min_var.get().strip() != "" and not _is_float(y_min_var.get().strip()):
            messagebox.showerror("Y-axis", "Y min must be numeric (or leave blank)."); return
        if y_max_var.get().strip() != "" and not _is_float(y_max_var.get().strip()):
            messagebox.showerror("Y-axis", "Y max must be numeric (or leave blank)."); return

        try:
            hmax = int(float(hist_max.get()))
            if hmax < 1 or hmax > 30: raise ValueError
        except Exception:
            messagebox.showerror("Histograms", "Max histograms must be an integer between 1 and 30."); return

        cfg = {
            "csv": csv_path.get().strip(),
            "analysis": analysis.get(),
            "posthoc": posthoc.get(),
            "correction": correction.get(),
            "scope": scope.get(),
            "alpha": alpha.get().strip(),
            "tick": tick.get().strip(),
            "default_ref": default_ref_var.get(),
            "ref": ref_var.get(),
            "exclude": exclude_var.get().strip(),
            "mu": mu_var.get().strip(),
            "y_label": y_label_var.get().strip(),
            "save_fig": save_fig.get().strip(),
            "export_csv": export_csv.get().strip(),
            "export_xlsx": export_xlsx.get().strip(),
            "base_gap_factor": base_gap_factor_var.get().strip(),
            "stack_step_factor": stack_step_factor_var.get().strip(),
            "line_height_factor": line_height_factor_var.get().strip(),
            "collision_k": collision_k_var.get().strip(),
            "top_margin_factor": top_margin_factor_var.get().strip(),
            "show_hists": "True" if show_hists.get() else "False",
            "hist_bins": hist_bins.get().strip(),
            "hist_max": hist_max.get().strip(),
            "bracket_mode": bracket_mode.get().strip(),
            "selected_pairs": selected_pairs_var.get().strip(),
            "include_nonsig": "True" if include_nonsig.get() else "False",
            "factor_a": factor_a_var.get().strip(),
            "factor_b": factor_b_var.get().strip(),
            "series_factor": series_factor_var.get().strip(),
            "x_col": x_col_var.get().strip(),
            # Colors picked in UI
            "colors_json": json.dumps(ensure_colors_for_keys(categories, current_colors)),
            "plot_type": plot_type.get().strip(),
            "ci_level": ci_level.get().strip(),
            "sig_marker_mode": sig_marker_mode.get().strip(),
            # New y-axis range fields
            "y_min": y_min_var.get().strip(),
            "y_max": y_max_var.get().strip(),
        }
        execute_analysis(cfg, data_dict, df_data)

    def on_close():
        root.destroy()

    ttk.Button(btns, text="Run", command=on_run).grid(row=0, column=0, padx=6)
    ttk.Button(btns, text="Close", command=on_close).grid(row=0, column=1, padx=6)
    r += 1

    adv_toggle = ttk.Checkbutton(
        main,
        text="Show advanced significance bars (spacing)",
        variable=adv_show,
        command=lambda: (adv_frame.grid() if adv_show.get() else adv_frame.grid_remove())
    )
    adv_toggle.grid(row=r, column=0, columnspan=4, sticky="w")
    r += 1

    adv_frame = ttk.Frame(main, padding=(8,6))
    adv_frame.grid(row=r, column=0, columnspan=4, sticky="nsew")
    r += 1

    ttk.Label(adv_frame, text="Bracket base gap factor:").grid(row=0, column=0, sticky="w")
    ttk.Entry(adv_frame, textvariable=base_gap_factor_var, width=8).grid(row=0, column=1, sticky="w", padx=(4,0))
    ttk.Label(adv_frame, text="Stack step factor:").grid(row=0, column=2, sticky="w")
    ttk.Entry(adv_frame, textvariable=stack_step_factor_var, width=8).grid(row=0, column=3, sticky="w", padx=(4,0))

    ttk.Label(adv_frame, text="Line height factor:").grid(row=1, column=0, sticky="w")
    ttk.Entry(adv_frame, textvariable=line_height_factor_var, width=8).grid(row=1, column=1, sticky="w", padx=(4,0))
    ttk.Label(adv_frame, text="Collision K:").grid(row=1, column=2, sticky="w")
    ttk.Entry(adv_frame, textvariable=collision_k_var, width=8).grid(row=1, column=3, sticky="w", padx=(4,0))

    ttk.Label(adv_frame, text="Top margin factor:").grid(row=2, column=0, sticky="w")
    ttk.Entry(adv_frame, textvariable=top_margin_factor_var, width=8).grid(row=2, column=1, sticky="w", padx=(4,0))

    adv_frame.grid_remove()

    # I/O frame
    io_frame = ttk.Frame(main, padding=(0,4))
    io_frame.grid(row=r, column=0, columnspan=4, sticky="we")

    ttk.Label(io_frame, text="Save figure to:").grid(row=0, column=0, sticky="w")
    ttk.Entry(io_frame, textvariable=save_fig, width=34).grid(row=0, column=1, sticky="we", padx=(4,6))

    def _browse_save_fig():
        path = filedialog.asksaveasfilename(
            title="Save figure",
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("TIFF","*.tif *.tiff"),("SVG","*.svg"),("PDF","*.pdf"),("All files","*.*")]
        )
        if path:
            save_fig.set(path)
    ttk.Button(io_frame, text="Browse…", command=_browse_save_fig).grid(row=0, column=2, sticky="w", padx=(4,0))

    ttk.Label(io_frame, text="Export CSV to:").grid(row=0, column=3, sticky="w")
    ttk.Entry(io_frame, textvariable=export_csv, width=34).grid(row=0, column=4, sticky="we", padx=(4,0))

    def _browse_export_csv():
        path = filedialog.asksaveasfilename(
            title="Export CSV",
            defaultextension=".csv",
            filetypes=[("CSV","*.csv"),("All files","*.*")]
        )
        if path:
            export_csv.set(path)
    ttk.Button(io_frame, text="Browse…", command=_browse_export_csv).grid(row=0, column=5, sticky="w", padx=(4,0))

    ttk.Label(io_frame, text="Export Excel (.xlsx) to:").grid(row=1, column=0, sticky="w", pady=(4,0))
    ttk.Entry(io_frame, textvariable=export_xlsx, width=34).grid(row=1, column=1, sticky="we", padx=(4,6), pady=(4,0))

    def _browse_export_xlsx():
        path = filedialog.asksaveasfilename(
            title="Save results workbook",
            defaultextension=".xlsx",
            filetypes=[("Excel workbook","*.xlsx"),("All files","*.*")]
        )
        if path:
            export_xlsx.set(path)
    ttk.Button(io_frame, text="Browse…", command=_browse_export_xlsx).grid(row=1, column=2, sticky="w", padx=(4,0), pady=(4,0))

    for c in range(6):
        io_frame.grid_columnconfigure(c, weight=1)
    for c in range(4):
        main.grid_columnconfigure(c, weight=1)

    analysis_cb.bind("<<ComboboxSelected>>", update_posthoc_options)
    update_posthoc_options()
    on_bracket_mode_changed()

    root.mainloop()

# Main
def main():
    open_config_gui()

if __name__ == "__main__":
    main()
