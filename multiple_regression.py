# multiple_regression.py
# --------------------------------------------------------------------------------
# Multiple regression pop-up window (Tkinter) matching the style of your
# Power Analysis tool: modal Toplevel with minimize & caption buttons,
# import groups, select Y and multiple X, compute OLS, and show R²
# contributions per predictor (unique ΔR² and Shapley/LMG).
#
# Public entry:
#   MultipleRegressionWindow(master, get_groups_callback=None, title="Multiple regression")
#   - get_groups_callback: Optional[Callable[[], dict[str, np.ndarray]]]
#     Should return {group_name: 1D numpy array} with numeric data.
#
# Dependencies: numpy, scipy (for t-dist), tkinter. statsmodels optional.
# New in this merged version: Matplotlib plot window (Fit + Residuals) with R².
# --------------------------------------------------------------------------------
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
import math
import numpy as np
from typing import Optional, Callable, Dict, List, Tuple
from itertools import combinations
from random import shuffle, seed as rand_seed

# Optional: statsmodels for p-values/SE. We will fallback if missing.
try:
    import statsmodels.api as sm
    HAVE_SM = True
except Exception:
    HAVE_SM = False

# Optional: SciPy for t distribution CDF (p-values) if statsmodels missing
try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# Matplotlib for plotting in Tkinter
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# ------------------------------ Math helpers ------------------------------

def _ols_fit(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    OLS via lstsq.
    Returns: (beta, yhat, SSE, SST, R2)
    """
    # Solve
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    # SSE/SST/R²
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - np.mean(y))**2))
    r2 = 0.0 if sst <= 0 else 1.0 - (sse / sst)
    return beta, yhat, sse, sst, r2

def _adjusted_r2(r2: float, n: int, p: int) -> float:
    """Adjusted R² with intercept counted in p (so typical p = #predictors + 1)."""
    if n <= p or n <= 1:
        return float('nan')
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p)

def _add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones((X.shape[0], 1)), X])

def _unique_delta_r2(X: np.ndarray, y: np.ndarray, include_intercept: bool = True) -> Tuple[float, List[float]]:
    """
    Compute full R² and unique ΔR² for each predictor as:
    ΔR²_j = R²_full - R²_without_j
    """
    Xfull = _add_intercept(X) if include_intercept else X
    _, _, _, sst, r2_full = _ols_fit(Xfull, y)
    deltas: List[float] = []
    k = X.shape[1]
    for j in range(k):
        # model without j-th predictor
        keep = [i for i in range(k) if i != j]
        Xmj = X[:, keep]
        if include_intercept:
            Xmj = _add_intercept(Xmj)
        _, _, _, _, r2_mj = _ols_fit(Xmj, y)
        deltas.append(max(0.0, r2_full - r2_mj))  # guard small negative due to numeric issues
    return r2_full, deltas

def _r2_of_subset(X: np.ndarray, y: np.ndarray, subset: List[int], include_intercept: bool) -> float:
    if len(subset) == 0:
        # Intercept-only model (if include_intercept); otherwise 0 predictors with no intercept
        if include_intercept:
            Xsub = np.ones((X.shape[0], 1))
        else:
            return 0.0
    else:
        Xsub = X[:, subset]
    if include_intercept:
        Xsub = _add_intercept(Xsub)
    _, _, _, sst, r2 = _ols_fit(Xsub, y)
    return r2

def _shapley_lmg_exact(X: np.ndarray, y: np.ndarray, include_intercept: bool = True, max_exact: int = 10) -> List[float]:
    """
    Exact Shapley/LMG R² decomposition.
    For each predictor j, contribution is average of incremental R² over all permutations.
    Implemented via subset enumeration: sum over all S ⊆ P\{j}
        weight = S! * (k-|S|-1)! / k!
        contribution_j += weight * [R²(S ∪ {j}) - R²(S)]
    Complexity ~ O(k * 2^k). Use only for small k (<= max_exact).
    """
    k = X.shape[1]
    if k > max_exact:
        raise ValueError("Too many predictors for exact Shapley.")
    # Precompute R² for all subsets using bit masks
    r2_cache: Dict[int, float] = {}
    idx_all = list(range(k))
    for r in range(0, k + 1):
        for subset in combinations(idx_all, r):
            mask = 0
            for j in subset:
                mask |= (1 << j)
            r2_cache[mask] = _r2_of_subset(X, y, list(subset), include_intercept)
    # factorials
    fact = [1] * (k + 1)
    for i in range(2, k + 1):
        fact[i] = fact[i - 1] * i
    k_factorial = fact[k]

    contrib = [0.0] * k
    for j in idx_all:
        others = [i for i in idx_all if i != j]
        for r in range(0, k):
            for subset in combinations(others, r):
                mask_s = 0
                for a in subset:
                    mask_s |= (1 << a)
                mask_sj = mask_s | (1 << j)
                w = fact[r] * fact[k - r - 1] / k_factorial
                contrib[j] += w * (r2_cache[mask_sj] - r2_cache[mask_s])
    # Guard against tiny negatives
    contrib = [max(0.0, float(c)) for c in contrib]
    return contrib

def _shapley_lmg_mc(
    X: np.ndarray,
    y: np.ndarray,
    include_intercept: bool = True,
    n_perm: int = 2000,
    random_seed: Optional[int] = 1234,
) -> List[float]:
    """
    Monte-Carlo Shapley/LMG by averaging incremental R² over random permutations.
    """
    if random_seed is not None:
        rand_seed(random_seed)
    k = X.shape[1]
    idx_all = list(range(k))
    contrib = np.zeros(k, dtype=float)
    for _ in range(n_perm):
        order = idx_all[:]
        shuffle(order)
        selected: List[int] = []
        r2_prev = _r2_of_subset(X, y, selected, include_intercept)
        for j in order:
            selected_j = selected + [j]
            r2_new = _r2_of_subset(X, y, selected_j, include_intercept)
            contrib[j] += (r2_new - r2_prev)
            selected = selected_j
            r2_prev = r2_new
    contrib = np.maximum(0.0, contrib / float(max(1, n_perm)))
    return list(map(float, contrib))

def _standardize(arr: np.ndarray) -> np.ndarray:
    mu = np.nanmean(arr)
    sd = np.nanstd(arr, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    return (arr - mu) / sd

def _pvalues_from_beta(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute (se, t, p) for coefficients using classical OLS formulas.
    Assumes X includes intercept if model has one.
    """
    n, p = X.shape
    yhat = X @ beta
    resid = y - yhat
    sse = np.sum(resid**2)
    dof = max(1, n - p)
    sigma2 = sse / dof
    # (X'X)^-1 diagonal
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(0.0, np.diag(XtX_inv) * sigma2))
    tvals = np.where(se > 0, beta / se, np.nan)
    if HAVE_SCIPY:
        pvals = 2.0 * (1.0 - stats.t.cdf(np.abs(tvals), df=dof))
    else:
        # Fallback: normal approximation if SciPy missing
        from math import erf
        def norm_sf(z: float) -> float:
            return 0.5 * (1 - erf(z / math.sqrt(2)))
        pvals = np.array([2 * norm_sf(abs(float(t))) if np.isfinite(t) else np.nan for t in tvals])
    return se, tvals, pvals

# ------------------------------ UI class ------------------------------
class MultipleRegressionWindow(tk.Toplevel):
    PADX = 10
    PADY = 6
    WRAP = 320

    def __init__(
        self,
        master: Optional[tk.Misc] = None,
        get_groups_callback: Optional[Callable[[], Dict[str, np.ndarray]]] = None,
        title: str = "Multiple regression"
    ):
        super().__init__(master)
        self.title(title)
        self.transient(master)
        self.grab_set()
        self.resizable(True, True)
        # Best-effort "maximized" start
        try:
            self.state("zoomed")
        except Exception:
            try:
                self.attributes("-zoomed", True)
            except Exception:
                sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
                self.geometry(f"{sw}x{sh}+0+0")

        self.get_groups_callback = get_groups_callback

        # Header
        ttk.Label(
            self,
            text="Multiple regression – select Y and predictors (X).",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=self.PADX, pady=(self.PADY, 0))

        # Main content area: left panel (groups), right panel (options + results)
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=self.PADX, pady=self.PADY)
        main.grid_columnconfigure(0, weight=0)
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=1)

        # Left: groups list and selection
        left = ttk.LabelFrame(main, text="Groups (import from app)")
        left.grid(row=0, column=0, sticky="nsw", padx=(0, self.PADX), pady=(0, self.PADY))
        left.grid_rowconfigure(3, weight=1)

        self.btn_import = ttk.Button(left, text="Import groups", command=self._import_groups)
        self.btn_import.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(6, 6))

        ttk.Label(left, text="Target (Y):").grid(row=1, column=0, sticky="w")
        self.y_combo = ttk.Combobox(left, state="readonly", values=[])
        self.y_combo.grid(row=1, column=1, sticky="ew", padx=(6, 0))

        ttk.Label(left, text="Predictors (X):").grid(row=2, column=0, sticky="w", pady=(6, 0))
        # Select predictors via Listbox with MULTIPLE selection
        self.x_list = tk.Listbox(left, selectmode="extended", height=18, exportselection=False)
        self.x_list.grid(row=3, column=0, columnspan=2, sticky="nsew")
        sb = ttk.Scrollbar(left, orient="vertical", command=self.x_list.yview)
        self.x_list.configure(yscrollcommand=sb.set)
        sb.grid(row=3, column=2, sticky="ns")

        # Predictor order controls (affects "ordered incremental ΔR²" display)
        order_bar = ttk.Frame(left)
        order_bar.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        ttk.Button(order_bar, text="Select all", command=lambda: self.x_list.select_set(0, tk.END)).pack(side="left")
        ttk.Button(order_bar, text="Clear", command=lambda: self.x_list.select_clear(0, tk.END)).pack(side="left", padx=(6, 0))
        ttk.Button(order_bar, text="▲", width=3, command=self._move_up).pack(side="left", padx=(10, 0))
        ttk.Button(order_bar, text="▼", width=3, command=self._move_down).pack(side="left")

        # Right: options / actions / results
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(3, weight=1)

        # Options
        opts = ttk.LabelFrame(right, text="Options")
        opts.grid(row=0, column=0, sticky="ew", pady=(0, self.PADY))

        self.var_intercept = tk.BooleanVar(value=True)
        self.var_standardize = tk.BooleanVar(value=False)
        self.var_shapley_mode = tk.StringVar(value="auto")  # "auto" / "exact" / "mc"
        self.var_mc_perm = tk.StringVar(value="2000")
        self.var_seed = tk.StringVar(value="1234")

        row = 0
        ttk.Checkbutton(opts, text="Include intercept", variable=self.var_intercept).grid(row=row, column=0, sticky="w")
        ttk.Checkbutton(opts, text="Standardize X (z-score)", variable=self.var_standardize).grid(row=row, column=1, sticky="w", padx=(12, 0))
        row += 1
        ttk.Label(opts, text="Shapley/LMG mode:").grid(row=row, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(opts, textvariable=self.var_shapley_mode, state="readonly", width=10, values=("auto", "exact", "mc")).grid(row=row, column=1, sticky="w", pady=(6, 0))
        row += 1
        ttk.Label(opts, text="Monte-Carlo permutations (for mc/auto):").grid(row=row, column=0, sticky="w")
        ttk.Entry(opts, textvariable=self.var_mc_perm, width=8).grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(opts, text="Random seed:").grid(row=row, column=0, sticky="w")
        ttk.Entry(opts, textvariable=self.var_seed, width=8).grid(row=row, column=1, sticky="w")

        # Actions
        actions = ttk.Frame(right)
        actions.grid(row=1, column=0, sticky="ew", pady=(0, self.PADY))

        # NEW: keep last-run results so the plot button can use them
        self._last_fit = None  # type: Optional[Dict[str, object]]

        ttk.Button(actions, text="Run regression", command=self._run).pack(side="right")
        ttk.Button(actions, text="Show plot", command=self._show_plot_window).pack(side="right", padx=(0, 8))

        # Small status line (what was imported)
        self.lbl_status = ttk.Label(right, text="", foreground="#555")
        self.lbl_status.grid(row=2, column=0, sticky="w", pady=(0, 4))

        # Results (Text)
        results = ttk.LabelFrame(right, text="Results")
        results.grid(row=3, column=0, sticky="nsew")
        results.grid_columnconfigure(0, weight=1)
        results.grid_rowconfigure(0, weight=1)

        self.txt = tk.Text(results, wrap="word", height=22)
        self.txt.configure(font=("Consolas", 10))
        self.txt.grid(row=0, column=0, sticky="nsew")
        sbr = ttk.Scrollbar(results, orient="vertical", command=self.txt.yview)
        self.txt.configure(yscrollcommand=sbr.set)
        sbr.grid(row=0, column=1, sticky="ns")

        # Bottom bar
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=self.PADX, pady=(0, self.PADY))
        ttk.Button(bottom, text="Copy results to clipboard", command=self._copy_results).pack(side="left")
        ttk.Button(bottom, text="Minimize", command=self._minimize_self).pack(side="left", padx=(8, 0))
        ttk.Button(bottom, text="Close", command=self.destroy).pack(side="right")

        # Ensure Windows caption buttons (min/max/close) are present
        self._ensure_caption_buttons()

        # Keyboard shortcut: minimize
        self.bind_all("<Alt-m>", lambda e: self._minimize_self())
        self.bind_all("<Alt-M>", lambda e: self._minimize_self())

        # Internal storage
        self._groups: Dict[str, np.ndarray] = {}

    # ------------------------------ Window helpers ------------------------------
    def _minimize_self(self):
        try:
            self.iconify()
        except Exception:
            try:
                self.withdraw()
                self.after(50, self.deiconify)
            except Exception:
                pass

    def _ensure_caption_buttons(self):
        """Windows: force caption buttons visibility via style flags. No-op elsewhere."""
        try:
            import sys
            if sys.platform != "win32":
                return
            import ctypes
            hwnd = self.winfo_id()
            GWL_STYLE = -16
            WS_MINIMIZEBOX = 0x00020000
            WS_MAXIMIZEBOX = 0x00010000
            WS_SYSMENU = 0x00080000
            WS_THICKFRAME = 0x00040000
            GetWindowLong = ctypes.windll.user32.GetWindowLongW
            SetWindowLong = ctypes.windll.user32.SetWindowLongW
            SetWindowPos = ctypes.windll.user32.SetWindowPos
            style = GetWindowLong(hwnd, GWL_STYLE)
            style |= (WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU | WS_THICKFRAME)
            SetWindowLong(hwnd, GWL_STYLE, style)
            SWP_NOSIZE = 0x0001
            SWP_NOMOVE = 0x0002
            SWP_NOZORDER = 0x0004
            SWP_FRAMECHANGED = 0x0020
            SetWindowPos(hwnd, 0, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED)
        except Exception:
            pass

    def _copy_results(self):
        try:
            content = self.txt.get("1.0", "end").strip()
            self.clipboard_clear()
            self.clipboard_append(content)
            self.update()
        except Exception:
            pass

    # ------------------------------ Import & selection ------------------------------
    def _import_groups(self):
        if not self.get_groups_callback:
            messagebox.showinfo("Import", "No get_groups_callback provided by the app.")
            return
        try:
            data = self.get_groups_callback() or {}
            # Clean numeric arrays
            clean: Dict[str, np.ndarray] = {}
            for k, v in data.items():
                arr = np.asarray(v, dtype=float).ravel()
                arr = arr[np.isfinite(arr)]
                if arr.size > 1:
                    clean[k] = arr
            if not clean:
                messagebox.showinfo("Import", "No usable numeric groups were found.")
                return
            self._groups = clean
            names = sorted(clean.keys())
            self.y_combo["values"] = names
            self.x_list.delete(0, tk.END)
            for nm in names:
                self.x_list.insert(tk.END, nm)
            self.lbl_status.configure(text=f"Imported {len(names)} groups.")
        except Exception as ex:
            messagebox.showerror("Import", f"Could not import groups: {ex}")

    def _move_up(self):
        sel = list(self.x_list.curselection())
        if not sel:
            return
        for i in sel:
            if i == 0:
                continue
            txt = self.x_list.get(i)
            txt_above = self.x_list.get(i - 1)
            self.x_list.delete(i - 1, i)
            self.x_list.insert(i - 1, txt)
            self.x_list.insert(i, txt_above)
        self.x_list.select_clear(0, tk.END)
        for i in [max(0, s - 1) for s in sel]:
            self.x_list.select_set(i)

    def _move_down(self):
        sel = list(self.x_list.curselection())
        if not sel:
            return
        # reverse to avoid swap conflicts
        for i in reversed(sel):
            if i >= self.x_list.size() - 1:
                continue
            txt = self.x_list.get(i)
            txt_below = self.x_list.get(i + 1)
            self.x_list.delete(i, i + 1)
            self.x_list.insert(i, txt_below)
            self.x_list.insert(i + 1, txt)
        self.x_list.select_clear(0, tk.END)
        for i in [min(self.x_list.size() - 1, s + 1) for s in sel]:
            self.x_list.select_set(i)

    # ------------------------------ Main runner ------------------------------
    def _run(self):
        # Validate selections
        y_name = self.y_combo.get().strip()
        x_indices = list(self.x_list.curselection())
        x_names = [self.x_list.get(i) for i in x_indices]
        if not y_name:
            messagebox.showinfo("Run", "Please select a target (Y).")
            return
        if not x_names:
            messagebox.showinfo("Run", "Please select at least one predictor (X).")
            return
        if y_name in x_names:
            messagebox.showinfo("Run", "The target (Y) cannot also be a predictor.")
            return

        # Build aligned dataset: truncate to min length across selected series
        try:
            arrays = [self._groups[y_name]] + [self._groups[nm] for nm in x_names]
        except KeyError as ke:
            messagebox.showerror("Run", f"Group not found: {ke}")
            return
        lengths = [arr.size for arr in arrays]
        n_common = min(lengths)
        if n_common < 3:
            messagebox.showinfo("Run", "Not enough data points after alignment.")
            return
        # Align all to n_common (truncate from the end for simplicity)
        arrays = [arr[:n_common] for arr in arrays]
        y = arrays[0].astype(float)
        Xraw = np.column_stack(arrays[1:]).astype(float)  # shape (n, k)
        n, k = Xraw.shape

        # Standardize if requested
        X = Xraw.copy()
        if self.var_standardize.get():
            for j in range(k):
                X[:, j] = _standardize(X[:, j])

        include_intercept = self.var_intercept.get()

        # Fit OLS: coefficients, R²
        if include_intercept:
            Xfit = _add_intercept(X)
        else:
            Xfit = X

        if HAVE_SM:
            try:
                model = sm.OLS(y, Xfit)
                res = model.fit()
                beta = res.params
                se = res.bse
                tvals = res.tvalues
                pvals = res.pvalues
                r2_full = float(res.rsquared)
                r2_adj = float(res.rsquared_adj)
                yhat = np.asarray(res.fittedvalues, dtype=float)
            except Exception:
                # fallback to manual
                beta, yhat, sse, sst, r2_full = _ols_fit(Xfit, y)
                r2_adj = _adjusted_r2(r2_full, n, Xfit.shape[1])
                se, tvals, pvals = _pvalues_from_beta(Xfit, y, beta)
        else:
            beta, yhat, sse, sst, r2_full = _ols_fit(Xfit, y)
            r2_adj = _adjusted_r2(r2_full, n, Xfit.shape[1])
            se, tvals, pvals = _pvalues_from_beta(Xfit, y, beta)

        # Unique ΔR² for each predictor (drop-one)
        r2_full_chk, deltas = _unique_delta_r2(X, y, include_intercept=include_intercept)
        # guard potential tiny numeric mismatch
        if abs(r2_full - r2_full_chk) > 1e-10:
            r2_full = r2_full_chk

        # Shapley/LMG contributions
        mode = self.var_shapley_mode.get()
        max_exact = 10
        shapley: Optional[List[float]] = None
        try:
            if mode == "exact" or (mode == "auto" and k <= max_exact):
                shapley = _shapley_lmg_exact(X, y, include_intercept=include_intercept, max_exact=max_exact)
            else:
                n_perm = int(float(self.var_mc_perm.get()))
                seed_val = int(float(self.var_seed.get()))
                shapley = _shapley_lmg_mc(X, y, include_intercept=include_intercept, n_perm=n_perm, random_seed=seed_val)
        except Exception:
            shapley = None  # We'll just show ΔR² table if Shapley fails

        # Ordered incremental ΔR² in the selected order (optional extra)
        ordered_inc: List[float] = []
        selected: List[int] = []
        r2_prev = _r2_of_subset(X, y, selected, include_intercept)
        for j in range(k):
            subset = list(range(j + 1))  # predictors in the visual order (x_names)
            r2_new = _r2_of_subset(X, y, subset, include_intercept)
            ordered_inc.append(max(0.0, r2_new - r2_prev))
            r2_prev = r2_new

        # Save for plotting window
        self._last_fit = {
            "y": np.asarray(y, dtype=float),
            "yhat": np.asarray(yhat, dtype=float),
            "r2": float(r2_full),
            "y_name": y_name,
            "x_names": x_names[:],
        }

        # Compose output
        self.txt.delete("1.0", "end")
        self.txt.insert("end", f"Multiple regression (n={n}, predictors={k})\n")
        self.txt.insert("end", "-" * 72 + "\n")
        self.txt.insert("end", f"Y (target): {y_name}\n")
        self.txt.insert("end", f"X (predictors, in order): {', '.join(x_names)}\n")
        self.txt.insert("end", f"Intercept: {'Yes' if include_intercept else 'No'}  "
                        f"Standardize X: {'Yes' if self.var_standardize.get() else 'No'}\n\n")
        self.txt.insert("end", f"Model fit:\n")
        self.txt.insert("end", f" R² = {r2_full:.6f}\n")
        self.txt.insert("end", f" Adjusted R² = {r2_adj:.6f}\n\n")

        # Coefficient table
        self.txt.insert("end", "Coefficients (beta ± SE) [t, p]\n")
        names_for_coef = (["Intercept"] if include_intercept else []) + x_names
        for i, nm in enumerate(names_for_coef):
            b = float(beta[i])
            se_i = float(se[i]) if i < len(se) else float('nan')
            t_i = float(tvals[i]) if i < len(tvals) else float('nan')
            p_i = float(pvals[i]) if i < len(pvals) else float('nan')
            self.txt.insert("end", f" {nm:<18} {b:+.6g} ± {se_i:.6g} [t={t_i:.4g}, p={p_i:.4g}]\n")
        self.txt.insert("end", "\n")

        # Contributions
        self.txt.insert("end", "Unique contributions (drop-one ΔR²):\n")
        for nm, d in zip(x_names, deltas):
            pct = (d / r2_full * 100.0) if r2_full > 0 else float('nan')
            self.txt.insert("end", f" {nm:<18} ΔR² = {d:.6f} ({pct:.2f}% of R²)\n")
        self.txt.insert("end", "\n")

        self.txt.insert("end", "Ordered incremental contributions (per current predictor order):\n")
        cum = 0.0
        for nm, inc in zip(x_names, ordered_inc):
            cum += inc
            self.txt.insert("end", f" + {nm:<16} ΔR² = {inc:.6f}  cumulative R² = {cum:.6f}\n")
        self.txt.insert("end", "\n")

        if shapley is not None:
            self.txt.insert("end", "Shapley/LMG R² shares (unique + fair share of overlaps):\n")
            for nm, share in zip(x_names, shapley):
                pct = (share / r2_full * 100.0) if r2_full > 0 else float('nan')
                self.txt.insert("end", f" {nm:<18} share = {share:.6f} ({pct:.2f}% of R²)\n")
            self.txt.insert("end", "\n")

        self.txt.insert("end", "-" * 72 + "\n")
        self.lbl_status.configure(
            text=f"Aligned on n={n} rows. Imported groups: {len(self._groups)} (Y={y_name}; X={', '.join(x_names)})"
        )

    # ------------------------------ Plot window ------------------------------
    def _show_plot_window(self):
        # Guard: need a completed run
        if not getattr(self, "_last_fit", None):
            messagebox.showinfo("Plot", "Run the regression first, then show the plot.")
            return
        if not HAVE_MPL:
            messagebox.showwarning(
                "Plot",
                "Matplotlib is not available. Install it with: pip install matplotlib"
            )
            return

        y = self._last_fit["y"]  # type: ignore[index]
        yhat = self._last_fit["yhat"]  # type: ignore[index]
        r2 = self._last_fit["r2"]  # type: ignore[index]
        y_name = self._last_fit["y_name"]  # type: ignore[index]

        # Create a new Toplevel plot window
        win = tk.Toplevel(self)
        win.title(f"Plot – {y_name} (R² = {r2:.4f})")
        win.transient(self)
        win.grab_set()  # modal-like; remove if you prefer non-modal
        win.resizable(True, True)

        # Notebook with two tabs
        nb = ttk.Notebook(win)
        nb.pack(fill="both", expand=True)

        # --- Tab 1: Fit (Y vs ŷ) ---
        frame_fit = ttk.Frame(nb)
        nb.add(frame_fit, text="Fit")

        fig1 = Figure(figsize=(6, 4), dpi=100)
        ax1 = fig1.add_subplot(111)

        # Scatter of actual vs predicted
        ax1.scatter(yhat, y, s=20, alpha=0.8, color="#1f77b4", edgecolor="white", linewidth=0.5)
        # 45-degree line
        lo = float(np.nanmin([np.nanmin(yhat), np.nanmin(y)]))
        hi = float(np.nanmax([np.nanmax(yhat), np.nanmax(y)]))
        ax1.plot([lo, hi], [lo, hi], color="#d62728", linestyle="--", linewidth=1.2, label="Identity")

        ax1.set_title(f"Actual vs Predicted ({y_name})")
        ax1.set_xlabel("Predicted (ŷ)")
        ax1.set_ylabel("Actual (Y)")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.25)

        # R² annotation
        ax1.text(
            0.02, 0.96, f"R² = {r2:.4f}", transform=ax1.transAxes,
            ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )

        canvas1 = FigureCanvasTkAgg(fig1, master=frame_fit)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill="both", expand=True)

        # --- Tab 2: Residuals ---
        frame_res = ttk.Frame(nb)
        nb.add(frame_res, text="Residuals")

        resid = y - yhat
        fig2 = Figure(figsize=(6, 4), dpi=100)
        ax2 = fig2.add_subplot(111)

        ax2.plot(range(1, len(resid) + 1), resid, color="#2ca02c", linewidth=1.0)
        ax2.axhline(0.0, color="#999999", linestyle="--", linewidth=1.0)
        ax2.set_title("Residuals by Observation")
        ax2.set_xlabel("Observation")
        ax2.set_ylabel("Residual (Y - ŷ)")
        ax2.grid(True, alpha=0.25)

        canvas2 = FigureCanvasTkAgg(fig2, master=frame_res)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="both", expand=True)

        # Fit window size to content
        win.update_idletasks()
        try:
            win.minsize(win.winfo_reqwidth(), win.winfo_reqheight())
        except Exception:
            pass

# ------------------------------ Standalone demo ------------------------------
if __name__ == "__main__":
    # Minimal demo with synthetic data if run directly
    rng = np.random.default_rng(7)
    n = 120
    X1 = rng.normal(size=n)
    X2 = 0.5 * X1 + rng.normal(scale=0.7, size=n)  # correlated with X1
    X3 = rng.normal(size=n)
    y = 1.0 + 0.8 * X1 + 0.0 * X2 + 0.5 * X3 + rng.normal(scale=0.8, size=n)

    demo_groups = {
        "Y_synth": y,
        "X1": X1,
        "X2": X2,
        "X3": X3,
        "Noise": rng.normal(size=n)
    }

    def demo_get_groups():
        return demo_groups

    root = tk.Tk()
    root.title("Demo — Multiple regression")
    ttk.Button(
        root,
        text="Open multiple regression",
        command=lambda: MultipleRegressionWindow(root, get_groups_callback=demo_get_groups)
    ).pack(padx=10, pady=10)
    root.mainloop()
