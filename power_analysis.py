
# power_analysis.py
# -----------------------------------------------------------------------------
# Power analysis window for common statistical tests used in the app.
# - Independent two-sample t-test (equal/unequal allocation; 1- or 2-sided)
# - Paired/one-sample t-test
# - One-way ANOVA (Cohen's f)
# - Pearson correlation (Fisher-z approximation)
# - Simple linear regression (slope != 0; noncentral-t)
# - Mann–Whitney U (normal approximation via Cliff's δ / AUC)
#
# Uses statsmodels when available; otherwise falls back to SciPy-based
# analytic computations and numerical root finding where needed.
#
# Public entry point:
#   PowerAnalysisWindow(master, get_groups_callback)
# The optional get_groups_callback lets the main app prefill group stats from
# the current selection (category -> 1D numpy array).
# -----------------------------------------------------------------------------

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
import math
import numpy as np
from scipy import stats
from typing import Optional, Callable, Dict, Union

# Optional statsmodels (closed-form solvers for some tests)
try:
    from statsmodels.stats.power import (
        TTestIndPower, TTestPower, FTestAnovaPower
    )
    HAVE_SM = True
except Exception:
    HAVE_SM = False


# -----------------------------------------------------------------------------
# Utility: numerical root finding (bisection)
# -----------------------------------------------------------------------------
def _bisect_solve(func, lo, hi, target, tol=1e-6, max_iter=200):
    """Solve func(x) ~= target over [lo, hi] using bisection."""
    f_lo = func(lo) - target
    f_hi = func(hi) - target
    if math.isnan(f_lo) or math.isnan(f_hi):
        raise ValueError("Function returned NaN at bounds.")
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    if f_lo * f_hi > 0:
        # try expanding the bracket
        for _ in range(30):
            span = hi - lo
            lo = max(1e-9, lo - span)
            hi = hi + span
            f_lo = func(lo) - target
            f_hi = func(hi) - target
            if f_lo * f_hi <= 0:
                break
        else:
            raise ValueError("Bisection bounds do not bracket the target.")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = func(mid) - target
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


# -----------------------------------------------------------------------------
# Effect-size helpers
# -----------------------------------------------------------------------------
def cohen_d_indep_from_summary(mean1, sd1, n1, mean2, sd2, n2):
    """Cohen's d for independent groups using pooled SD."""
    s_p = math.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))
    if s_p <= 0:
        return 0.0
    return (mean1 - mean2) / s_p


def cohen_d_paired_from_summary(mean_diff, sd_diff):
    """Cohen's d for paired/one-sample from mean difference and SD of differences."""
    if sd_diff <= 0:
        return 0.0
    return mean_diff / sd_diff


def f_from_group_means(means: np.ndarray, s_within: float, ns: Union[np.ndarray, float]):
    """
    Cohen's f for one-way ANOVA.
    For balanced: f^2 = Var(means) / sigma_within^2
    For unbalanced: uses weighted mean & variance of means.
    """
    means = np.asarray(means, dtype=float)
    if np.isscalar(ns):
        ns = np.full_like(means, float(ns), dtype=float)
    else:
        ns = np.asarray(ns, dtype=float)
    mu_bar = np.average(means, weights=ns)
    num = np.sum(ns * (means - mu_bar) ** 2) / np.sum(ns)  # weighted variance of means
    if s_within <= 0:
        return 0.0
    return math.sqrt(num / (s_within ** 2))


# -----------------------------------------------------------------------------
# Power math (fallbacks when statsmodels is not present)
# -----------------------------------------------------------------------------
def power_t_independent(d, n1, n2, alpha=0.05, alternative="two-sided"):
    """
    Power for independent t-test with Cohen's d, sample sizes n1, n2.
    df = n1 + n2 - 2; ncp = d * sqrt(n1*n2/(n1+n2))
    """
    df = n1 + n2 - 2
    if df <= 0:
        return 0.0
    ncp = d * math.sqrt(n1 * n2 / (n1 + n2))
    if alternative == "two-sided":
        tcrit = stats.t.isf(alpha / 2, df)
        return float(stats.nct.sf(tcrit, df, ncp) + stats.nct.cdf(-tcrit, df, ncp))
    elif alternative == "larger":
        tcrit = stats.t.isf(alpha, df)
        return float(stats.nct.sf(tcrit, df, ncp))
    else:
        tcrit = stats.t.ppf(alpha, df)
        return float(stats.nct.cdf(tcrit, df, ncp))


def power_t_paired(d, n, alpha=0.05, alternative="two-sided"):
    """
    Power for paired/one-sample t with effect size d=mean_diff/sd_diff.
    df = n-1; ncp = d * sqrt(n)
    """
    df = n - 1
    if df <= 0:
        return 0.0
    ncp = d * math.sqrt(n)
    if alternative == "two-sided":
        tcrit = stats.t.isf(alpha / 2, df)
        return float(stats.nct.sf(tcrit, df, ncp) + stats.nct.cdf(-tcrit, df, ncp))
    elif alternative == "larger":
        tcrit = stats.t.isf(alpha, df)
        return float(stats.nct.sf(tcrit, df, ncp))
    else:
        tcrit = stats.t.ppf(alpha, df)
        return float(stats.nct.cdf(tcrit, df, ncp))


def power_anova_f(f, k, ns, alpha=0.05):
    """
    One-way ANOVA power using noncentral F with effect size f and groups k.
    If ns is scalar -> balanced; else array of group sizes (unbalanced),
    uses total N for lambda ≈ f^2 * N (common approximation).
    """
    if np.isscalar(ns):
        N = k * int(ns)
        df1 = k - 1
        df2 = N - k
    else:
        ns = np.asarray(ns, dtype=float)
        N = int(np.sum(ns))
        df1 = k - 1
        df2 = N - k
    if df1 <= 0 or df2 <= 0:
        return 0.0
    lam = (f ** 2) * N
    fcrit = stats.f.isf(alpha, df1, df2)
    return float(stats.ncf.sf(fcrit, df1, df2, lam))


def power_correlation(n, r_alt, alpha=0.05, alternative="two-sided"):
    """
    Power for H0: rho=0 using Fisher-z approximation.
    Z = atanh(r) * sqrt(n-3) ~ Normal(mean=atanh(r_alt)*sqrt(n-3), sd=1)
    """
    if n < 4:
        return 0.0
    mu = np.arctanh(np.clip(r_alt, -0.999999, 0.999999)) * math.sqrt(n - 3.0)
    if alternative == "two-sided":
        zcrit = stats.norm.isf(alpha / 2)
        return float(stats.norm.sf(zcrit - mu) + stats.norm.cdf(-zcrit - mu))
    elif alternative == "larger":
        zcrit = stats.norm.isf(alpha)
        return float(stats.norm.sf(zcrit - mu))
    else:
        zcrit = stats.norm.isf(alpha)
        return float(stats.norm.cdf(-zcrit - mu))


def power_regression_slope(n, r_alt, alpha=0.05, alternative="two-sided"):
    """
    Power for testing slope != 0 in simple linear regression.
    Equivalent to testing correlation r != 0 via noncentral t:
    df = n-2; ncp = r * sqrt(df / (1 - r^2))
    """
    df = n - 2
    if df <= 0 or abs(r_alt) >= 1.0:
        return 0.0
    denom = max(1e-12, (1 - r_alt ** 2))
    ncp = r_alt * math.sqrt(df / denom)
    if alternative == "two-sided":
        tcrit = stats.t.isf(alpha / 2, df)
        return float(stats.nct.sf(tcrit, df, ncp) + stats.nct.cdf(-tcrit, df, ncp))
    elif alternative == "larger":
        tcrit = stats.t.isf(alpha, df)
        return float(stats.nct.sf(tcrit, df, ncp))
    else:
        tcrit = stats.t.ppf(alpha, df)
        return float(stats.nct.cdf(tcrit, df, ncp))


def power_mann_whitney(n1, n2, delta_cliff=None, auc=None, alpha=0.05, alternative="two-sided"):
    """
    Approximate Mann–Whitney U power via normal approximation.
    Relationship (no ties):
    AUC = P(X > Y) = (δ + 1)/2
    Cliff's delta δ = 2*AUC - 1
    Under H0: E[U] = n1*n2/2; Var[U] = n1*n2*(n1+n2+1)/12
    Under H1: E[U] ≈ n1*n2*AUC.
    """
    if auc is None and delta_cliff is None:
        return 0.0
    if auc is None:
        auc = 0.5 * (delta_cliff + 1.0)
    auc = float(np.clip(auc, 1e-6, 1 - 1e-6))
    mu0 = n1 * n2 / 2.0
    var0 = n1 * n2 * (n1 + n2 + 1) / 12.0
    if var0 <= 0:
        return 0.0
    sd0 = math.sqrt(var0)
    mu1 = n1 * n2 * auc
    z_alt_mean = (mu1 - mu0) / sd0  # mean of Z under H1 (sd ≈ 1)
    if alternative == "two-sided":
        zcrit = stats.norm.isf(alpha / 2)
        return float(stats.norm.sf(zcrit - z_alt_mean) + stats.norm.cdf(-zcrit - z_alt_mean))
    elif alternative == "larger":
        zcrit = stats.norm.isf(alpha)
        return float(stats.norm.sf(zcrit - z_alt_mean))
    else:
        zcrit = stats.norm.isf(alpha)
        return float(stats.norm.cdf(-zcrit - z_alt_mean))


# -----------------------------------------------------------------------------
# Tkinter UI
# -----------------------------------------------------------------------------
class PowerAnalysisWindow(tk.Toplevel):
    PADX = 10
    PADY = 6
    WRAP = 260  # wraplength for labels placed right of widgets

    def __init__(
        self,
        master: Optional[tk.Misc] = None,
        get_groups_callback: Optional[Callable[[], Dict[str, np.ndarray]]] = None,
        title: str = "Power analysis"
    ):
        super().__init__(master)
        self.title(title)
        self.transient(master)
        self.grab_set()

        # --- FULL SCREEN (best effort) ---
        try:
            self.state("zoomed")  # Windows/macOS
        except Exception:
            try:
                self.attributes("-zoomed", True)  # Many Linux builds
            except Exception:
                sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
                self.geometry(f"{sw}x{sh}+0+0")

        self.get_groups_callback = get_groups_callback

        info = "Statsmodels: " + ("available" if HAVE_SM else "not installed – using SciPy formulas")
        ttk.Label(self, text=info, foreground="#666").pack(anchor="w", padx=self.PADX, pady=(self.PADY, 0))

        # Notebook (tabs)
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=self.PADX, pady=self.PADY)

        # Keyboard shortcuts: minimize window (Alt+M / Alt+m)
        self.bind_all("<Alt-m>", lambda e: self._minimize_self())
        self.bind_all("<Alt-M>", lambda e: self._minimize_self())

        # Build tabs
        self._build_ttest_ind_tab()
        self._build_ttest_paired_tab()
        self._build_anova_tab()
        self._build_corr_tab()
        self._build_regression_tab()
        self._build_mwu_tab()

        self._build_bottom_bar()

    # ---------- Tiny helpers to keep layout consistent ----------
    def _entry_label(self, parent, row, col, var, width, label_text, *,
                     sticky="w", padx=(4, 18), entry_padx=(0, 0)):
        """Place Entry then its label to the right; return next free column."""
        e = ttk.Entry(parent, textvariable=var, width=width)
        e.grid(row=row, column=col, sticky=sticky, padx=entry_padx, pady=(0, 0))
        ttk.Label(parent, text=label_text, wraplength=self.WRAP).grid(
            row=row, column=col + 1, sticky="w", padx=padx
        )
        return col + 2

    def _combo_label(self, parent, row, col, var, values, width, label_text, *,
                     sticky="w", padx=(4, 18), combo_padx=(0, 0)):
        """Place Combobox then its label to the right; return next free column."""
        cb = ttk.Combobox(parent, textvariable=var, state="readonly", values=values, width=width)
        cb.grid(row=row, column=col, sticky=sticky, padx=combo_padx, pady=(0, 0))
        ttk.Label(parent, text=label_text, wraplength=self.WRAP).grid(
            row=row, column=col + 1, sticky="w", padx=padx
        )
        return col + 2

    def _result_box(self, parent):
        txt = tk.Text(parent, height=12, wrap="word")
        txt.configure(font=("Consolas", 10))
        txt.grid(sticky="nsew", padx=self.PADX, pady=(self.PADY, 0))
        return txt

    def _line(self, parent):
        sep = ttk.Separator(parent, orient="horizontal")
        sep.grid(sticky="ew", padx=self.PADX, pady=(self.PADY, self.PADY))

    def _get_groups(self):
        if self.get_groups_callback:
            try:
                data = self.get_groups_callback() or {}
                out = {k: np.asarray(v, dtype=float) for k, v in data.items()}
                out = {k: v[np.isfinite(v)] for k, v in out.items()}
                return {k: v for k, v in out.items() if v.size > 0}
            except Exception as ex:
                messagebox.showwarning("Data", f"Could not fetch groups from app: {ex}")
        return {}

    # ---------- Window control helpers ----------
    def _minimize_self(self):
        """
        Minimize (iconify) this window. Works even when zoomed.
        """
        try:
            self.iconify()
        except Exception:
            try:
                self.withdraw()
                self.after(50, self.deiconify)
            except Exception:
                pass

    # ---------- Scrollable tab scaffold ----------
    def _make_scrollable_tab(self, tab_parent):
        """
        Returns (outer_frame, body) where body is a scrollable frame.
        """
        outer = ttk.Frame(tab_parent)
        outer.pack(fill="both", expand=True)

        # Scrollable body
        sc = ttk.Frame(outer)
        sc.pack(fill="both", expand=True)
        canvas = tk.Canvas(sc, highlightthickness=0)
        vsb = ttk.Scrollbar(sc, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        body = ttk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=body, anchor="nw")

        # Make the inner frame stretchy
        body.grid_columnconfigure(0, weight=1)

        def _on_body_config(_):
            canvas.configure(scrollregion=canvas.bbox("all"))
        body.bind("<Configure>", _on_body_config)

        def _on_canvas_config(e):
            try:
                canvas.itemconfigure(win_id, width=e.width)
            except Exception:
                pass
        canvas.bind("<Configure>", _on_canvas_config)

        # Mousewheel scrolling
        def _wheel_win(event):
            if event.delta:
                canvas.yview_scroll(int(-event.delta / 120), "units")

        def _wheel_x11(event):
            if event.num == 4:
                canvas.yview_scroll(-3, "units")
            elif event.num == 5:
                canvas.yview_scroll(3, "units")

        canvas.bind("<Enter>", lambda _e: (
            canvas.bind_all("<MouseWheel>", _wheel_win),
            canvas.bind_all("<Button-4>", _wheel_x11),
            canvas.bind_all("<Button-5>", _wheel_x11),
        ))
        canvas.bind("<Leave>", lambda _e: (
            canvas.unbind_all("<MouseWheel>"),
            canvas.unbind_all("<Button-4>"),
            canvas.unbind_all("<Button-5>"),
        ))

        return outer, body

    # ---------- Shared section builders ----------
    def _settings_frame(self, parent):
        lf = ttk.LabelFrame(parent, text="Settings")
        lf.grid(sticky="ew", padx=self.PADX, pady=(self.PADY, 0))
        for c in range(6):
            lf.grid_columnconfigure(c, weight=0)
        # Alpha / Power / Alternative as control->label pairs
        alpha = tk.StringVar(value="0.05")
        power = tk.StringVar(value="0.8")
        alt = tk.StringVar(value="two-sided")

        col = 0
        col = self._entry_label(lf, 0, col, alpha, 8, "α (Type I error)")
        col = self._entry_label(lf, 0, col, power, 8, "Desired power (1-β)")
        _ = self._combo_label(lf, 0, col, alt, ("two-sided", "larger", "smaller"), 11, "Alternative")

        return alpha, power, alt, lf

    def _section(self, parent, title):
        lf = ttk.LabelFrame(parent, text=title)
        lf.grid(sticky="ew", padx=self.PADX, pady=(self.PADY, 0))
        return lf

    # ---------- Tabs ----------
    def _build_ttest_ind_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Two‑sample t‑test")

        _outer, body = self._make_scrollable_tab(tab)

        # SETTINGS
        alpha, power, alt, settings = self._settings_frame(body)

        # MODE
        mode_bar = ttk.Frame(body)
        mode_bar.grid(sticky="ew", padx=self.PADX, pady=(self.PADY, 0))
        mode = tk.StringVar(value="posthoc")
        ttk.Label(mode_bar, text="Mode").pack(side="left", padx=(0, 6))
        ttk.Radiobutton(mode_bar, text="Post hoc (power from n, d)", variable=mode, value="posthoc").pack(side="left")
        ttk.Radiobutton(mode_bar, text="A priori (n from power, d)", variable=mode, value="apriori").pack(side="left", padx=(10, 0))
        ttk.Radiobutton(mode_bar, text="Sensitivity (d from n, power)", variable=mode, value="sensitivity").pack(side="left", padx=(10, 0))

        # PARAMETERS
        params = self._section(body, "Parameters")
        for c in range(8):
            params.grid_columnconfigure(c, weight=0)

        d_var = tk.StringVar(value="0.5")
        n1_var = tk.StringVar(value="20")
        n2_var = tk.StringVar(value="20")
        ratio_var = tk.StringVar(value="1.0")

        col = 0
        col = self._entry_label(params, 0, col, d_var, 10, "Effect size d (Cohen)")
        col = self._entry_label(params, 0, col, n1_var, 8, "n1")
        col = self._entry_label(params, 0, col, n2_var, 8, "n2")
        _ = self._entry_label(params, 0, col, ratio_var, 8, "Allocation ratio n2/n1 (for a priori)")

        # SUMMARY HELPERS
        helpers = self._section(body, "Summary → d (optional)")
        for c in range(12):
            helpers.grid_columnconfigure(c, weight=0)

        m1, s1, n1s = tk.StringVar(), tk.StringVar(), tk.StringVar()
        m2, s2, n2s = tk.StringVar(), tk.StringVar(), tk.StringVar()

        col = 0
        col = self._entry_label(helpers, 0, col, m1, 10, "Mean1")
        col = self._entry_label(helpers, 0, col, s1, 10, "SD1")
        col = self._entry_label(helpers, 0, col, n1s, 8, "n1")
        col = self._entry_label(helpers, 0, col, m2, 10, "Mean2")
        col = self._entry_label(helpers, 0, col, s2, 10, "SD2")
        _ = self._entry_label(helpers, 0, col, n2s, 8, "n2")

        actions = ttk.Frame(body); actions.grid(sticky="ew", padx=self.PADX, pady=(self.PADY, 0))
        imported_label = ttk.Label(actions, text="", foreground="#555")
        imported_label.pack(side="left")

        def _prefill_from_groups():
            groups = self._get_groups()
            if len(groups) < 2:
                messagebox.showinfo("Prefill", "Need at least 2 groups in the current selection.")
                return
            (g1, g2) = list(groups.keys())[:2]
            a, b = groups[g1], groups[g2]
            m1.set(f"{np.nanmean(a):.6g}"); s1.set(f"{np.nanstd(a, ddof=1):.6g}"); n1s.set(str(a.size))
            m2.set(f"{np.nanmean(b):.6g}"); s2.set(f"{np.nanstd(b, ddof=1):.6g}"); n2s.set(str(b.size))
            imported_label.config(text=f"Imported groups: {g1} (n={a.size}), {g2} (n={b.size})")

        def _compute_d_from_summary():
            try:
                vv = [float(x) for x in (m1.get(), s1.get(), n1s.get(), m2.get(), s2.get(), n2s.get())]
                d_est = cohen_d_indep_from_summary(vv[0], vv[1], int(vv[2]), vv[3], vv[4], int(vv[5]))
                d_var.set(f"{d_est:.6g}")
            except Exception as ex:
                messagebox.showerror("d from summary", f"Error: {ex}")

        ttk.Button(actions, text="Prefill from current groups", command=_prefill_from_groups).pack(side="right", padx=(8,0))
        ttk.Button(actions, text="Compute d from summary", command=_compute_d_from_summary).pack(side="right")

        # RUN just above RESULTS
        run_bar = ttk.Frame(body); run_bar.grid(sticky="ew", padx=self.PADX, pady=(self.PADY, 0))
        ttk.Button(run_bar, text="Run",
                   command=lambda: self._run_ttest_ind(alpha, power, alt, mode, d_var, n1_var, n2_var, ratio_var, res)
                   ).pack(side="right")

        # RESULTS
        results = self._section(body, "Results")
        results.grid_columnconfigure(0, weight=1)
        res = self._result_box(results)

    def _build_ttest_paired_tab(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="Paired / One‑sample t‑test")
        _outer, body = self._make_scrollable_tab(tab)

        alpha, power, alt, _settings = self._settings_frame(body)

        mode_bar = ttk.Frame(body); mode_bar.grid(sticky="ew", padx=self.PADX, pady=(self.PADY, 0))
        mode = tk.StringVar(value="posthoc")
        ttk.Label(mode_bar, text="Mode").pack(side="left", padx=(0, 6))
        ttk.Radiobutton(mode_bar, text="Post hoc", variable=mode, value="posthoc").pack(side="left")
        ttk.Radiobutton(mode_bar, text="A priori", variable=mode, value="apriori").pack(side="left", padx=(10, 0))
        ttk.Radiobutton(mode_bar, text="Sensitivity", variable=mode, value="sensitivity").pack(side="left", padx=(10, 0))

        params = self._section(body, "Parameters")
        d_var = tk.StringVar(value="0.5")
        n_var = tk.StringVar(value="20")
        col = 0
        col = self._entry_label(params, 0, col, d_var, 10, "Effect size d = mean_diff / sd_diff")
        _ = self._entry_label(params, 0, col, n_var, 8, "n (pairs)")

        helpers = self._section(body, "Summary → d (optional)")
        md = tk.StringVar(); sd = tk.StringVar()
        col = 0
        col = self._entry_label(helpers, 0, col, md, 10, "mean_diff")
        _ = self._entry_label(helpers, 0, col, sd, 10, "sd_diff")

        ttk.Button(helpers, text="Compute d",
                   command=lambda: self._compute_d_paired(md, sd, d_var)
                   ).grid(row=0, column=4, padx=(self.PADX//2, 0))

        # RUN just above RESULTS
        run_bar = ttk.Frame(body); run_bar.grid(sticky="ew", padx=self.PADX, pady=(self.PADY, 0))
        ttk.Button(run_bar, text="Run",
                   command=lambda: self._run_ttest_paired(alpha, power, alt, mode, d_var, n_var, res)
                   ).pack(side="right")

        results = self._section(body, "Results"); results.grid_columnconfigure(0, weight=1)
        res = self._result_box(results)

    def _compute_d_paired(self, md, sd, d_var):
        try:
            d_est = cohen_d_paired_from_summary(float(md.get()), float(sd.get()))
            d_var.set(f"{d_est:.6g}")
        except Exception as ex:
            messagebox.showerror("d from summary", f"Error: {ex}")

    def _build_anova_tab(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="One‑way ANOVA")
        _outer, body = self._make_scrollable_tab(tab)

        alpha, power, _alt, _settings = self._settings_frame(body)

        mode_bar = ttk.Frame(body); mode_bar.grid(sticky="ew", padx=self.PADX, pady=(self.PADY, 0))
        mode = tk.StringVar(value="posthoc")
        ttk.Label(mode_bar, text="Mode").pack(side="left", padx=(0, 6))
        ttk.Radiobutton(mode_bar, text="Post hoc", variable=mode, value="posthoc").pack(side="left")
        ttk.Radiobutton(mode_bar, text="A priori", variable=mode, value="apriori").pack(side="left", padx=(10, 0))
        ttk.Radiobutton(mode_bar, text="Sensitivity", variable=mode, value="sensitivity").pack(side="left", padx=(10, 0))

        params = self._section(body, "Parameters")
        k_var = tk.StringVar(value="3")
        f_var = tk.StringVar(value="0.25")
        n_any = tk.StringVar(value="20")

        col = 0
        col = self._entry_label(params, 0, col, k_var, 8, "Groups (k)")
        col = self._entry_label(params, 0, col, f_var, 10, "Effect size f (Cohen)")
        _ = self._entry_label(params, 0, col, n_any, 22, "n per group (balanced) or list e.g. 20,18,22")

        helpers = self._section(body, "Summary → f (optional)")
        means_str = tk.StringVar(); s_within = tk.StringVar()

        col = 0
        col = self._entry_label(helpers, 0, col, means_str, 28, "Means (comma-separated)")
        _ = self._entry_label(helpers, 0, col, s_within, 10, "SD_within")

        ttk.Button(helpers, text="Compute f",
                   command=lambda: self._compute_f_from_means(means_str, s_within, n_any, f_var, k_var)
                   ).grid(row=0, column=4, padx=(self.PADX//2, 0))

        # --- Prefill from current groups for ANOVA ---
        imported_label = ttk.Label(helpers, text="", foreground="#555")
        imported_label.grid(row=1, column=1, columnspan=4, sticky="w", padx=(8,0), pady=(8,0))

        def _prefill_anova():
            groups = self._get_groups()
            if len(groups) < 2:
                messagebox.showinfo("Prefill", "Need at least 2 groups in the current selection.")
                return

            names = list(groups.keys())
            arrs  = [groups[nm] for nm in names]

            ns_list    = [a.size for a in arrs]
            means_list = [float(np.nanmean(a)) for a in arrs]
            sds_list   = [float(np.nanstd(a, ddof=1)) for a in arrs]

            N = sum(ns_list); k = len(arrs)
            denom = (N - k)
            if denom <= 0:
                s_within_pooled = 0.0
            else:
                num = sum((ns_list[i] - 1) * (sds_list[i] ** 2) for i in range(k))
                s_within_pooled = math.sqrt(num / denom)

            means_str.set(",".join(f"{m:.6g}" for m in means_list))
            s_within.set(f"{s_within_pooled:.6g}")
            n_any.set(",".join(str(n) for n in ns_list))
            k_var.set(str(k))

            try:
                f_est = f_from_group_means(np.array(means_list, dtype=float),
                                           s_within_pooled,
                                           np.array(ns_list, dtype=float))
                f_var.set(f"{f_est:.6g}")
            except Exception:
                pass

            imported_label.config(
                text="Imported groups: " + ", ".join(f"{nm} (n={nn})" for nm, nn in zip(names, ns_list))
            )

        ttk.Button(helpers, text="Prefill from current groups", command=_prefill_anova)\
            .grid(row=1, column=0, sticky="w", pady=(8,0))
        # --- end prefill ---

        # RUN just above RESULTS
        run_bar = ttk.Frame(body); run_bar.grid(sticky="ew", padx=self.PADX, pady=(self.PADY, 0))
        ttk.Button(run_bar, text="Run",
                   command=lambda: self._run_anova(alpha, power, mode, k_var, f_var, n_any, res)
                   ).pack(side="right")

        results = self._section(body, "Results"); results.grid_columnconfigure(0, weight=1)
        res = self._result_box(results)

    def _compute_f_from_means(self, means_str, s_within, n_any, f_var, k_var):
        try:
            means = np.array([float(x) for x in means_str.get().replace(";", ",").split(",") if x.strip()])
            sw = float(s_within.get())
            nval = n_any.get().strip()
            if "," in nval:
                ns = np.array([float(x) for x in nval.replace(";", ",").split(",") if x.strip()], dtype=float)
            else:
                ns = float(nval)
            f = f_from_group_means(means, sw, ns)
            f_var.set(f"{f:.6g}")
            k_var.set(str(len(means)))
        except Exception as ex:
            messagebox.showerror("f from means", f"Error: {ex}")

    def _build_corr_tab(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="Correlation (Pearson)")
        _outer, body = self._make_scrollable_tab(tab)

        alpha, power, alt, _settings = self._settings_frame(body)

        mode_bar = ttk.Frame(body); mode_bar.grid(sticky="ew", padx=self.PADX, pady=(self.PADY, 0))
        mode = tk.StringVar(value="posthoc")
        ttk.Label(mode_bar, text="Mode").pack(side="left", padx=(0, 6))
        ttk.Radiobutton(mode_bar, text="Post hoc", variable=mode, value="posthoc").pack(side="left")
        ttk.Radiobutton(mode_bar, text="A priori", variable=mode, value="apriori").pack(side="left", padx=(10,0))
        ttk.Radiobutton(mode_bar, text="Sensitivity", variable=mode, value="sensitivity").pack(side="left", padx=(10,0))

        params = self._section(body, "Parameters")
        r_var = tk.StringVar(value="0.3")
        n_var = tk.StringVar(value="40")

        col = 0
        col = self._entry_label(params, 0, col, r_var, 10, "Correlation under H1 (r)")
        _ = self._entry_label(params, 0, col, n_var, 10, "n")

        # RUN just above RESULTS
        run_bar = ttk.Frame(body); run_bar.grid(sticky="ew", padx=self.PADX, pady=(self.PADY, 0))
        ttk.Button(run_bar, text="Run",
                   command=lambda: self._run_corr(alpha, power, alt, mode, r_var, n_var, res)
                   ).pack(side="right")

        results = self._section(body, "Results"); results.grid_columnconfigure(0, weight=1)
        res = self._result_box(results)

    def _build_regression_tab(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="Linear regression (slope)")
        _outer, body = self._make_scrollable_tab(tab)

        alpha, power, alt, _settings = self._settings_frame(body)

        mode_bar = ttk.Frame(body); mode_bar.grid(sticky="ew", padx=self.PADX, pady=(self.PADY, 0))
        mode = tk.StringVar(value="posthoc")
        ttk.Label(mode_bar, text="Mode").pack(side="left", padx=(0, 6))
        ttk.Radiobutton(mode_bar, text="Post hoc", variable=mode, value="posthoc").pack(side="left")
        ttk.Radiobutton(mode_bar, text="A priori", variable=mode, value="apriori").pack(side="left", padx=(10,0))
        ttk.Radiobutton(mode_bar, text="Sensitivity", variable=mode, value="sensitivity").pack(side="left", padx=(10,0))

        params = self._section(body, "Parameters")
        r_var = tk.StringVar(value="0.3"); n_var = tk.StringVar(value="40")

        col = 0
        col = self._entry_label(params, 0, col, r_var, 10, "Correlation equivalent r (relates to slope)")
        _ = self._entry_label(params, 0, col, n_var, 10, "n")

        # RUN just above RESULTS
        run_bar = ttk.Frame(body); run_bar.grid(sticky="ew", padx=self.PADX, pady=(self.PADY, 0))
        ttk.Button(run_bar, text="Run",
                   command=lambda: self._run_regression(alpha, power, alt, mode, r_var, n_var, res)
                   ).pack(side="right")

        results = self._section(body, "Results"); results.grid_columnconfigure(0, weight=1)
        res = self._result_box(results)

    def _build_mwu_tab(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="Mann–Whitney U (approx)")
        _outer, body = self._make_scrollable_tab(tab)

        alpha, power, alt, _settings = self._settings_frame(body)

        mode_bar = ttk.Frame(body); mode_bar.grid(sticky="ew", padx=self.PADX, pady=(self.PADY, 0))
        mode = tk.StringVar(value="posthoc")
        ttk.Label(mode_bar, text="Mode").pack(side="left", padx=(0, 6))
        ttk.Radiobutton(mode_bar, text="Post hoc", variable=mode, value="posthoc").pack(side="left")
        ttk.Radiobutton(mode_bar, text="A priori", variable=mode, value="apriori").pack(side="left", padx=(10,0))
        ttk.Radiobutton(mode_bar, text="Sensitivity", variable=mode, value="sensitivity").pack(side="left", padx=(10,0))

        params = self._section(body, "Parameters")
        delta_var = tk.StringVar(value="")
        auc_var = tk.StringVar(value="0.65")
        n1_var = tk.StringVar(value="20")
        n2_var = tk.StringVar(value="20")

        col = 0
        col = self._entry_label(params, 0, col, delta_var, 10, "Cliff's δ (−1..1)")
        col = self._entry_label(params, 0, col, auc_var, 10, "AUC (0..1)")
        col = self._entry_label(params, 0, col, n1_var, 8, "n1")
        _ = self._entry_label(params, 0, col, n2_var, 8, "n2")

        helpers = self._section(body, "Prefill from data (optional)")
        imported_label = ttk.Label(helpers, text="", foreground="#555")
        imported_label.grid(row=0, column=1, sticky="w", padx=(8,0))

        ttk.Button(helpers, text="Prefill from current groups + estimate δ",
                   command=lambda: self._prefill_estimate_delta(n1_var, n2_var, delta_var, auc_var, imported_label)
                   ).grid(row=0, column=0, padx=(0, 0), pady=(0, 0), sticky="w")

        # RUN just above RESULTS
        run_bar = ttk.Frame(body); run_bar.grid(sticky="ew", padx=self.PADX, pady=(self.PADY, 0))
        ttk.Button(run_bar, text="Run",
                   command=lambda: self._run_mwu(alpha, power, alt, mode, delta_var, auc_var, n1_var, n2_var, res)
                   ).pack(side="right")

        results = self._section(body, "Results"); results.grid_columnconfigure(0, weight=1)
        res = self._result_box(results)

    def _prefill_estimate_delta(self, n1_var, n2_var, delta_var, auc_var, imported_label):
        groups = self._get_groups()
        if len(groups) < 2:
            messagebox.showinfo("Prefill", "Need at least 2 groups in the current selection.")
            return
        (g1, g2) = list(groups.keys())[:2]
        x, y = groups[g1], groups[g2]
        n1_var.set(str(x.size)); n2_var.set(str(y.size))
        allv = np.concatenate([x, y])
        ranks = stats.rankdata(allv, method="average")
        rx = ranks[:x.size]
        A = (np.sum(rx)/x.size - (x.size + 1)/2.0)/y.size + 0.5
        A = float(np.clip(A, 1e-6, 1 - 1e-6))
        delta = 2*A - 1
        delta_var.set(f"{delta:.6g}")
        auc_var.set(f"{A:.6g}")
        imported_label.config(text=f"Imported groups: {g1} (n={x.size}), {g2} (n={y.size})")

    # ---------- runners ----------
    def _run_ttest_ind(self, alpha, power, alt, mode, d_var, n1_var, n2_var, ratio_var, resbox):
        try:
            a = float(alpha.get()); tgt = float(power.get()); alternative = alt.get()
            d = float(d_var.get())
            n1 = int(float(n1_var.get())); n2 = int(float(n2_var.get()))
            ratio = float(ratio_var.get())
            if n1 < 2: n1 = 2
            if n2 < 2: n2 = 2
        except Exception as ex:
            messagebox.showerror("Input", f"Invalid inputs: {ex}"); return

        resbox.delete("1.0", "end")
        if mode.get() == "posthoc":
            if HAVE_SM:
                try:
                    pw = TTestIndPower().power(effect_size=d, nobs1=n1, alpha=a, ratio=(n2/n1), alternative=alternative)
                except Exception:
                    pw = power_t_independent(d, n1, n2, alpha=a, alternative=alternative)
            else:
                pw = power_t_independent(d, n1, n2, alpha=a, alternative=alternative)
            resbox.insert("end", f"Post hoc power (two-sample t): {pw:.4f}\n")
        elif mode.get() == "apriori":
            if HAVE_SM:
                try:
                    n1_req = TTestIndPower().solve_power(effect_size=d, power=tgt, alpha=a, ratio=ratio, alternative=alternative)
                    n1_req = math.ceil(n1_req)
                    n2_req = math.ceil(ratio * n1_req)
                except Exception:
                    def f(n1cand):
                        n1c = max(2, int(round(n1cand)))
                        n2c = max(2, int(round(ratio * n1c)))
                        return power_t_independent(d, n1c, n2c, alpha=a, alternative=alternative)
                    n1_req = math.ceil(_bisect_solve(f, 2, 20000, tgt))
                    n2_req = math.ceil(ratio * n1_req)
            else:
                def f(n1cand):
                    n1c = max(2, int(round(n1cand)))
                    n2c = max(2, int(round(ratio * n1c)))
                    return power_t_independent(d, n1c, n2c, alpha=a, alternative=alternative)
                n1_req = math.ceil(_bisect_solve(f, 2, 20000, tgt))
                n2_req = math.ceil(ratio * n1_req)
            resbox.insert("end", f"A priori sample size (per group): n1={n1_req}, n2={n2_req}\n")
        else:  # sensitivity
            def g(dcand):
                return power_t_independent(dcand, n1, n2, alpha=a, alternative=alternative)
            try:
                d_req = _bisect_solve(g, 1e-6, 5.0, tgt)
            except Exception:
                d_req = float("nan")
            resbox.insert("end", f"Sensitivity effect size d to reach power={tgt:.3f}: d≈{d_req:.4f}\n")

    def _run_ttest_paired(self, alpha, power, alt, mode, d_var, n_var, resbox):
        try:
            a = float(alpha.get()); tgt = float(power.get()); alternative = alt.get()
            d = float(d_var.get()); n = int(float(n_var.get()))
            if n < 2: n = 2
        except Exception as ex:
            messagebox.showerror("Input", f"Invalid inputs: {ex}"); return

        resbox.delete("1.0", "end")
        if mode.get() == "posthoc":
            if HAVE_SM:
                try:
                    pw = TTestPower().power(effect_size=d, nobs=n, alpha=a, alternative=alternative)
                except Exception:
                    pw = power_t_paired(d, n, alpha=a, alternative=alternative)
            else:
                pw = power_t_paired(d, n, alpha=a, alternative=alternative)
            resbox.insert("end", f"Post hoc power (paired/one-sample): {pw:.4f}\n")
        elif mode.get() == "apriori":
            if HAVE_SM:
                try:
                    n_req = TTestPower().solve_power(effect_size=d, power=tgt, alpha=a, alternative=alternative)
                    n_req = math.ceil(n_req)
                except Exception:
                    def f(ncand):
                        return power_t_paired(d, max(2, int(round(ncand))), alpha=a, alternative=alternative)
                    n_req = math.ceil(_bisect_solve(f, 2, 20000, tgt))
            else:
                def f(ncand):
                    return power_t_paired(d, max(2, int(round(ncand))), alpha=a, alternative=alternative)
                n_req = math.ceil(_bisect_solve(f, 2, 20000, tgt))
            resbox.insert("end", f"A priori sample size (pairs): n={n_req}\n")
        else:
            def g(dcand):
                return power_t_paired(dcand, n, alpha=a, alternative=alternative)
            try:
                d_req = _bisect_solve(g, 1e-6, 5.0, tgt)
            except Exception:
                d_req = float("nan")
            resbox.insert("end", f"Sensitivity d to reach power={tgt:.3f}: d≈{d_req:.4f}\n")

    def _parse_ns(self, s: str, k_expected: Optional[int] = None):
        s = s.strip()
        if "," in s:
            ns = [int(float(x)) for x in s.replace(";", ",").split(",") if x.strip()]
            if k_expected is not None and len(ns) != k_expected:
                raise ValueError("Number of n values must match k.")
            return np.array(ns, dtype=int)
        else:
            n = int(float(s))
            return n

    def _run_anova(self, alpha, power, mode, k_var, f_var, n_any, resbox):
        try:
            a = float(alpha.get()); tgt = float(power.get())
            k = int(float(k_var.get()))
            f = float(f_var.get())
            ns = self._parse_ns(n_any.get(), None)
        except Exception as ex:
            messagebox.showerror("Input", f"Invalid inputs: {ex}"); return

        resbox.delete("1.0", "end")
        if mode.get() == "posthoc":
            if HAVE_SM and not isinstance(ns, np.ndarray):
                try:
                    pw = FTestAnovaPower().power(effect_size=f, k_groups=k, nobs=ns, alpha=a)
                except Exception:
                    pw = power_anova_f(f, k, ns, alpha=a)
            else:
                pw = power_anova_f(f, k, ns, alpha=a)
            resbox.insert("end", f"Post hoc power (ANOVA): {pw:.4f}\n")
        elif mode.get() == "apriori":
            if HAVE_SM:
                try:
                    n_req = FTestAnovaPower().solve_power(effect_size=f, k_groups=k, alpha=a, power=tgt)
                    n_req = math.ceil(n_req)
                    resbox.insert("end", f"A priori sample size per group (balanced): n={n_req}\n")
                    return
                except Exception:
                    pass
            def f_pwr(n_per_group):
                n_grp = max(2, int(round(n_per_group)))
                return power_anova_f(f, k, n_grp, alpha=a)
            n_req = math.ceil(_bisect_solve(f_pwr, 2, 50000, tgt))
            resbox.insert("end", f"A priori sample size per group (balanced): n={n_req}\n")
        else:
            def f_obj(fcand):
                return power_anova_f(fcand, k, ns, alpha=a)
            try:
                f_req = _bisect_solve(f_obj, 1e-6, 5.0, tgt)
            except Exception:
                f_req = float("nan")
            resbox.insert("end", f"Sensitivity effect size f to reach power={tgt:.3f}: f≈{f_req:.4f}\n")

    def _run_corr(self, alpha, power, alt, mode, r_var, n_var, resbox):
        try:
            a = float(alpha.get()); tgt = float(power.get()); alternative = alt.get()
            r = float(r_var.get()); n = int(float(n_var.get()))
        except Exception as ex:
            messagebox.showerror("Input", f"Invalid inputs: {ex}"); return

        resbox.delete("1.0", "end")
        if mode.get() == "posthoc":
            pw = power_correlation(n=n, r_alt=r, alpha=a, alternative=alternative)
            resbox.insert("end", f"Post hoc power (correlation): {pw:.4f}\n")
        elif mode.get() == "apriori":
            def f(ncand):
                return power_correlation(n=int(round(ncand)), r_alt=r, alpha=a, alternative=alternative)
            n_req = math.ceil(_bisect_solve(f, 4, 200000, tgt))
            resbox.insert("end", f"A priori sample size (n): {n_req}\n")
        else:
            def g(rcand):
                return power_correlation(n=n, r_alt=rcand, alpha=a, alternative=alternative)
            try:
                r_req = _bisect_solve(g, 1e-6, 0.999, tgt)
            except Exception:
                r_req = float("nan")
            resbox.insert("end", f"Sensitivity r to reach power={tgt:.3f}: r≈{r_req:.4f}\n")

    def _run_regression(self, alpha, power, alt, mode, r_var, n_var, resbox):
        try:
            a = float(alpha.get()); tgt = float(power.get()); alternative = alt.get()
            r = float(r_var.get()); n = int(float(n_var.get()))
        except Exception as ex:
            messagebox.showerror("Input", f"Invalid inputs: {ex}"); return

        resbox.delete("1.0", "end")
        if mode.get() == "posthoc":
            pw = power_regression_slope(n=n, r_alt=r, alpha=a, alternative=alternative)
            resbox.insert("end", f"Post hoc power (slope test): {pw:.4f}\n")
        elif mode.get() == "apriori":
            def f(ncand):
                return power_regression_slope(n=int(round(ncand)), r_alt=r, alpha=a, alternative=alternative)
            n_req = math.ceil(_bisect_solve(f, 4, 200000, tgt))
            resbox.insert("end", f"A priori sample size (n): {n_req}\n")
        else:
            def g(rcand):
                return power_regression_slope(n=n, r_alt=rcand, alpha=a, alternative=alternative)
            try:
                r_req = _bisect_solve(g, 1e-6, 0.999, tgt)
            except Exception:
                r_req = float("nan")
            resbox.insert("end", f"Sensitivity r (equiv.) to reach power={tgt:.3f}: r≈{r_req:.4f}\n")

    def _run_mwu(self, alpha, power, alt, mode, delta_var, auc_var, n1_var, n2_var, resbox):
        try:
            a = float(alpha.get()); tgt = float(power.get()); alternative = alt.get()
            n1 = int(float(n1_var.get())); n2 = int(float(n2_var.get()))
            delta = delta_var.get().strip()
            auc = auc_var.get().strip()
            delta = float(delta) if delta != "" else None
            auc = float(auc) if auc != "" else None
        except Exception as ex:
            messagebox.showerror("Input", f"Invalid inputs: {ex}"); return

        resbox.delete("1.0", "end")
        if mode.get() == "posthoc":
            pw = power_mann_whitney(n1, n2, delta_cliff=delta, auc=auc, alpha=a, alternative=alternative)
            resbox.insert("end", f"Post hoc power (Mann–Whitney; approx): {pw:.4f}\n")
        elif mode.get() == "apriori":
            if delta is None and auc is None:
                messagebox.showinfo("A priori", "Provide either Cliff's δ or AUC."); return
            def f(ncand):
                n = max(3, int(round(ncand)))
                return power_mann_whitney(n, n, delta_cliff=delta, auc=auc, alpha=a, alternative=alternative)
            n_req = math.ceil(_bisect_solve(f, 3, 100000, tgt))
            resbox.insert("end", f"A priori (balanced) sample size: n1=n2={n_req}\n")
        else:
            def g_delta(d):
                return power_mann_whitney(n1, n2, delta_cliff=d, auc=None, alpha=a, alternative=alternative)
            try:
                d_req = _bisect_solve(g_delta, 1e-4, 0.999, tgt)
            except Exception:
                d_req = float("nan")
            resbox.insert("end", f"Sensitivity Cliff's δ (δ) to reach power={tgt:.3f}: δ≈{d_req:.4f}\n")

    # ---------- bottom ----------
    def _build_bottom_bar(self):
        bar = ttk.Frame(self); bar.pack(fill="x", padx=self.PADX, pady=(0, self.PADY))
        ttk.Button(bar, text="Copy results to clipboard", command=self._copy_results).pack(side="left")
        ttk.Button(bar, text="Minimize", command=self._minimize_self).pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="Close", command=self.destroy).pack(side="right")

    def _copy_results(self):
        try:
            tab = self.nb.nametowidget(self.nb.select())
            # Find last Text widget in this tab
            stack = [tab]
            last_text = None
            while stack:
                w = stack.pop()
                if isinstance(w, tk.Text):
                    last_text = w
                stack.extend(w.winfo_children())
            if not last_text:
                return
            content = last_text.get("1.0", "end").strip()
            self.clipboard_clear()
            self.clipboard_append(content)
            self.update()
        except Exception:
            pass


# Standalone run for manual testing
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Demo – Power analysis")
    ttk.Button(
        root, text="Open power analysis",
        command=lambda: PowerAnalysisWindow(master=root, get_groups_callback=None)
    ).pack(padx=10, pady=10)
    root.mainloop()
