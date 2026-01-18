
# ============================================================
# analysis_extensions.py
# Additional plotting methods for integration with the main GUI
# ============================================================

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# KAPLAN–MEIER SURVIVAL CURVE
# ============================================================
def plot_kaplan_meier(ax, df, time_col: str = "time", event_col: str = "event"):
    """
    Plot a Kaplan–Meier survival curve using lifelines if installed.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    df : pandas.DataFrame
        Must contain time_col and event_col.
    time_col : str
        Column name with durations.
    event_col : str
        Column name with event indicators (1=event, 0=censored).
    """
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        ax.text(
            0.5, 0.5,
            "Install 'lifelines' to use Kaplan–Meier\npip install lifelines",
            ha="center", va="center", transform=ax.transAxes
        )
        return ax

    if time_col not in df.columns or event_col not in df.columns:
        ax.text(
            0.5, 0.5,
            f"Missing columns:\n'{time_col}' and '{event_col}'",
            ha="center", va="center", transform=ax.transAxes
        )
        return ax

    kmf = KaplanMeierFitter()
    kmf.fit(durations=df[time_col], event_observed=df[event_col])
    kmf.plot_survival_function(ax=ax)
    ax.set_title("Kaplan–Meier Survival Curve")
    ax.set_xlabel(time_col)
    ax.set_ylabel("Survival Probability")
    ax.grid(True, alpha=0.3)
    return ax


# ============================================================
# EXPONENTIAL REGRESSION — GLOBAL
# y = a * exp(b * x)
# ============================================================
def run_and_plot_exponential_regression_global(
    ax,
    x_vec,
    y_vec,
    color: str = "#2F3B52"
):
    """
    Global exponential regression: fits across all points (x,y).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x_vec, y_vec : array-like
    color : str
        Line/marker color
    """
    x = np.asarray(x_vec, dtype=float)
    y = np.asarray(y_vec, dtype=float)

    # Need positive y for log transform
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if mask.sum() < 2:
        ax.text(
            0.5, 0.5,
            "Not enough positive Y values for exp fit",
            transform=ax.transAxes, ha="center", va="center"
        )
        return ax

    logy = np.log(y[mask])
    slope, intercept, r, p, stderr = stats.linregress(x[mask], logy)
    a = np.exp(intercept)
    b = slope
    r2 = r * r

    xfit = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 200)
    yfit = a * np.exp(b * xfit)

    ax.plot(
        xfit, yfit, lw=2, color=color,
        label=f"Exp fit: y={a:.3g}e^({b:.3g}x), R²={r2:.3f}"
    )
    ax.scatter(
        x[mask], y[mask], s=28, color=color,
        edgecolor="#333", zorder=3
    )
    ax.legend(loc="best")
    return ax


# ============================================================
# EXPONENTIAL REGRESSION — SERIES
# Separate exponential fit per series
# ============================================================
def run_and_plot_exponential_regressions_series(
    ax,
    x_vec,
    categories,
    series_means,
    colors
):
    """
    Performs exponential regression for each series independently.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x_vec : array-like
        X coordinates (per category).
    categories : List[str]
        Category labels (used only for xtick text elsewhere).
    series_means : Dict[str, array-like]
        {series_name : array of y-values per category}
    colors : Dict[str, str]
        {series_name : HEX color}
    """
    x = np.asarray(x_vec, dtype=float)
    any_plotted = False

    for series, y in series_means.items():
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
        if mask.sum() < 2:
            continue

        logy = np.log(y[mask])
        slope, intercept, r, p, stderr = stats.linregress(x[mask], logy)
        a = np.exp(intercept)
        b = slope
        r2 = r * r

        xfit = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 200)
        yfit = a * np.exp(b * xfit)
        col = colors.get(series, "#2F3B52")

        ax.plot(
            xfit, yfit, lw=2, color=col,
            label=f"{series} Exp fit: R²={r2:.3f}"
        )
        ax.scatter(
            x[mask], y[mask], s=28, color=col,
            edgecolor="#333333", zorder=3
        )
        any_plotted = True

    if any_plotted:
        ax.legend(loc="best")
    return ax


# ============================================================
# BOXPLOT (per-category) — aligned to 0..N-1 positions
# ============================================================
def plot_box(ax, categories, data, colors):
    vals = [np.asarray(data[c], dtype=float) for c in categories]
    pos = np.arange(len(categories), dtype=float)  # 0-based alignment

    bp = ax.boxplot(
        vals,
        positions=pos,          # align with 0..N-1
        patch_artist=True,
        labels=None             # labels set below for full control
    )

    # Color boxes per category
    for patch, cat in zip(bp['boxes'], categories):
        patch.set_facecolor(colors.get(cat, "#999999"))
        patch.set_alpha(0.75)
        patch.set_edgecolor("#444444")

    # Style medians/whiskers/caps
    for artist in bp.get('whiskers', []) + bp.get('caps', []) + bp.get('medians', []):
        artist.set_color("#444444")

    # Pin the x-axis exactly over our integer grid (no padding),
    # so brackets (computed on 0..N-1) line up visually.
    ax.set_xlim(-0.5, len(categories) - 0.5)
    ax.margins(x=0)                # remove horizontal padding
    ax.autoscale(enable=False, axis="x")  # keep x fixed during later drawing

    ax.set_xticks(pos)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)

    return ax


# ============================================================
# VIOLIN PLOT (per-category) — aligned to 0..N-1 positions
# ============================================================
def plot_violin(ax, categories, data, colors):
    """
    Draw a violin plot with one violin per category, aligned to 0..N-1,
    matching execute_analysis() x-coordinates and bracket logic.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    categories : List[str]
    data : Dict[str, List[float]]
        Mapping category -> numeric values
    colors : Dict[str, str]
        Mapping category -> HEX color
    """
    vals = [np.asarray(data[c], dtype=float) for c in categories]
    pos = np.arange(len(categories), dtype=float)  # 0-based alignment

    v = ax.violinplot(
        vals,
        positions=pos,          # align with 0..N-1
        showmeans=False,
        showmedians=True,
        showextrema=True
    )

    # Color each body to match the category
    for i, body in enumerate(v['bodies']):
        body.set_facecolor(colors.get(categories[i], "#999999"))
        body.set_alpha(0.75)
        body.set_edgecolor("#444444")

    # Style internal lines if present
    for k in ("cmins", "cmaxes", "cbars", "cmedians"):
        if k in v:
            v[k].set_color("#444444")

    # Pin the x-axis over our integer grid (no padding)
    ax.set_xlim(-0.5, len(categories) - 0.5)
    ax.margins(x=0)
    ax.autoscale(enable=False, axis="x")  # keep x fixed during later drawing

    ax.set_xticks(pos)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)

    return ax
