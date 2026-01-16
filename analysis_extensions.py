
# ============================================================
# analysis_extensions.py
# Additional plotting methods for integration with the main GUI
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# KAPLAN–MEIER SURVIVAL CURVE
# ============================================================

def plot_kaplan_meier(ax, df, time_col="time", event_col="event"):
    """Plot a Kaplan–Meier survival curve using lifelines if installed."""
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        ax.text(0.5, 0.5,
                "Install 'lifelines' to use Kaplan–Meier\npip install lifelines",
                ha="center", va="center")
        return ax

    if time_col not in df.columns or event_col not in df.columns:
        ax.text(0.5, 0.5,
                f"Missing columns:\n'{time_col}' and '{event_col}'",
                ha="center", va="center")
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

def run_and_plot_exponential_regression_global(ax, x_vec, y_vec, color="#2F3B52"):
    """Global exponential regression: fits means across all groups."""
    x = np.asarray(x_vec, dtype=float)
    y = np.asarray(y_vec, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if mask.sum() < 2:
        ax.text(0.5, 0.5,
                "Not enough positive Y values for exp fit",
                transform=ax.transAxes,
                ha="center", va="center")
        return ax

    logy = np.log(y[mask])
    slope, intercept, r, p, stderr = stats.linregress(x[mask], logy)

    a = np.exp(intercept)
    b = slope
    r2 = r * r

    xfit = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 200)
    yfit = a * np.exp(b * xfit)

    ax.plot(xfit, yfit, lw=2, color=color,
            label=f"Exp fit: y={a:.3g}e^({b:.3g}x), R²={r2:.3f}")

    ax.scatter(x[mask], y[mask], s=28, color=color, edgecolor="#333", zorder=3)
    ax.legend(loc="best")

    return ax


# ============================================================
# EXPONENTIAL REGRESSION — SERIES
# Separate exponential fit per series
# ============================================================

def run_and_plot_exponential_regressions_series(ax, x_vec, categories, series_means, colors):
    """
    Performs exponential regression for each series independently.
    series_means: dict {series_name: array of y-values for each group}
    """
    x = np.asarray(x_vec, dtype=float)

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

        ax.plot(xfit, yfit, lw=2, color=col,
                label=f"{series} Exp fit: R²={r2:.3f}")

        ax.scatter(x[mask], y[mask], s=28, color=col,
                   edgecolor="#333333", zorder=3)

    ax.legend(loc="best")
    return ax
