
from __future__ import annotations 
import os, json, shutil, tempfile, traceback 
from typing import Dict, List, Tuple, Optional 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import stats 
from matplotlib.ticker import ScalarFormatter, FuncFormatter 
# Local tools (same folder as this script) 
from geometry_canvas import InteractivePlane 
from analysis_extensions import ( 
    plot_kaplan_meier, 
    run_and_plot_exponential_regression_global, 
    run_and_plot_exponential_regressions_series 
) 
from scientific_calculator import launch_calculator 
from multiple_regression import MultipleRegressionWindow 
from categorical_tests import CategoricalTestsWindow 
from power_analysis import PowerAnalysisWindow 
from survival_tool import SurvivalAnalysisWindow  
from chemical_calculator import launch_chemical_calculator

# Local import path insurance 
import sys 
_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) 
if _THIS_DIR not in sys.path: 
    sys.path.insert(0, _THIS_DIR) 
from parametric_tests import ParametricTestsWindow # NEW 
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

# ---------------------------------------------------------------------
# Light Dust Blue theme helper (NEW)
# ---------------------------------------------------------------------

# --- Modern stylish theme (merged in by gui_style_upgrade) ---
def apply_light_dust_blue_theme(root: "tk.Tk"):
    """
    Modern slate-indigo theme. Kept under the original name so every existing
    call site picks it up automatically. White entries/comboboxes, padded
    buttons with hover/press states, clean notebook tabs, polished Treeview.
    """
    import sys
    from tkinter import ttk, font as tkfont

    P = {
        "bg":          "#EEF2F8",
        "surface":     "#FFFFFF",
        "surface_alt": "#F5F7FB",
        "border":      "#D5DCE6",
        "fg":          "#1B2433",
        "fg_muted":    "#5B6677",
        "accent":      "#3B5BDB",
        "accent_hov":  "#4C6EF5",
        "accent_act":  "#2F49B8",
        "accent_fg":   "#FFFFFF",
        "danger":      "#E03131",
        "tab_bg":      "#E3E9F2",
        "tab_sel":     "#FFFFFF",
    }

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    # ---- Fonts ----
    family = "Segoe UI"
    if sys.platform == "darwin":
        family = "SF Pro Text"
    elif sys.platform.startswith("linux"):
        for cand in ("Inter", "Ubuntu", "Cantarell", "DejaVu Sans"):
            try:
                tkfont.Font(family=cand); family = cand; break
            except Exception:
                continue
    for fn in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
        try: tkfont.nametofont(fn).configure(family=family, size=10)
        except Exception: pass
    try: tkfont.nametofont("TkHeadingFont").configure(family=family, size=10, weight="bold")
    except Exception: pass
    heading_font = (family, 11, "bold")
    title_font   = (family, 16, "bold")

    # ---- Root + plain tk widget defaults ----
    root.configure(background=P["bg"])
    try:
        root.option_add("*Font", tkfont.nametofont("TkDefaultFont"))
        root.option_add("*selectBackground", P["accent"])
        root.option_add("*selectForeground", P["accent_fg"])
        root.option_add("*Listbox.background", P["surface"])
        root.option_add("*Listbox.foreground", P["fg"])
        root.option_add("*Listbox.selectBackground", P["accent"])
        root.option_add("*Listbox.selectForeground", P["accent_fg"])
        root.option_add("*Listbox.borderWidth", 0)
        root.option_add("*Listbox.highlightThickness", 1)
        root.option_add("*Listbox.highlightBackground", P["border"])
        root.option_add("*Text.background", P["surface"])
        root.option_add("*Text.foreground", P["fg"])
        root.option_add("*Text.borderWidth", 1)
        root.option_add("*Text.highlightThickness", 0)
        root.option_add("*Menu.background", P["surface"])
        root.option_add("*Menu.foreground", P["fg"])
        root.option_add("*Menu.activeBackground", P["accent"])
        root.option_add("*Menu.activeForeground", P["accent_fg"])
        root.option_add("*TCombobox*Listbox.background", P["surface"])
        root.option_add("*TCombobox*Listbox.foreground", P["fg"])
        root.option_add("*TCombobox*Listbox.selectBackground", P["accent"])
        root.option_add("*TCombobox*Listbox.selectForeground", P["accent_fg"])
        root.option_add("*TCombobox*Listbox.borderWidth", 0)
    except Exception:
        pass

    # ---- ttk base ----
    style.configure(".", background=P["bg"], foreground=P["fg"],
                    fieldbackground=P["surface"], bordercolor=P["border"],
                    lightcolor=P["border"], darkcolor=P["border"])

    # Frames / labels
    style.configure("TFrame", background=P["bg"])
    style.configure("Card.TFrame", background=P["surface"], relief="flat", borderwidth=1)
    style.configure("Toolbar.TFrame", background=P["surface_alt"])
    style.configure("TLabel", background=P["bg"], foreground=P["fg"])
    style.configure("Muted.TLabel", background=P["bg"], foreground=P["fg_muted"])
    style.configure("Heading.TLabel", background=P["bg"], foreground=P["fg"], font=heading_font)
    style.configure("Title.TLabel", background=P["bg"], foreground=P["accent"], font=title_font)
    style.configure("TLabelframe", background=P["bg"], foreground=P["fg_muted"],
                    bordercolor=P["border"], relief="solid", borderwidth=1)
    style.configure("TLabelframe.Label", background=P["bg"], foreground=P["fg_muted"],
                    font=heading_font, padding=(4, 0))
    style.configure("TSeparator", background=P["border"])

    # Buttons
    style.configure("TButton",
                    background=P["surface"], foreground=P["fg"],
                    bordercolor=P["border"], lightcolor=P["surface"],
                    darkcolor=P["border"], focusthickness=0,
                    padding=(12, 6), relief="flat")
    style.map("TButton",
              background=[("active", P["surface_alt"]), ("pressed", P["border"])],
              foreground=[("disabled", P["fg_muted"])])
    style.configure("Accent.TButton",
                    background=P["accent"], foreground=P["accent_fg"],
                    bordercolor=P["accent"], lightcolor=P["accent"],
                    darkcolor=P["accent"], padding=(14, 7), relief="flat")
    style.map("Accent.TButton",
              background=[("active", P["accent_hov"]), ("pressed", P["accent_act"])])
    style.configure("Danger.TButton",
                    background=P["danger"], foreground="#FFFFFF",
                    bordercolor=P["danger"], padding=(12, 6), relief="flat")
    style.map("Danger.TButton",
              background=[("active", "#C92A2A"), ("pressed", "#A51111")])

    # Entries / Combobox / Spinbox  → white field, indigo focus ring
    for w in ("TEntry", "TCombobox", "TSpinbox"):
        style.configure(w,
                        fieldbackground=P["surface"], background=P["surface"],
                        foreground=P["fg"], bordercolor=P["border"],
                        lightcolor=P["border"], darkcolor=P["border"],
                        padding=5, relief="flat")
        style.map(w,
                  fieldbackground=[("readonly", P["surface"]), ("disabled", P["surface_alt"])],
                  foreground=[("disabled", P["fg_muted"])],
                  bordercolor=[("focus", P["accent"])],
                  lightcolor=[("focus", P["accent"])],
                  darkcolor=[("focus", P["accent"])])
    # Keep combobox selection white-on-dark-text like original
    style.map("TCombobox",
              selectbackground=[("readonly", P["surface"]), ("!readonly", P["surface"])],
              selectforeground=[("readonly", P["fg"]),      ("!readonly", P["fg"])])

    # Checks / radios
    for w in ("TCheckbutton", "TRadiobutton"):
        style.configure(w, background=P["bg"], foreground=P["fg"],
                        focuscolor=P["bg"], padding=2)
        style.map(w, background=[("active", P["bg"])],
                  foreground=[("disabled", P["fg_muted"])])

    # Treeview
    style.configure("Treeview",
                    background=P["surface"], fieldbackground=P["surface"],
                    foreground=P["fg"], bordercolor=P["border"],
                    rowheight=26, borderwidth=0)
    style.configure("Treeview.Heading",
                    background=P["surface_alt"], foreground=P["fg"],
                    font=heading_font, relief="flat", padding=(8, 6))
    style.map("Treeview",
              background=[("selected", P["accent"])],
              foreground=[("selected", P["accent_fg"])])
    style.map("Treeview.Heading", background=[("active", P["border"])])

    # Notebook
    style.configure("TNotebook", background=P["bg"], borderwidth=0,
                    tabmargins=(8, 6, 8, 0))
    style.configure("TNotebook.Tab",
                    background=P["tab_bg"], foreground=P["fg_muted"],
                    padding=(16, 8), borderwidth=0, font=(family, 10, "bold"))
    style.map("TNotebook.Tab",
              background=[("selected", P["tab_sel"]), ("active", P["surface_alt"])],
              foreground=[("selected", P["accent"]), ("active", P["fg"])],
              expand=[("selected", (1, 1, 1, 0))])

    # Scrollbars / Progressbar / Scale / Paned
    style.configure("Vertical.TScrollbar",
                    background=P["bg"], troughcolor=P["bg"],
                    bordercolor=P["bg"], arrowcolor=P["fg_muted"],
                    gripcount=0, relief="flat")
    style.configure("Horizontal.TScrollbar",
                    background=P["bg"], troughcolor=P["bg"],
                    bordercolor=P["bg"], arrowcolor=P["fg_muted"],
                    gripcount=0, relief="flat")
    style.map("Vertical.TScrollbar",
              background=[("active", P["border"]), ("pressed", P["accent"])])
    style.map("Horizontal.TScrollbar",
              background=[("active", P["border"]), ("pressed", P["accent"])])
    style.configure("Horizontal.TProgressbar",
                    background=P["accent"], troughcolor=P["surface_alt"],
                    bordercolor=P["border"], lightcolor=P["accent"],
                    darkcolor=P["accent"])
    style.configure("TScale", background=P["bg"], troughcolor=P["surface_alt"])
    style.configure("TPanedwindow", background=P["bg"])
    style.configure("Sash", sashthickness=6, background=P["border"])

    # Windows: enable native immersive title bar where supported
    if sys.platform == "win32":
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 20, ctypes.byref(ctypes.c_int(0)), ctypes.sizeof(ctypes.c_int)
            )
        except Exception:
            pass




# ---------------------------------------------------------------------
# I/O helpers

# ---------------------------------------------------------------------
# Integrated Results Panel (NEW)
# ---------------------------------------------------------------------
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)

# ---------------------------------------------------------------------------
# GraphPad Prism-style defaults (clean, publication-quality look)
# ---------------------------------------------------------------------------
def _apply_graphpad_style():
    import matplotlib as _mpl
    # Pick a Prism-like sans font if available, else fall back gracefully
    _preferred = ["Arial", "Helvetica", "Helvetica Neue", "Liberation Sans",
                  "DejaVu Sans"]
    _mpl.rcParams.update({
        # Figure
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        # Fonts
        "font.family": "sans-serif",
        "font.sans-serif": _preferred,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        # Axes / spines  (Prism: only left + bottom, thick)
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 1.6,
        "axes.edgecolor": "#222222",
        "axes.titlepad": 10,
        "axes.labelpad": 6,
        "axes.axisbelow": True,
        # Ticks pointing out, thick
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        "xtick.major.pad": 5,
        "ytick.major.pad": 5,
        # No grid (Prism default)
        "axes.grid": False,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        # Lines & markers
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.0,
        "patch.linewidth": 1.2,
        "patch.edgecolor": "#222222",
        # Error bars
        "errorbar.capsize": 4,
        # Colour cycle (Prism-ish, colour-blind friendly)
        "axes.prop_cycle": _mpl.cycler(color=[
            "#1f77b4", "#d62728", "#2ca02c", "#9467bd",
            "#ff7f0e", "#17becf", "#8c564b", "#e377c2",
        ]),
    })

_apply_graphpad_style()


def _prettify_figure(fig):
    """Apply Prism-style polish to every axes in *fig* (idempotent, safe)."""
    try:
        for ax in fig.get_axes():
            # Hide top/right spines, thicken remaining ones
            for side in ("top", "right"):
                if side in ax.spines:
                    ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                if side in ax.spines:
                    ax.spines[side].set_linewidth(1.6)
                    ax.spines[side].set_color("#222222")
            # Ticks: outward, thick
            ax.tick_params(axis="both", which="major",
                           direction="out", length=6, width=1.6,
                           colors="#222222", pad=5)
            ax.tick_params(axis="both", which="minor",
                           direction="out", length=3, width=1.0,
                           colors="#222222")
            # Bold-ish axis labels and title
            try:
                ax.xaxis.label.set_fontweight("bold")
                ax.yaxis.label.set_fontweight("bold")
                if ax.get_title():
                    ax.title.set_fontweight("bold")
            except Exception:
                pass
            # Legend: no frame
            leg = ax.get_legend()
            if leg is not None:
                leg.set_frame_on(False)
    except Exception:
        pass


_RESULTS_PANEL = None  # set when the integrated GUI is built


def _fmt_cell(v):
    try:
        if isinstance(v, float):
            if pd.isna(v):
                return ""
            return f"{v:.4g}"
    except Exception:
        pass
    return "" if v is None else str(v)



def _attach_pan_zoom(canvas, fig):
    """Cursor-anchored scroll zoom + Shift/middle drag pan + double-click reset.
    Works on every Axes in `fig`. The matplotlib toolbar still works alongside."""
    state = {"home": None, "pan": None}

    def _save_home():
        try:
            state["home"] = [(ax, ax.get_xlim(), ax.get_ylim()) for ax in fig.axes]
        except Exception:
            state["home"] = None
    try:
        canvas.get_tk_widget().after(60, _save_home)
    except Exception:
        _save_home()

    def _ax_at(event):
        if getattr(event, "inaxes", None) is not None:
            return event.inaxes
        for ax in fig.axes:
            try:
                if ax.get_visible() and ax.bbox.contains(event.x, event.y):
                    return ax
            except Exception:
                continue
        return None

    def on_scroll(event):
        ax = _ax_at(event)
        if ax is None:
            return
        base = 1.2
        factor = 1.0 / base if event.button == "up" else base
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        try:
            xdata, ydata = ax.transData.inverted().transform((event.x, event.y))
        except Exception:
            xdata, ydata = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        ax.set_xlim(xdata - (xdata - x0) * factor, xdata + (x1 - xdata) * factor)
        ax.set_ylim(ydata - (ydata - y0) * factor, ydata + (y1 - ydata) * factor)
        canvas.draw_idle()

    def on_press(event):
        is_pan = (event.button == 2) or (
            event.button == 1 and isinstance(event.key, str) and "shift" in event.key
        )
        if not is_pan:
            return
        ax = _ax_at(event)
        if ax is None:
            return
        state["pan"] = (ax, event.x, event.y, ax.get_xlim(), ax.get_ylim())

    def on_motion(event):
        if state["pan"] is None or event.x is None:
            return
        ax, x0, y0, xlim, ylim = state["pan"]
        try:
            inv = ax.transData.inverted()
            p0 = inv.transform((x0, y0))
            p1 = inv.transform((event.x, event.y))
        except Exception:
            return
        dx, dy = p0[0] - p1[0], p0[1] - p1[1]
        ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
        ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
        canvas.draw_idle()

    def on_release(event):
        state["pan"] = None
        if getattr(event, "dblclick", False) and state["home"]:
            for ax, xl, yl in state["home"]:
                ax.set_xlim(xl)
                ax.set_ylim(yl)
            canvas.draw_idle()

    canvas.mpl_connect("scroll_event", on_scroll)
    canvas.mpl_connect("button_press_event", on_press)
    canvas.mpl_connect("motion_notify_event", on_motion)
    canvas.mpl_connect("button_release_event", on_release)


class ResultsPanel(ttk.Frame):
    """Right-hand notebook that embeds tables and matplotlib figures."""

    def __init__(self, master):
        super().__init__(master, padding=(6, 6))
        header = ttk.Frame(self)
        header.pack(fill="x")
        ttk.Label(header, text="Results",
                  font=("TkDefaultFont", 11, "bold")).pack(side="left")
        ttk.Button(header, text="Clear all", command=self.clear
                   ).pack(side="right")
        ttk.Button(header, text="Close current", command=self.close_current
                   ).pack(side="right", padx=(0, 6))
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, pady=(6, 0))
        self._empty = None
        self._show_welcome()
        self._fig_counter = 0
        # Hook fired whenever a new result tab is added — used to raise
        # the outer Spreadsheet/Results notebook to the Results pane.
        self.on_activity = None


    def _show_welcome(self):
        self._empty = ttk.Label(
            self.nb,
            text="Run an analysis — tables and figures appear here as tabs.",
            padding=24, anchor="center", justify="center",
        )
        self.nb.add(self._empty, text="Welcome")

    def _drop_welcome(self):
        if self._empty is not None:
            try:
                self.nb.forget(self._empty)
            except Exception:
                pass
            self._empty = None

    def clear(self):
        for t in list(self.nb.tabs()):
            self.nb.forget(t)
        self._show_welcome()

    def close_current(self):
        cur = self.nb.select()
        if cur:
            self.nb.forget(cur)
            if not self.nb.tabs():
                self._show_welcome()

    def _add_tab(self, widget, title):
        self._drop_welcome()
        short = title if len(title) <= 38 else title[:35] + "…"
        self.nb.add(widget, text=short)
        self.nb.select(widget)
        try:
            if callable(self.on_activity):
                self.on_activity()
        except Exception:
            pass


    def add_dataframe(self, df, title):
        frame = ttk.Frame(self.nb)
        cols = [str(c) for c in df.columns]
        tv = ttk.Treeview(frame, columns=cols, show="headings", height=20)
        for c in cols:
            tv.heading(c, text=c)
            try:
                sample = df[c].astype(str).head(20).tolist()
            except Exception:
                sample = [""]
            w = max(80, min(280, 9 * max([len(c)] + [len(s) for s in sample])))
            tv.column(c, width=w, anchor="w", stretch=True)
        for _, row in df.iterrows():
            tv.insert("", "end", values=[_fmt_cell(v) for v in row.tolist()])
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tv.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tv.xview)
        tv.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tv.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="we")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self._add_tab(frame, title)

    def add_rows(self, rows, title):
        if not rows:
            self.add_text("(no rows)", title); return
        self.add_dataframe(pd.DataFrame(rows), title)

    def add_text(self, text, title):
        frame = ttk.Frame(self.nb)
        txt = tk.Text(frame, wrap="word", height=20)
        txt.insert("1.0", text)
        txt.configure(state="disabled")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
        txt.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self._add_tab(frame, title)

    def add_figure(self, fig, title=None):
        self._fig_counter += 1
        if not title:
            try:
                title = (fig._suptitle.get_text() if fig._suptitle else "") \
                        or (fig.axes[0].get_title() if fig.axes else "") \
                        or f"Figure {self._fig_counter}"
            except Exception:
                title = f"Figure {self._fig_counter}"
        frame = ttk.Frame(self.nb)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        # Polish to Prism/GraphPad-style look before drawing
        try:
            _prettify_figure(fig)
        except Exception:
            pass
        # Use constrained_layout so labels/titles fit the Tk canvas at any size,
        # and re-apply on container resize. Zoom/pan only change axis limits,
        # so axes positions stay stable while the graph keeps its scale.
        try:
            fig.set_layout_engine("constrained")
        except Exception:
            try:
                fig.tight_layout()
            except Exception:
                pass
        canvas.draw()
        def _on_resize(event, _f=fig, _c=canvas):
            try:
                _f.canvas.draw_idle()
            except Exception:
                pass
        canvas.get_tk_widget().bind("<Configure>", _on_resize, add="+")
        toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        # Cursor-anchored scroll zoom + Shift/middle drag pan + dbl-click reset
        try:
            _attach_pan_zoom(canvas, fig)
        except Exception:
            pass
        frame._fig = fig
        frame._canvas = canvas
        frame._toolbar = toolbar
        self._add_tab(frame, title)


def _embed_pending_figures():
    """Capture all currently-open matplotlib figures into the results panel."""
    if _RESULTS_PANEL is None:
        return False
    nums = plt.get_fignums()
    if not nums:
        return True
    for n in nums:
        fig = plt.figure(n)
        _RESULTS_PANEL.add_figure(fig)
    for n in list(plt.get_fignums()):
        try:
            plt._pylab_helpers.Gcf.destroy(n)
        except Exception:
            pass
    return True


# Monkey-patch plt.show so any plt.show() call lands in the results panel
# when the integrated GUI is active. Falls back to the normal window when not.
_ORIG_PLT_SHOW = plt.show
def _plt_show_integrated(*args, **kwargs):
    if _RESULTS_PANEL is None:
        return _ORIG_PLT_SHOW(*args, **kwargs)
    _embed_pending_figures()
plt.show = _plt_show_integrated

# ---------------------------------------------------------------------

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
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------
# Color helpers
# NEW: Edge color + hatch (pattern) support
EDGE_COLORS = {}  # e.g., {"Group1": "black", "Group2": None}
HATCHES = {}      # e.g., {"Group1": "//", "Group2": "xx", "Group3": ""}

# Rotating infill patterns auto-assigned per category (in addition to color).
HATCH_PALETTE = ["//", "xx", "..", "++", "\\\\", "oo", "**", "||", "--", "OO"]

def clean_hatch(h):
    # Return the hatch string as-is. Stripping backslashes silently turned
    # "\\" patterns into "no infill", so we only normalize non-strings.
    if not isinstance(h, str):
        return ""
    return h

def ensure_hatches_for_keys(keys, hatches=None):
    """Auto-assign a unique infill pattern to each category, preserving any
    user-specified entries. Also updates the global HATCHES dict so bar
    plots pick up patterns automatically."""
    out = dict(hatches or {})
    for i, k in enumerate(keys):
        # Only auto-assign when the key is missing entirely. An explicit ""
        # means the user chose "(none)" / solid and must be preserved.
        if k not in out:
            out[k] = HATCH_PALETTE[i % len(HATCH_PALETTE)]
    HATCHES.update(out)
    return out

 
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
# ---------------------------------------------------------------------
# Stats, formatting, tables, plots 
def adjust_pvals(pvals: List[float], method: str = "holm") -> np.ndarray: 
    pvals = np.asarray(pvals, dtype=float) 
    if method is None or str(method).strip().lower() == 'none': 
        return pvals 
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
    if p < 0.0001: return "p<0.0001" 
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
        if p < 0.0001: return "****" 
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
# Q–Q plots and tables 
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
    # Add scroll-wheel zoom for y-axis (except Pie chart) 
    def on_scroll(event): 
        if event.inaxes != ax: 
            return 
        cur_ylim = ax.get_ylim() 
        y_min, y_max = cur_ylim 
        range_y = (y_max - y_min) 
        base_scale = 1.1 
        if event.button == 'up': 
            scale_factor = 1 / base_scale 
        elif event.button == 'down': 
            scale_factor = base_scale 
        else: 
            scale_factor = 1 
        new_range = range_y * scale_factor 
        center = event.ydata if event.ydata is not None else (y_min + y_max) / 2 
        new_ymin = center - new_range / 2 
        new_ymax = center + new_range / 2 
        ax.set_ylim(new_ymin, new_ymax) 
        ax.figure.canvas.draw_idle() 
    fig.canvas.mpl_connect('scroll_event', on_scroll) 
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
# ---------------------------------------------------------------------
# NEW: builder for the "long table" (export + viewer reuse) 
def build_pairwise_long_table(pairwise_results) -> pd.DataFrame: 
    """ 
    Build the formatted 'long' table for pairwise results, reusing the same 
    formatting shown in the popup (stat to 3dp, p values in 3e/≤0.1 style). 
    """ 
    if not pairwise_results: 
        return pd.DataFrame(columns=["ref", "group", "name", "stat", "p_raw", "p_adj"]) 
    df = pd.DataFrame(pairwise_results).copy() 
    for c in ["ref", "group", "name", "stat", "p_raw", "p_adj"]: 
        if c not in df.columns: 
            df[c] = np.nan 
    def _fmt_stat(x): 
        return "" if pd.isna(x) else f"{x:.3f}" 
    def _fmt_p(x): 
        if pd.isna(x): 
            return "" 
        try: 
            x = float(x) 
        except Exception: 
            return str(x) 
        if 0.001 <= x < 0.1: 
            s = f"{x:.6f}".rstrip("0").rstrip(".") 
        else: 
            s = f"{x:.3e}" 
        return s 
    df_long = df[["ref", "group", "name", "stat", "p_raw", "p_adj"]].copy() 
    df_long["stat"] = df_long["stat"].map(_fmt_stat) 
    df_long["p_raw"] = df_long["p_raw"].map(_fmt_p) 
    df_long["p_adj"] = df_long["p_adj"].map(_fmt_p) 
    df_long = df_long.sort_values(["ref", "group", "name"], kind="stable") 
    return df_long 
# ---------------------------------------------------------------------
# NEW: scrollable popup for the significance (pairwise) long table 
def show_pairwise_long_table_popup(df_long: pd.DataFrame, title: str = "Pairwise comparisons — p-values (long table)"): 
    if _RESULTS_PANEL is not None:
        _RESULTS_PANEL.add_dataframe(df_long, title)
        return
    win = tk.Toplevel()
    win.title(title)
    win.geometry("1100x600")
    frame = ttk.Frame(win, padding=(8, 8))
    frame.pack(fill="both", expand=True)
    cols = [str(c) for c in df_long.columns]
    tv = ttk.Treeview(frame, columns=cols, show="headings", height=20)
    for c in cols:
        tv.heading(c, text=c)
        tv.column(c, width=140, anchor="w", stretch=True)
    for _, row in df_long.iterrows():
        tv.insert("", "end", values=[_fmt_cell(v) for v in row.tolist()])
    vsb = ttk.Scrollbar(frame, orient="vertical", command=tv.yview)
    tv.configure(yscrollcommand=vsb.set)
    tv.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")

def show_pairwise_long_table_figure(pairwise_results): 
    if not pairwise_results: 
        print("\n(No pairwise comparisons to display)") 
        return 
    df_long = build_pairwise_long_table(pairwise_results) 
    # Thresholds for switching to the scrollable popup 
    ROW_THRESHOLD = 2 
    COL_THRESHOLD = 8 
    if len(df_long) > ROW_THRESHOLD or df_long.shape[1] > COL_THRESHOLD: 
        show_pairwise_long_table_popup(df_long, title="Pairwise comparisons — p-values (long table)") 
        return 
    # For smaller tables, keep the figure but improve readability 
    fig = _make_table_figure( 
        df_long, 
        title="Pairwise comparisons — p-values (long table)", 
        max_height_in=22.0 
    ) 
    # Optionally bump font size and row height 
    for a in fig.axes: 
        for artist in a.get_children(): 
            if hasattr(artist, 'set_fontsize') and hasattr(artist, 'scale'): 
                try: 
                    artist.set_fontsize(10) # larger text 
                    artist.scale(1.0, 1.50) # taller rows 
                except Exception: 
                    pass 
    plt.tight_layout() 
def show_normality_table(rows: List[Dict[str, str]], title: str = "Normality tests"): 
    if _RESULTS_PANEL is not None:
        _RESULTS_PANEL.add_rows(rows, title)
        return
    win = tk.Toplevel()
    win.title(title)
    win.geometry("1100x600")
    frame = ttk.Frame(win, padding=(8, 8))
    frame.pack(fill="both", expand=True)
    df_n = pd.DataFrame(rows) if rows else pd.DataFrame()
    cols = [str(c) for c in df_n.columns] if not df_n.empty else ["info"]
    tv = ttk.Treeview(frame, columns=cols, show="headings", height=20)
    for c in cols:
        tv.heading(c, text=c)
        tv.column(c, width=140, anchor="w", stretch=True)
    if df_n.empty:
        tv.insert("", "end", values=("(no rows)",))
    else:
        for _, row in df_n.iterrows():
            tv.insert("", "end", values=[_fmt_cell(v) for v in row.tolist()])
    vsb = ttk.Scrollbar(frame, orient="vertical", command=tv.yview)
    tv.configure(yscrollcommand=vsb.set)
    tv.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")

def show_descriptives_table(df: pd.DataFrame, title: str = "Descriptives"): 
    if _RESULTS_PANEL is not None:
        _RESULTS_PANEL.add_dataframe(df, title)
        return
    win = tk.Toplevel()
    win.title(title)
    win.geometry("1100x600")
    frame = ttk.Frame(win, padding=(8, 8))
    frame.pack(fill="both", expand=True)
    cols = [str(c) for c in df.columns]
    tv = ttk.Treeview(frame, columns=cols, show="headings", height=20)
    for c in cols:
        tv.heading(c, text=c)
        tv.column(c, width=140, anchor="w", stretch=True)
    for _, row in df.iterrows():
        tv.insert("", "end", values=[_fmt_cell(v) for v in row.tolist()])
    vsb = ttk.Scrollbar(frame, orient="vertical", command=tv.yview)
    tv.configure(yscrollcommand=vsb.set)
    tv.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")

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
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------
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
    """ Boxplot aligned to 0..N-1 with fixed xlim to match bracket x-positions. """ 
    import numpy as np 
    vals = [np.asarray(data[c], dtype=float) for c in categories] 
    pos = np.arange(len(categories), dtype=float) # 0-based alignment 
    bp = ax.boxplot( 
        vals, 
        positions=pos, 
        patch_artist=True, 
        labels=None 
    ) 
    for patch, cat in zip(bp['boxes'], categories): 
        patch.set_facecolor(colors.get(cat, "#999999")) 
        patch.set_alpha(0.75) 
        patch.set_edgecolor("#444444") 
    for artist in bp.get('whiskers', []) + bp.get('caps', []) + bp.get('medians', []): 
        artist.set_color("#444444") 
    # Pin x-axis 
    ax.set_xlim(-0.5, len(categories) - 0.5) 
    ax.margins(x=0) 
    ax.set_xticks(pos) 
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8) 
    ax.autoscale(enable=False, axis="x") 
    return ax 
def plot_violin(ax, categories, data, colors): 
    """ Violin plot aligned to 0..N-1 with fixed xlim to match bracket x-positions. """ 
    import numpy as np 
    vals = [np.asarray(data[c], dtype=float) for c in categories] 
    pos = np.arange(len(categories), dtype=float) 
    v = ax.violinplot( 
        vals, 
        positions=pos, 
        showmeans=False, 
        showmedians=True, 
        showextrema=True 
    ) 
    for i, body in enumerate(v['bodies']): 
        body.set_facecolor(colors.get(categories[i], "#999999")) 
        body.set_alpha(0.75) 
        body.set_edgecolor("#444444") 
    for k in ("cmins", "cmaxes", "cbars", "cmedians"): 
        if k in v: 
            v[k].set_color("#444444") 
    ax.set_xlim(-0.5, len(categories) - 0.5) 
    ax.margins(x=0) 
    ax.set_xticks(pos) 
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8) 
    ax.autoscale(enable=False, axis="x") 
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
                   # edgecolor doubles as the hatch color in matplotlib;
                   # keep linewidth=0 so the pattern is colored without
                   # drawing a thick outline around the bar.
                   edgecolor=EDGE_COLORS.get(cat, "#444444"),
                   linewidth=0,
                   hatch=clean_hatch(HATCHES.get(cat, "")), alpha=0.80, zorder=1) 
        ax.errorbar(x, means, yerr=ci_half, fmt="None", ecolor="black", 
                    elinewidth=1.5, capsize=6, capthick=1.5, zorder=2, label=ci_label) 
    elif line: 
        ax.plot(x, means, color="#333333", marker="o", lw=1.8, zorder=2, label="Mean") 
        ax.errorbar(x, means, yerr=ci_half, fmt="None", ecolor="#333333", 
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
# ---------------------------------------------------------------------
# Execute analysis 
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
        lo, hi = sorted([i1, i2]) 
        base_y = float(np.nanmax(bar_tops[lo:hi+1])) + base_gap 
        comparisons.append({"x1": float(x[i1]), "x2": float(x[i2]), "base_y": base_y, 
                            "text": format_sig_marker(p_adj, sig_marker_mode)}) 
    comparisons.sort(key=lambda d: abs(d["x2"] - d["x1"])) 
    brackets = [] 
    def _x_overlaps(b1, b2, eps=1e-9): 
        lo1, hi1 = min(b1["x1"], b1["x2"]), max(b1["x1"], b1["x2"]) 
        lo2, hi2 = min(b2["x1"], b2["x2"]), max(b2["x1"], b2["x2"]) 
        return not (hi1 < lo2 - eps or hi2 < lo1 - eps) 
    for c in comparisons: 
        y = c["base_y"] 
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
        lo, hi = sorted([i1, i2]) 
        base_y = float(np.nanmax(bar_tops[lo:hi+1])) + base_gap 
        candidates.append({"x1": float(x[i1]), "x2": float(x[i2]), "base_y": base_y, 
                           "text": format_sig_marker(p_adj, sig_marker_mode)}) 
    candidates.sort(key=lambda d: abs(d["x2"] - d["x1"])) 
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
def draw_cross_graph_bracket(fig, ax1, ax2, x1, x2, y, text, line_h=0.02): 
    """Draw a significance bracket spanning two subplots.""" 
    p1 = ax1.transData.transform((x1, y)) 
    p2 = ax2.transData.transform((x2, y)) 
    inv = fig.transFigure.inverted() 
    fp1 = inv.transform(p1) 
    fp2 = inv.transform(p2) 
    line = plt.Line2D([fp1[0], fp1[0], fp2[0], fp2[0]], 
                      [fp1[1], fp1[1]+line_h, fp2[1]+line_h, fp2[1]], 
                      transform=fig.transFigure, color='black', lw=1.2, zorder=10) 
    fig.add_artist(line) 
    fig.text((fp1[0]+fp2[0])/2, fp1[1]+line_h, text, 
             ha='center', va='bottom', fontsize=8, color='black') 
def execute_analysis(cfg: Dict[str, str], data_dict: Dict[str, List[float]], df_data: pd.DataFrame, result_hook=None): 
    """ 
    UPDATED: 
    - Analysis uses full data (categories_all, data_dict, df_data). 
    - Plotting uses filtered selection (categories_plot, data_plot, df_plot) based on: 
      cfg["selected_groups"] and cfg["hide_deselected_plot"]. 
    - Calls result_hook({"pairwise_results": ..., "alpha": ...}) when done. 
    """ 
    # ---- selection for plotting vs analysis universe ---- 
    categories_all = list(data_dict.keys()) 
    default_ref = cfg["default_ref"] if cfg["default_ref"] else ("-dox" if "-dox" in categories_all else (categories_all[0] if categories_all else "")) 
    ref = cfg["ref"] if cfg["ref"] else default_ref 
    exclude_vs = cfg["exclude"] if cfg["exclude"] else None 
    def _get_float(val: str, default: float) -> float: 
        try: return float(val) 
        except: return default 
    try: alpha = float(cfg["alpha"]) 
    except Exception: alpha = 0.05 
    try: mu_val = float(cfg["mu"]) 
    except Exception: mu_val = 0.0 
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
    def _fs(v, d):
        try:
            f = float(str(v).strip());  return f if f > 0 else d
        except Exception:
            return d
    _yls_fs = _fs(cfg.get("y_label_fontsize"), 15)
    _ytk_fs = _fs(cfg.get("y_tick_fontsize"), 10)
    _sig_fs = _fs(cfg.get("sig_fontsize"), 10)
    base_gap_factor = _get_float(cfg["base_gap_factor"], 0.08) 
    stack_step_factor= _get_float(cfg["stack_step_factor"], 0.09) 
    line_height_factor= _get_float(cfg["line_height_factor"],0.025) 
    collision_k = _get_float(cfg["collision_k"], 0.75) 
    top_margin_factor= _get_float(cfg["top_margin_factor"], 0.12) 
    try: 
        colors_cat = json.loads(cfg.get("colors_json","{}")) 
        if not isinstance(colors_cat, dict): colors_cat = {} 
    except Exception: 
        colors_cat = {} 
    colors_cat = ensure_colors_for_keys(categories_all, colors_cat) 
    # Pull per-group infill patterns + pattern colors from cfg, then auto-fill
    # any missing patterns so every category gets a unique one.
    try:
        user_hatches = json.loads(cfg.get("hatches_json", "{}")) or {}
        if not isinstance(user_hatches, dict): user_hatches = {}
    except Exception:
        user_hatches = {}
    try:
        user_hatch_colors = json.loads(cfg.get("hatch_colors_json", "{}")) or {}
        if not isinstance(user_hatch_colors, dict): user_hatch_colors = {}
    except Exception:
        user_hatch_colors = {}
    HATCHES.clear()
    HATCHES.update({k: v for k, v in user_hatches.items() if isinstance(v, str)})
    # Do not auto-assign patterns. Bars are plain unless the user picks one.
    # Pattern color = matplotlib hatch color, which follows the patch edgecolor.
    EDGE_COLORS.clear()
    for k, v in user_hatch_colors.items():
        if isinstance(v, str) and v.startswith("#") and len(v) in (4, 7):
            EDGE_COLORS[k] = v
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
    # ---- selection flags from cfg for plotting ---- 
    selected_csv = (cfg.get("selected_groups") or "").strip() 
    hide_deselected_flag = (cfg.get("hide_deselected_plot", "False").lower() == "true") 
    selected_set = set([s for s in selected_csv.split(",") if s]) if selected_csv else set(categories_all) 
    # ---- honor user-specified group order (NEW) ---- 
    order_csv = (cfg.get("group_order") or "").strip() 
    order_list = [s.strip() for s in order_csv.split(",")] if order_csv else [] 
    order_list = [g for g in order_list if g in categories_all] 
    if order_list: 
        if hide_deselected_flag: 
            base = [g for g in order_list if g in selected_set] 
        else: 
            base = order_list[:] 
        tail = [g for g in categories_all if g not in base and ((not hide_deselected_flag) or (g in selected_set))] 
        categories_plot = base + tail 
    else: 
        categories_plot = [g for g in categories_all if (not hide_deselected_flag) or (g in selected_set)] 
    data_plot = {g: data_dict[g] for g in categories_plot} 
    df_plot = df_data[df_data["group"].isin(categories_plot)].copy() 
    # ---- Helper to deliver hook safely ---- 
    def _deliver_hook(pairwise_results_list): 
        if result_hook is not None: 
            try: 
                result_hook({"pairwise_results": pairwise_results_list or [], "alpha": alpha}) 
            except Exception: 
                pass 
    # ========== ANALYSIS (uses full data — categories_all, data_dict, df_data) ========== 
    only_plot = (cfg["analysis"] == "None") 
    if not only_plot: 
        diagnostics(data_dict) 
    pairwise_results: List[Dict[str, float]] = [] 
    single_annos: List[Dict[str, float]] = [] 
    ph = cfg["posthoc"]; scope = cfg["scope"]; correction = cfg["correction"] 
    if not only_plot: 
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
            _deliver_hook([]) # no pairwise here 
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
            _deliver_hook([]) # no pairwise here 
            return 
        if cfg["analysis"] == "ANOVA": 
            group_arrays = [np.array(data_dict[c], dtype=float) for c in categories_all] 
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
            group_arrays = [np.array(data_dict[c], dtype=float) for c in categories_all] 
            h_stat, p_val = stats.kruskal(*group_arrays) 
            print(f"\nGlobal test (Kruskal–Wallis): H={float(h_stat):.3f}, p={float(p_val):.3e}") 
            if ph == "Dunn's": 
                pairwise_results = run_dunn_all_pairs(df_data, correction=correction) 
            elif ph in {"vs_ref (Mann–Whitney)","all_pairs (Mann–Whitney)"}: 
                if "vs_ref" in ph: 
                    out = [] 
                    for g in categories_all: 
                        if g == ref or (exclude_vs and g == exclude_vs): continue 
                        u_stat, p_val = stats.mannwhitneyu(data_dict[ref], data_dict[g], alternative="two-sided") 
                        out.append({"ref": ref, "group": g, "name": "Mann–Whitney U", "stat": float(u_stat), "p_raw": float(p_val)}) 
                    p_adj = adjust_pvals([r["p_raw"] for r in out], method=correction) 
                    for r, pa in zip(out, p_adj): r["p_adj"] = float(pa) 
                    pairwise_results = out 
                else: 
                    out = [] 
                    for i in range(len(categories_all)): 
                        for j in range(i+1, len(categories_all)): 
                            g1, g2 = categories_all[i], categories_all[j] 
                            u_stat, p_val = stats.mannwhitneyu(data_dict[g1], data_dict[g2], alternative="two-sided") 
                            out.append({"ref": g1, "group": g2, "name": "Mann–Whitney U", "stat": float(u_stat), "p_raw": float(p_val)}) 
                    p_adj = adjust_pvals([r["p_raw"] for r in out], method=correction) 
                    for r, pa in zip(out, p_adj): r["p_adj"] = float(pa) 
                    pairwise_results = out 
        elif cfg["analysis"] == "Mann–Whitney (2 groups)": 
            if len(categories_all) < 2: 
                print("\nMann–Whitney: need at least 2 groups.") 
            else: 
                out = [] 
                if scope == "vs_ref": 
                    for g in categories_all: 
                        if g == ref or (exclude_vs and g == exclude_vs): continue 
                        u_stat, p_val = stats.mannwhitneyu(data_dict[ref], data_dict[g], alternative="two-sided") 
                        out.append({"ref": ref, "group": g, "name": "Mann–Whitney U", "stat": float(u_stat), "p_raw": float(p_val)}) 
                else: 
                    for i in range(len(categories_all)): 
                        for j in range(i+1, len(categories_all)): 
                            g1, g2 = categories_all[i], categories_all[j] 
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
        # ---- EXPORTS: RAW + LONG TABLE ---- 
        if cfg["export_csv"] and pairwise_results: 
            pd.DataFrame(pairwise_results).to_csv(cfg["export_csv"], index=False) 
            print(f"\nExported post-hoc results to: {cfg['export_csv']}") 
        if pairwise_results: 
            df_long = build_pairwise_long_table(pairwise_results) 
            _write_xlsx(df_long, "posthoc_long") 
            if cfg["export_csv"]: 
                base, ext = os.path.splitext(cfg["export_csv"]) 
                long_csv_path = f"{base}-long{ext or '.csv'}" 
                df_long.to_csv(long_csv_path, index=False) 
                print(f"\nExported long table to: {long_csv_path}") 
            _write_xlsx(pd.DataFrame(pairwise_results), "posthoc") 
        # Show significance table 
        show_pairwise_long_table_figure(pairwise_results) 
    # ===================== PLOTTING (uses categories_plot, data_plot, df_plot) ===================== 
    # If nothing selected and hide is on: show a blank note figure and exit 
    if len(categories_plot) == 0: 
        plt.rcParams.update({"figure.dpi":160, "axes.spines.top":False, "axes.spines.right":False, "axes.linewidth":1.0}) 
        fig, ax = plt.subplots(figsize=(6.4, 3.6)) 
        ax.axis("off") 
        ax.text(0.5, 0.5, "No groups selected to plot", ha="center", va="center", fontsize=12) 
        plt.tight_layout() 
        if cfg["save_fig"]: 
            ext = os.path.splitext(cfg["save_fig"])[1].lower() 
            if ext in {".tif", ".tiff"}: 
                fig.savefig(cfg["save_fig"], dpi=300, format="tiff", bbox_inches="tight") 
            else: 
                fig.savefig(cfg["save_fig"], dpi=300, bbox_inches="tight") 
            print(f"\nSaved figure to: {cfg['save_fig']}") 
        plt.show(block=True) 
        if cfg["show_hists"].lower() == "true": 
            print("Histograms skipped: no groups selected.") 
        if excel_writer is not None: 
            try: 
                cfg_df = pd.DataFrame(sorted(cfg.items()), columns=["key","value"]) 
                cfg_df.to_excel(excel_writer, sheet_name="config", index=False) 
                df_data.to_excel(excel_writer, sheet_name="raw_data", index=False) 
                excel_writer.close() 
                print(f"\nExported Excel workbook to: {xlsx_path} (sheets: {_excel_sheets_written + 2})") 
            except Exception as e: 
                print(f"\n[Excel] Failed to finalize workbook: {e}") 
        _deliver_hook(pairwise_results) 
        return 
    global_max = max([max(vals) for vals in data_plot.values()]) if any(len(v) for v in data_plot.values()) else 1.0 
    base_gap = base_gap_factor * global_max 
    stack_step= stack_step_factor * global_max 
    line_h = line_height_factor* global_max 
    x = np.arange(len(categories_plot)) 
    means = np.array([np.mean(data_plot[c]) for c in categories_plot], dtype=float) 
    sems = np.array([np.std(data_plot[c], ddof=1)/np.sqrt(len(data_plot[c])) if len(data_plot[c])>1 else 0 for c in categories_plot], dtype=float) 
    bar_tops = np.array([max(data_plot[c]) if len(data_plot[c]) > 0 else (means[i] if len(means)>i else 0) 
                         for i, c in enumerate(categories_plot)], dtype=float) 
    def _nice_number(xv: float) -> float: 
        if xv <= 0: return 1.0 
        exp = np.floor(np.log10(xv)); frac = xv / (10 ** exp) 
        if frac <= 1: nice = 1 
        elif frac <= 2: nice = 2 
        elif frac <= 5: nice = 5 
        else: nice = 10 
        return nice * (10 ** exp) 
    mode = cfg.get("bracket_mode", "Default ref only") 
    selected_csv_pairs = cfg.get("selected_pairs", "").strip() 
    selected_tokens = [s for s in (selected_csv_pairs.split(",") if selected_csv_pairs else []) if s] 
    include_nonsig_flag = (cfg.get("include_nonsig","False").lower() == "true") 
    if not only_plot: 
        if mode == "All significant" or mode == "All (ignore significance)": 
            all_tokens = [] 
            cats_sorted = sorted(categories_plot) 
            for i in range(len(cats_sorted)): 
                for j in range(i+1, len(cats_sorted)): 
                    all_tokens.append(f"{cats_sorted[i]}\n{cats_sorted[j]}") 
            require_sig = (mode == "All significant") 
            bracket_layout, required_ymax = compute_brackets_from_selection( 
                categories_plot, pairwise_results, alpha, x, bar_tops, 
                base_gap, stack_step, line_h, all_tokens, require_sig, sig_marker_mode, collision_k, top_margin_factor 
            ) 
        elif mode == "Custom…": 
            bracket_layout, required_ymax = compute_brackets_from_selection( 
                categories_plot, pairwise_results, alpha, x, bar_tops, 
                base_gap, stack_step, line_h, selected_tokens, not include_nonsig_flag, sig_marker_mode, collision_k, top_margin_factor 
            ) 
        else: 
            bracket_layout, required_ymax = compute_brackets_default_only( 
                categories_plot, pairwise_results, alpha, x, bar_tops, 
                base_gap, stack_step, line_h, default_ref, sig_marker_mode, collision_k, top_margin_factor 
            ) 
    else: 
        bracket_layout, required_ymax = [], max(bar_tops) * 1.05 if len(bar_tops) else 1.0 
    if len(bracket_layout) > 20: 
        bracket_layout = bracket_layout[:20] 
    if not only_plot and cfg["analysis"] == "t_one_sample" and single_annos: 
        if default_ref in categories_plot: 
            required_ymax = max(required_ymax, bar_tops[categories_plot.index(default_ref)] + base_gap + line_h) 
    tallest = float(np.nanmax(bar_tops)) if len(bar_tops) else 1.0 
    desired_step = max(1.0, tallest / 4.0) 
    step = _nice_number(desired_step) 
    step = max(1, int(round(step))) 
    if tallest > 4 * step: 
        step = max(1, int(_nice_number(tallest / 4.0 * 1.01))) 
    labeled_max = 4 * step 
    if y_min_user is not None and y_max_user is not None: 
        y_min = y_min_user 
        y_max = y_max_user 
    else: 
        y_min = 0.0 
        tallest_bar = float(np.nanmax(bar_tops)) if len(bar_tops) else 1.0 
        y_max = tallest_bar * 1.2 
        y_max = max(y_max, required_ymax) 
    plt.rcParams.update({"figure.dpi":160, "axes.spines.top":False, "axes.spines.right":False, "axes.linewidth":1.0}) 
    height_in = max(3.6, 3.4 + 0.28 * (0 if only_plot else max(0, min(len(bracket_layout), 20)))) 
    width_in = max(3.2, 1.2 + 0.37 * len(categories_plot)) 
    ptype = (cfg.get("plot_type") or "Bar + scatter").strip() 
    cross_graph = (cfg.get('cross_graph_brackets','False').lower() == 'False') 
    if cross_graph: 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width_in*2, height_in)) 
        draw_cross_graph_bracket(fig, ax1, ax2, 0, 0, y_max*0.9, 'p<0.05') 
    plt.ion() 
    fig, ax = plt.subplots(figsize=(width_in, height_in)) 
    def on_scroll(event): 
        if event.inaxes != ax: 
            return 
        if ptype == "Pie chart": 
            return 
        cur_ylim = ax.get_ylim() 
        y_min_c, y_max_c = cur_ylim 
        range_y = (y_max_c - y_min_c) 
        base_scale = 1.1 
        if event.button == 'up': 
            scale_factor = 1 / base_scale 
        elif event.button == 'down': 
            scale_factor = base_scale 
        else: 
            scale_factor = 1 
        new_range = range_y * scale_factor 
        center = event.ydata if event.ydata is not None else (y_min_c + y_max_c) / 2 
        new_ymin = center - new_range / 2 
        new_ymax = center + new_range / 2 
        ax.set_ylim(new_ymin, new_ymax) 
        ax.figure.canvas.draw_idle() 
    fig.canvas.mpl_connect('scroll_event', on_scroll) 
    series_factor = cfg.get("series_factor","").strip() 
    x_col = cfg.get("x_col","").strip() 
    series_means = None 
    series_colors = None 
    x_vec = None 
    if ptype in {"Lines (series)","Areas (series)","Regression (series)","Exponential regression (series)","Regression (global)","Exponential regression (global)"}: 
        x_vec = build_group_x(categories_plot, df_plot, x_col if x_col else None) 
        if ptype in {"Lines (series)","Areas (series)","Regression (series)"}: 
            series_means = build_series_means(categories_plot, df_plot, series_factor if series_factor else None) 
            series_levels_list = list(series_means.keys()) 
            series_colors = ensure_colors_for_keys(series_levels_list, {}) 
    if ptype == "Strip": 
        plot_strip(ax, categories_plot, data_plot, colors_cat) 
    elif ptype == "Mean ± CI": 
        n_vec = np.array([len(data_plot[c]) for c in categories_plot], dtype=float) 
        df_vec = np.maximum(n_vec - 1, 1) 
        tcrit = stats.t.ppf(1 - 0.05/2, df=df_vec) 
        ci_half = sems * tcrit 
        plot_mean_ci(ax, categories_plot, means, ci_half, colors_cat, bar=False, line=False, ci_label="95% CI") 
    elif ptype == "Line ± CI": 
        n_vec = np.array([len(data_plot[c]) for c in categories_plot], dtype=float) 
        df_vec = np.maximum(n_vec - 1, 1) 
        tcrit = stats.t.ppf(1 - 0.05/2, df=df_vec) 
        ci_half = sems * tcrit 
        plot_mean_ci(ax, categories_plot, means, ci_half, colors_cat, bar=False, line=True, ci_label="95% CI") 
    elif ptype == "Line (means)": 
        plot_line_means(ax, categories_plot, means) 
    elif ptype == "Boxplot": 
        plot_box(ax, categories_plot, data_plot, colors_cat) 
    elif ptype == "Violin": 
        plot_violin(ax, categories_plot, data_plot, colors_cat) 
    elif ptype == "Area (quartiles stacked)": 
        plot_area_quartiles(ax, categories_plot, data_plot) 
    elif ptype == "Lines (series)": 
        plot_multi_lines(ax, x_vec, categories_plot, series_means, series_colors) 
    elif ptype == "Areas (series)": 
        plot_multi_areas(ax, x_vec, categories_plot, series_means, series_colors) 
    elif ptype == "Regression (series)": 
        run_and_plot_regressions(ax, x_vec, categories_plot, series_means, series_colors) 
    elif ptype == "Regression (global)": 
        y_means_global = np.array([df_plot[df_plot["group"] == g]["value"].mean() for g in categories_plot], dtype=float) 
        run_and_plot_regression_global(ax, x_vec, categories_plot, y_means_global, color="#2F3B52") 
    elif ptype == "Pie chart": 
        plot_pie(ax, categories_plot, data_plot, colors_cat, value_mode="sum") 
    elif ptype == "Exponential regression (series)": 
        x_vec = build_group_x(categories_plot, df_plot, x_col if x_col else None) 
        series_means = build_series_means( 
            categories_plot, 
            df_plot, 
            series_factor if series_factor else None 
        ) 
        series_levels_list = list(series_means.keys()) 
        series_colors = ensure_colors_for_keys(series_levels_list, {}) 
        run_and_plot_exponential_regressions_series( 
            ax, 
            x_vec, 
            categories_plot, 
            series_means, 
            series_colors 
        ) 
    elif ptype == "Exponential regression (global)": 
        x_vec = build_group_x(categories_plot, df_plot, x_col if x_col else None) 
        y_means = np.array( 
            [df_plot[df_plot["group"] == g]["value"].mean() for g in categories_plot], 
            dtype=float 
        ) 
        run_and_plot_exponential_regression_global(ax, x_vec, y_means) 
    elif ptype == "Kaplan–Meier survival": 
        # Requires df_plot to contain 'time' and 'event' 
        plot_kaplan_meier(ax, df_plot) 
    else: 
        # Bar + scatter 
        bar_width = 0.70 
        for i, cat in enumerate(categories_plot): 
            ax.bar(x[i], means[i], width=bar_width, color=colors_cat.get(cat,"#999999"),
                   # edgecolor doubles as the hatch color; linewidth=0 keeps
                   # the bar outline invisible while the pattern is colored.
                   edgecolor=EDGE_COLORS.get(cat, "#444444"),
                   linewidth=0,
                   hatch=clean_hatch(HATCHES.get(cat, "")), alpha=0.80, zorder=1) 
        ax.errorbar(x, means, yerr=sems, fmt="None", ecolor="gray", elinewidth=1.5, capsize=6, capthick=1.5, zorder=2) 
        rng = np.random.default_rng(42) 
        for i, cat in enumerate(categories_plot): 
            yvals = np.array(data_plot[cat], dtype=float) 
            jitter = rng.normal(0, 0.04, size=len(yvals)) 
            ax.scatter(np.full_like(yvals, x[i], dtype=float) + jitter, yvals, 
                       s=28, color=colors_cat.get(cat,"#999999"), 
                       edgecolor="#444444", linewidths=0.6, alpha=1.0, zorder=3) 
        ax.set_xticks(x) 
        ax.set_xticklabels(categories_plot, rotation=45, ha="right", fontsize=8) 
    if ptype == "Pie chart": 
        ax.set_ylabel("") 
    else: 
        ax.set_ylabel(y_axis_label, fontsize=_yls_fs) 
    if not ptype.startswith("Regression") and ptype != "Pie chart": 
        ax.set_ylim(y_min, y_max) 
    tick_val = cfg.get("tick", "auto") 
    def _rebuild_y_ticks(y_min, y_max, tick_val, fallback_step): 
        import math 
        s = str(tick_val).strip().lower() if tick_val is not None else 'auto' 
        step = None 
        if s != 'auto': 
            try: 
                step = float(s) 
            except Exception: 
                step = None 
        if step is None or not np.isfinite(step) or step <= 0: 
            ticks = np.linspace(y_min, y_max, 5) 
            return np.asarray(ticks, dtype=float), None 
        k = math.floor((y_min + 1e-12) / step) 
        start = k * step 
        n_max = int(math.ceil((y_max - start) / step)) + 1 
        ticks = start + step * np.arange(n_max) 
        eps = step * 1e-6 
        ticks = ticks[(ticks >= y_min - eps) & (ticks <= y_max + eps)] 
        ticks = np.round(ticks, 6) 
        decimals = 0 if abs(step - round(step)) < 1e-9 else 1 
        return ticks, decimals 
    if ptype != "Pie chart": 
        ticks, decimals = _rebuild_y_ticks(y_min, y_max, tick_val, step) 
        ax.set_yticks(ticks) 
        ax.tick_params(axis="y", labelsize=_ytk_fs) 
        if decimals is None: 
            formatter = ScalarFormatter(useMathText=True) 
            formatter.set_scientific(False) 
            formatter.set_useOffset(False) 
            ax.yaxis.set_major_formatter(formatter) 
        else: 
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _ : f"{v:.{decimals}f}")) 
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
                ha="center", va="bottom", fontsize=_sig_fs, color="black", clip_on=False, zorder=4 
            ) 
    if (not only_plot and cfg["analysis"] == "t_one_sample" and single_annos and ptype != "Pie chart"): 
        if default_ref in categories_plot: 
            xi = categories_plot.index(default_ref); y_top = bar_tops[xi] + base_gap 
            for r in single_annos: 
                ax.text( 
                    x[xi], y_top + line_h, format_sig_marker(r["p_adj"], sig_marker_mode), 
                    ha="center", va="bottom", fontsize=_sig_fs, color="black", clip_on=False, zorder=4 
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
    plt.show(block=True) 
    if cfg["show_hists"].lower() == "true": 
        bins_spec = cfg["hist_bins"] 
        try: 
            max_h = int(float(cfg["hist_max"])) 
            max_h = min(30, max(1, max_h)) 
        except: max_h = 30 
        plot_histograms_panel(categories=categories_plot, data=data_plot, colors=colors_cat, max_groups=max_h, 
                              bins_spec=bins_spec, title=f"Histograms per group (max {max_h}, bins={bins_spec})") 
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
    _deliver_hook(pairwise_results) 
# ---------------------------------------------------------------------
# GUI (SCROLLABLE START MENU ADDED) 

# =====================================================================
#  Integrated spreadsheet editor (merged from spreadsheet_editor.py)
#  - SpreadsheetFrame: embeds directly in the main GUI (no popup)
#  - Browse Excel/CSV button loads the file straight into the grid
#  - "Apply" pushes data into data_dict / df_data / categories via
#    the on_apply callback, exactly like loading a file from disk.
# =====================================================================
def _sse_to_float(s: str):
    s = (s or "").strip().replace(",", ".")
    if s == "" or s.lower() in {"nan", "na", "n/a", "-"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _sse_fmt(v):
    try:
        f = float(v)
        if f.is_integer():
            return str(int(f))
        return f"{f:g}"
    except Exception:
        return str(v)


class SpreadsheetFrame(ttk.Frame):
    """GraphPad-style editable grid, embedded inside any parent frame."""
    DEFAULT_ROWS = 25
    DEFAULT_COLS = 4
    CELL_W = 12
    HDR_W = 14

    def __init__(self, master, on_apply=None, **kw):
        super().__init__(master, **kw)
        self.on_apply = on_apply
        self._headers = []
        self._cells = []
        self._header_entries = []
        self._cell_entries = []
        self._col_labels = []
        self._row_labels = []
        self._sel_cols = set()
        self._sel_rows = set()
        self._sel_anchor_col = None
        self._sel_anchor_row = None
        self._build_toolbar()
        self._build_grid_area()
        self._build_footer()
        self._init_blank(self.DEFAULT_ROWS, self.DEFAULT_COLS)
        self.bind_all("<Delete>", self._on_delete_key)
        self.bind_all("<Control-c>", self._on_copy_key)
        self.bind_all("<Command-c>", self._on_copy_key)

    def _build_toolbar(self):
        bar = ttk.Frame(self, padding=(6, 4))
        bar.pack(fill="x")
        ttk.Button(bar, text="Browse Excel/CSV…", command=self.browse_file).pack(side="left", padx=2)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(bar, text="Clear", command=self.clear_all).pack(side="left", padx=2)
        ttk.Button(bar, text="Export…", command=self.export_file).pack(side="left", padx=2)
        ttk.Button(bar, text="Apply ✓", command=self.apply).pack(side="right", padx=2)
        self._status_var = tk.StringVar(value="Click a column letter or row # to select. Shift/Ctrl-click to extend.")
        ttk.Label(bar, textvariable=self._status_var, foreground="#566275").pack(side="right", padx=10)

    def _build_grid_area(self):
        wrap = ttk.Frame(self, padding=(6, 0))
        wrap.pack(fill="both", expand=True)
        self._canvas = tk.Canvas(wrap, highlightthickness=0, background="#FFFFFF", height=260)
        vbar = ttk.Scrollbar(wrap, orient="vertical", command=self._canvas.yview)
        hbar = ttk.Scrollbar(wrap, orient="horizontal", command=self._canvas.xview)
        self._canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="we")
        wrap.grid_rowconfigure(0, weight=1)
        wrap.grid_columnconfigure(0, weight=1)
        self._inner = ttk.Frame(self._canvas)
        self._canvas.create_window((0, 0), window=self._inner, anchor="nw")
        self._inner.bind("<Configure>",
                         lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Enter>", lambda e: self._canvas.bind_all("<MouseWheel>", self._on_wheel))
        self._canvas.bind("<Leave>", lambda e: self._canvas.unbind_all("<MouseWheel>"))

    def _on_wheel(self, event):
        self._canvas.yview_scroll(int(-event.delta / 120), "units")

    def _build_footer(self):
        foot = ttk.Frame(self, padding=(6, 2))
        foot.pack(fill="x")
        ttk.Label(foot, text="Click column letter / row # to select. Shift-click = range, Ctrl/Cmd-click = toggle. "
                            "Delete = clear, Ctrl/Cmd+C = copy, +V = paste. Right-click = insert/delete.",
                  foreground="#566275").pack(side="left")

    def _init_blank(self, n_rows, n_cols):
        self._headers = [tk.StringVar(value=f"Group {i+1}") for i in range(n_cols)]
        self._cells = [[tk.StringVar(value="") for _ in range(n_cols)] for _ in range(n_rows)]
        self._sel_cols.clear(); self._sel_rows.clear()
        self._render_grid()

    @staticmethod
    def _col_letter(i):
        s = ""
        n = i
        while True:
            s = chr(ord("A") + (n % 26)) + s
            n = n // 26 - 1
            if n < 0:
                break
        return s

    def _render_grid(self):
        for w in self._inner.winfo_children():
            w.destroy()
        self._header_entries = []
        self._cell_entries = []
        self._col_labels = []
        self._row_labels = []
        corner = tk.Label(self._inner, text="◢", width=4, anchor="center",
                          bg="#DDE3EE", fg="#3A4252", borderwidth=1, relief="solid")
        corner.grid(row=0, column=0, sticky="nsew")
        corner.bind("<Button-1>", lambda e: self._select_all())
        for c in range(len(self._headers)):
            lbl = tk.Label(self._inner, text=self._col_letter(c), width=self.HDR_W,
                           anchor="center", bg="#DDE3EE", fg="#3A4252",
                           borderwidth=1, relief="solid", font=("TkDefaultFont", 9, "bold"))
            lbl.grid(row=0, column=c + 1, sticky="nsew")
            lbl.bind("<Button-1>", lambda e, ci=c: self._click_col(e, ci))
            lbl.bind("<Button-3>", lambda e, ci=c: self._col_menu(e, ci))
            lbl.bind("<Button-2>", lambda e, ci=c: self._col_menu(e, ci))
            self._col_labels.append(lbl)
        ttk.Label(self._inner, text="#", width=4, anchor="center",
                  background="#EEF2F8", foreground="#566275",
                  borderwidth=1, relief="solid").grid(row=1, column=0, sticky="nsew")
        for c, var in enumerate(self._headers):
            e = ttk.Entry(self._inner, textvariable=var, width=self.HDR_W, justify="center",
                          font=("TkDefaultFont", 10, "bold"))
            e.grid(row=1, column=c + 1, sticky="nsew")
            self._header_entries.append(e)
        for r, row_vars in enumerate(self._cells):
            rl = tk.Label(self._inner, text=str(r + 1), width=4, anchor="center",
                          bg="#F5F7FB", fg="#566275", borderwidth=1, relief="solid")
            rl.grid(row=r + 2, column=0, sticky="nsew")
            rl.bind("<Button-1>", lambda e, ri=r: self._click_row(e, ri))
            rl.bind("<Button-3>", lambda e, ri=r: self._row_menu(e, ri))
            rl.bind("<Button-2>", lambda e, ri=r: self._row_menu(e, ri))
            self._row_labels.append(rl)
            row_entries = []
            for c, v in enumerate(row_vars):
                e = tk.Entry(self._inner, textvariable=v, width=self.CELL_W, justify="right",
                             relief="solid", borderwidth=1, highlightthickness=0)
                e.grid(row=r + 2, column=c + 1, sticky="nsew")
                e.bind("<Button-1>", lambda ev, rr=r, cc=c: self._click_cell(rr, cc), add="+")
                e.bind("<Control-v>", self._on_paste_event)
                e.bind("<Command-v>", self._on_paste_event)
                e.bind("<Up>", lambda ev, rr=r, cc=c: self._move(rr - 1, cc))
                e.bind("<Down>", lambda ev, rr=r, cc=c: self._move(rr + 1, cc))
                e.bind("<Return>", lambda ev, rr=r, cc=c: self._move(rr + 1, cc))
                e.bind("<Left>", self._maybe_left)
                e.bind("<Right>", self._maybe_right)
                e.bind("<Button-3>", lambda ev, rr=r, cc=c: self._cell_menu(ev, rr, cc))
                e.bind("<Button-2>", lambda ev, rr=r, cc=c: self._cell_menu(ev, rr, cc))
                row_entries.append(e)
            self._cell_entries.append(row_entries)
        self._status_var.set(f"{len(self._cells)} rows × {len(self._headers)} columns")
        self._inner.update_idletasks()
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._apply_selection_styles()

    # --- selection ---
    SEL_HDR_BG = "#4A90E2"
    SEL_HDR_FG = "#FFFFFF"
    SEL_CELL_BG = "#DCE7F5"
    HDR_BG = "#DDE3EE"
    HDR_FG = "#3A4252"
    ROW_BG = "#F5F7FB"
    ROW_FG = "#566275"

    def _apply_selection_styles(self):
        for c, lbl in enumerate(self._col_labels):
            if c in self._sel_cols:
                lbl.configure(bg=self.SEL_HDR_BG, fg=self.SEL_HDR_FG)
            else:
                lbl.configure(bg=self.HDR_BG, fg=self.HDR_FG)
        for r, lbl in enumerate(self._row_labels):
            if r in self._sel_rows:
                lbl.configure(bg=self.SEL_HDR_BG, fg=self.SEL_HDR_FG)
            else:
                lbl.configure(bg=self.ROW_BG, fg=self.ROW_FG)
        for r, row in enumerate(self._cell_entries):
            for c, e in enumerate(row):
                selected = (c in self._sel_cols) or (r in self._sel_rows)
                try:
                    e.configure(bg=(self.SEL_CELL_BG if selected else "#FFFFFF"))
                except tk.TclError:
                    pass
        n = self._selection_count()
        if n:
            self._status_var.set(
                f"Selected: {len(self._sel_cols)} col(s), {len(self._sel_rows)} row(s) — {n} cell(s)"
            )

    def _selection_count(self):
        nrows, ncols = len(self._cells), len(self._headers)
        if not nrows or not ncols:
            return 0
        cells = set()
        for c in self._sel_cols:
            for r in range(nrows):
                cells.add((r, c))
        for r in self._sel_rows:
            for c in range(ncols):
                cells.add((r, c))
        return len(cells)

    def _clear_selection(self):
        self._sel_cols.clear()
        self._sel_rows.clear()
        self._sel_anchor_col = None
        self._sel_anchor_row = None

    def _select_all(self):
        self._sel_cols = set(range(len(self._headers)))
        self._sel_rows = set(range(len(self._cells)))
        self._apply_selection_styles()

    def _click_col(self, event, ci):
        shift = bool(event.state & 0x0001)
        ctrl = bool(event.state & 0x0004) or bool(event.state & 0x8)
        self._sel_rows.clear()
        if shift and self._sel_anchor_col is not None:
            lo, hi = sorted((self._sel_anchor_col, ci))
            self._sel_cols = set(range(lo, hi + 1))
        elif ctrl:
            if ci in self._sel_cols:
                self._sel_cols.discard(ci)
            else:
                self._sel_cols.add(ci)
                self._sel_anchor_col = ci
        else:
            self._sel_cols = {ci}
            self._sel_anchor_col = ci
        self._apply_selection_styles()

    def _click_row(self, event, ri):
        shift = bool(event.state & 0x0001)
        ctrl = bool(event.state & 0x0004) or bool(event.state & 0x8)
        self._sel_cols.clear()
        if shift and self._sel_anchor_row is not None:
            lo, hi = sorted((self._sel_anchor_row, ri))
            self._sel_rows = set(range(lo, hi + 1))
        elif ctrl:
            if ri in self._sel_rows:
                self._sel_rows.discard(ri)
            else:
                self._sel_rows.add(ri)
                self._sel_anchor_row = ri
        else:
            self._sel_rows = {ri}
            self._sel_anchor_row = ri
        self._apply_selection_styles()

    def _click_cell(self, r, c):
        if self._sel_cols or self._sel_rows:
            self._clear_selection()
            self._apply_selection_styles()

    def _on_delete_key(self, event):
        if not (self._sel_cols or self._sel_rows):
            return
        for c in self._sel_cols:
            for r in range(len(self._cells)):
                self._cells[r][c].set("")
        for r in self._sel_rows:
            for c in range(len(self._headers)):
                self._cells[r][c].set("")
        self._status_var.set("Cleared selection.")
        return "break"

    def _on_copy_key(self, event):
        if not (self._sel_cols or self._sel_rows):
            return
        if self._sel_cols and not self._sel_rows:
            cols = sorted(self._sel_cols)
            lines = ["\t".join(self._headers[c].get() for c in cols)]
            for r in range(len(self._cells)):
                lines.append("\t".join(self._cells[r][c].get() for c in cols))
        elif self._sel_rows and not self._sel_cols:
            rows = sorted(self._sel_rows)
            lines = ["\t".join(self._cells[r][c].get() for c in range(len(self._headers))) for r in rows]
        else:
            rows = sorted(self._sel_rows) if self._sel_rows else list(range(len(self._cells)))
            cols = sorted(self._sel_cols) if self._sel_cols else list(range(len(self._headers)))
            lines = ["\t".join(self._cells[r][c].get() for c in cols) for r in rows]
        text = "\n".join(lines)
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            self._status_var.set(f"Copied {len(lines)} line(s).")
        except Exception:
            pass
        return "break"

    def _move(self, r, c):
        if 0 <= r < len(self._cells) and 0 <= c < len(self._headers):
            e = self._cell_entries[r][c]
            e.focus_set()
            e.icursor("end")
        return "break"

    def _focus_pos(self, w):
        for r, row in enumerate(self._cell_entries):
            for c, e in enumerate(row):
                if e is w:
                    return r, c
        return None

    def _maybe_left(self, event):
        if event.widget.index("insert") == 0:
            p = self._focus_pos(event.widget)
            if p:
                return self._move(p[0], p[1] - 1)

    def _maybe_right(self, event):
        w = event.widget
        if w.index("insert") == len(w.get()):
            p = self._focus_pos(w)
            if p:
                return self._move(p[0], p[1] + 1)

    # --- context menus ---
    def _cell_menu(self, event, r, c):
        # If clicked cell isn't part of current selection, select its column
        if not (c in self._sel_cols or r in self._sel_rows):
            self._sel_rows.clear()
            self._sel_cols = {c}
            self._sel_anchor_col = c
            self._apply_selection_styles()
        m = tk.Menu(self, tearoff=0)
        if self._sel_cols:
            cols_lbl = ", ".join(self._col_letter(x) for x in sorted(self._sel_cols))
            m.add_command(label=f"Insert column left of {self._col_letter(c)}",
                          command=lambda: self.insert_column(c))
            m.add_command(label=f"Insert column right of {self._col_letter(c)}",
                          command=lambda: self.insert_column(c + 1))
            m.add_command(label=f"Delete column(s) {cols_lbl}",
                          command=self.delete_selected_columns)
            m.add_command(label=f"Clear column(s) {cols_lbl}",
                          command=self.clear_selected_columns)
        if self._sel_rows:
            if self._sel_cols:
                m.add_separator()
            rows_lbl = ", ".join(str(x + 1) for x in sorted(self._sel_rows))
            m.add_command(label=f"Insert row above {r+1}",
                          command=lambda: self.insert_row(r))
            m.add_command(label=f"Insert row below {r+1}",
                          command=lambda: self.insert_row(r + 1))
            m.add_command(label=f"Delete row(s) {rows_lbl}",
                          command=self.delete_selected_rows)
        m.add_separator()
        m.add_command(label="Clear selected cells", command=lambda: self._on_delete_key(None))
        m.tk_popup(event.x_root, event.y_root)
        return "break"

    def _col_menu(self, event, ci):
        if ci not in self._sel_cols:
            self._sel_rows.clear()
            self._sel_cols = {ci}
            self._sel_anchor_col = ci
            self._apply_selection_styles()
        sel = sorted(self._sel_cols)
        label = ", ".join(self._col_letter(c) for c in sel)
        m = tk.Menu(self, tearoff=0)
        m.add_command(label=f"Insert column left of {self._col_letter(ci)}",
                      command=lambda: self.insert_column(ci))
        m.add_command(label=f"Insert column right of {self._col_letter(ci)}",
                      command=lambda: self.insert_column(ci + 1))
        m.add_separator()
        m.add_command(label=f"Delete column(s) {label}", command=self.delete_selected_columns)
        m.add_command(label=f"Clear column(s) {label}", command=self.clear_selected_columns)
        m.tk_popup(event.x_root, event.y_root)

    def _row_menu(self, event, ri):
        if ri not in self._sel_rows:
            self._sel_cols.clear()
            self._sel_rows = {ri}
            self._sel_anchor_row = ri
            self._apply_selection_styles()
        sel = sorted(self._sel_rows)
        label = ", ".join(str(r + 1) for r in sel)
        m = tk.Menu(self, tearoff=0)
        m.add_command(label=f"Insert row above {ri+1}", command=lambda: self.insert_row(ri))
        m.add_command(label=f"Insert row below {ri+1}", command=lambda: self.insert_row(ri + 1))
        m.add_separator()
        m.add_command(label=f"Delete row(s) {label}", command=self.delete_selected_rows)
        m.tk_popup(event.x_root, event.y_root)

    # --- actions ---
    def add_column(self):
        self.insert_column(len(self._headers))

    def insert_column(self, idx):
        idx = max(0, min(idx, len(self._headers)))
        self._headers.insert(idx, tk.StringVar(value=f"Group {len(self._headers)+1}"))
        for row in self._cells:
            row.insert(idx, tk.StringVar(value=""))
        self._clear_selection()
        self._render_grid()

    def remove_column(self):
        if self._headers:
            self.delete_column(len(self._headers) - 1)

    def delete_column(self, idx):
        if not (0 <= idx < len(self._headers)) or len(self._headers) <= 1:
            return
        self._headers.pop(idx)
        for row in self._cells:
            row.pop(idx)
        self._render_grid()

    def delete_selected_columns(self):
        if not self._sel_cols:
            self.remove_column()
            return
        for idx in sorted(self._sel_cols, reverse=True):
            if len(self._headers) <= 1:
                break
            self._headers.pop(idx)
            for row in self._cells:
                row.pop(idx)
        self._clear_selection()
        self._render_grid()

    def clear_column(self, idx):
        if 0 <= idx < len(self._headers):
            for row in self._cells:
                row[idx].set("")

    def clear_selected_columns(self):
        for idx in self._sel_cols:
            self.clear_column(idx)

    def add_row(self):
        self.insert_row(len(self._cells))

    def insert_row(self, idx):
        idx = max(0, min(idx, len(self._cells)))
        self._cells.insert(idx, [tk.StringVar(value="") for _ in self._headers])
        self._clear_selection()
        self._render_grid()

    def remove_row(self):
        if self._cells:
            self.delete_row(len(self._cells) - 1)

    def delete_row(self, idx):
        if not (0 <= idx < len(self._cells)) or len(self._cells) <= 1:
            return
        self._cells.pop(idx)
        self._render_grid()

    def delete_selected_rows(self):
        if not self._sel_rows:
            self.remove_row()
            return
        for idx in sorted(self._sel_rows, reverse=True):
            if len(self._cells) <= 1:
                break
            self._cells.pop(idx)
        self._clear_selection()
        self._render_grid()


    def clear_all(self):
        if not messagebox.askyesno("Clear", "Clear all cells (keep group names)?", parent=self.winfo_toplevel()):
            return
        for row in self._cells:
            for v in row:
                v.set("")

    def _on_paste_event(self, event):
        try:
            text = self.clipboard_get()
        except Exception:
            return
        if "\n" not in text and "\t" not in text:
            return
        w = event.widget
        anchor = (0, 0)
        for r, row in enumerate(self._cell_entries):
            for c, e in enumerate(row):
                if e is w:
                    anchor = (r, c)
                    break
        ar, ac = anchor
        rows = [r for r in text.replace("\r", "").split("\n") if r != ""]
        sep = "\t" if any("\t" in r for r in rows) else ("," if any("," in r for r in rows) else None)
        parsed = [r.split(sep) if sep else [r] for r in rows]
        needed_cols = ac + max(len(r) for r in parsed)
        while len(self._headers) < needed_cols:
            self._headers.append(tk.StringVar(value=f"Group {len(self._headers)+1}"))
            for row in self._cells:
                row.append(tk.StringVar(value=""))
        needed_rows = ar + len(parsed)
        while len(self._cells) < needed_rows:
            self._cells.append([tk.StringVar(value="") for _ in self._headers])
        for i, row in enumerate(parsed):
            for j, val in enumerate(row):
                self._cells[ar + i][ac + j].set(val.strip())
        self._render_grid()
        self._status_var.set(f"Pasted {len(parsed)} × {max(len(r) for r in parsed)}")
        return "break"

    # --- file I/O ---
    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Open spreadsheet",
            filetypes=[("CSV/Excel", "*.csv *.tsv *.xlsx *.xls"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            if path.lower().endswith((".xlsx", ".xls")):
                sheet = _pick_excel_sheet_dialog(path)
                if sheet is None:
                    return
                df = pd.read_excel(path, sheet_name=sheet)
            elif path.lower().endswith(".tsv"):
                df = pd.read_csv(path, sep="\t")
            else:
                df = pd.read_csv(path)
                sheet = None
        except Exception as e:
            messagebox.showerror("Open", str(e))
            return
        dd = {str(c): [x for x in df[c].tolist() if pd.notna(x)] for c in df.columns}
        self.load_from_dict(dd)
        suffix = f" [{sheet}]" if sheet else ""
        self._status_var.set(f"Loaded {os.path.basename(path)}{suffix} — {len(df)} rows × {len(df.columns)} cols")

    def export_file(self):
        dd, df, _ = self.collect()
        if df.empty:
            messagebox.showinfo("Export", "Grid is empty.")
            return
        wide = pd.DataFrame({k: pd.Series(v) for k, v in dd.items()})
        path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv")])
        if not path:
            return
        try:
            if path.lower().endswith(".csv"):
                wide.to_csv(path, index=False)
            else:
                wide.to_excel(path, index=False)
        except Exception as e:
            messagebox.showerror("Export", str(e))
            return
        self._status_var.set(f"Saved {os.path.basename(path)}")

    def load_from_dict(self, dd):
        if not dd:
            self._init_blank(self.DEFAULT_ROWS, self.DEFAULT_COLS)
            return
        names = list(dd.keys())
        max_n = max((len(v) for v in dd.values()), default=0)
        n_rows = max(max_n, self.DEFAULT_ROWS)
        self._headers = [tk.StringVar(value=str(n)) for n in names]
        self._cells = []
        for r in range(n_rows):
            row = []
            for n in names:
                vals = dd[n]
                row.append(tk.StringVar(value=("" if r >= len(vals) or vals[r] is None
                                                else _sse_fmt(vals[r]))))
            self._cells.append(row)
        self._render_grid()

    def collect(self):
        names = [h.get().strip() or f"Group {i+1}" for i, h in enumerate(self._headers)]
        seen, uniq = {}, []
        for n in names:
            if n in seen:
                seen[n] += 1
                uniq.append(f"{n} ({seen[n]})")
            else:
                seen[n] = 1
                uniq.append(n)
        dd, rows_long = {}, []
        for c, name in enumerate(uniq):
            vals = []
            for r in range(len(self._cells)):
                v = _sse_to_float(self._cells[r][c].get())
                if v is not None:
                    vals.append(v)
                    rows_long.append({"group": name, "value": v})
            dd[name] = vals
        df = pd.DataFrame(rows_long, columns=["group", "value"])
        return dd, df, uniq

    def apply(self):
        try:
            dd, df, cats = self.collect()
        except Exception as e:
            messagebox.showerror("Apply", str(e))
            return
        if not any(len(v) > 0 for v in dd.values()):
            messagebox.showwarning("Apply", "No numeric values found.")
            return
        if self.on_apply:
            try:
                self.on_apply(dd, df, cats)
                self._status_var.set(f"Applied — {len(cats)} groups, "
                                     f"{sum(len(v) for v in dd.values())} values")
            except Exception as e:
                messagebox.showerror("Apply", str(e))


# Backward-compat: keep a Toplevel version in case some other code imports it
class SpreadsheetEditor(tk.Toplevel):
    def __init__(self, master, initial_data=None, on_apply=None,
                 title="Spreadsheet — edit data"):
        super().__init__(master)
        self.title(title)
        self.geometry("980x640")
        frame = SpreadsheetFrame(self, on_apply=lambda dd, df, cats: (
            on_apply(dd, df, cats) if on_apply else None, self.destroy()))
        frame.pack(fill="both", expand=True)
        if initial_data:
            frame.load_from_dict(initial_data)
        try:
            self.transient(master)
        except Exception:
            pass
# =====================================================================
#  End integrated spreadsheet editor
# =====================================================================


def open_config_gui(): 
    import platform 
    root = tk.Tk() 

    # ---- Apply Light Dust Blue theme (NEW) ----
    apply_light_dust_blue_theme(root)

    root.title("Analysis Setup") 
    root.geometry("1600x900") # Larger main window 
    root.resizable(True, True) 

    # SCROLLABLE WRAPPER 
    # Integrated results layout: controls on the left, results on the right
    global _RESULTS_PANEL
    _paned = ttk.PanedWindow(root, orient="horizontal")
    _paned.pack(fill="both", expand=True)
    outer = ttk.Frame(_paned)

    # GraphPad-style right pane: switch between Spreadsheet (data entry)
    # and Results (tables / figures from each analysis).
    right_nb = ttk.Notebook(_paned)
    spreadsheet_tab = ttk.Frame(right_nb, padding=(6, 6))
    results_tab = ttk.Frame(right_nb)
    right_nb.add(spreadsheet_tab, text="📊 Spreadsheet")
    right_nb.add(results_tab, text="📈 Results")

    _RESULTS_PANEL = ResultsPanel(results_tab)
    _RESULTS_PANEL.pack(fill="both", expand=True)
    # Auto-switch to Results whenever a new result tab is added.
    _RESULTS_PANEL.on_activity = lambda: right_nb.select(results_tab)

    _paned.add(outer, weight=3)
    _paned.add(right_nb, weight=4)
    right_nb.select(spreadsheet_tab)  # start on Spreadsheet

    # Set the canvas background to match theme (NEW)
    canvas = tk.Canvas(outer, highlightthickness=0, background="#B7C9E2")
    vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vscroll.set)
    canvas.pack(side="left", fill="both", expand=True)
    vscroll.pack(side="right", fill="y")

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



    # -- Tool opener: Multiple regression --
    def open_multiple_regression():
        # Build a groups callback consistent with the current UI state 
        def _groups_cb():
            groups_all = data_dict or {}
            if hide_deselected_plot.get():
                chosen = [k for k, v in selected_groups.items() if v.get()]
            else:
                chosen = list(groups_all.keys())
            out = {}
            for k in chosen:
                arr = np.asarray(groups_all.get(k, []), dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    out[k] = arr
            return out
        try:
            MultipleRegressionWindow(
                master=root,
                get_groups_callback=_groups_cb, # use the local callback 
                title="Multiple regression"
            )
        except Exception as ex:
            messagebox.showerror("Multiple regression", f"Could not open window:\n{ex}")

    # -- Tool opener: Power Analysis (define before button) --
    def _open_power_analysis():
        def _groups_cb():
            # Use loaded data_dict; optionally restrict to selected groups 
            groups_all = data_dict or {}
            if hide_deselected_plot.get():
                chosen = [k for k, v in selected_groups.items() if v.get()]
            else:
                chosen = list(groups_all.keys())
            # Convert to clean numpy vectors and drop empties 
            out = {}
            for k in chosen:
                arr = np.asarray(groups_all.get(k, []), dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    out[k] = arr
            return out
        PowerAnalysisWindow(master=root, get_groups_callback=_groups_cb)

    # -- Mousewheel helpers for scroll --
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

    # -- Top toolbar (Canvas, Calculator, Power analysis, Multiple regression, etc.) --
    def build_toolbar(parent):
        bar = ttk.Frame(parent)
        bar.grid(row=0, column=0, columnspan=6, sticky="we", pady=(0, 8))
        for c in range(6):
            parent.grid_columnconfigure(c, weight=1)

        # --- Logo dropdown menu (sole entry point for toolbar actions) ---
        logo_menu = tk.Menu(bar, tearoff=0)

        logo_menu.add_command(label="Canvas", command=lambda: InteractivePlane(root))
        logo_menu.add_command(label="Calculator", command=lambda: launch_calculator(master=None))
        logo_menu.add_command(label="Power analysis…", command=_open_power_analysis)
        logo_menu.add_command(label="Multiple regression", command=open_multiple_regression)
        logo_menu.add_command(label="Categorical tests", command=lambda: CategoricalTestsWindow(master=root))
        logo_menu.add_command(label="Parametric tests", command=lambda: ParametricTestsWindow(master=root, title="Parametric tests"))
        logo_menu.add_command(label="Survival (KM)", command=lambda: SurvivalAnalysisWindow(master=root))
        logo_menu.add_command(label="Chemical Calculator", command=launch_chemical_calculator)

        def _show_logo_menu(event):
            w = event.widget
            logo_menu.post(w.winfo_rootx(), w.winfo_rooty() + w.winfo_height())

        try:
            logo_big = tk.PhotoImage(file="lq_logo.png")
            logo_small = logo_big.subsample(12, 12)
            logo_lbl = ttk.Label(bar, image=logo_small, cursor="hand2")
            logo_lbl.image = logo_small
            logo_lbl.pack(side="left", padx=(0, 7))
        except Exception as e:
            print("Toolbar logo could not be loaded:", e)
            logo_lbl = ttk.Label(bar, text="☰ Menu", cursor="hand2")
            logo_lbl.pack(side="left", padx=(0, 7))

        logo_lbl.bind("<Button-1>", _show_logo_menu)

        return bar
    toolbar = build_toolbar(main) # place at the top 

    # Main layout/content 
    data_dict: Dict[str, List[float]] = {} 
    df_data = pd.DataFrame(columns=["group", "value"]) 
    categories: List[str] = [] 
    def default_ref(cats: List[str]) -> str: 
        return "-dox" if "-dox" in cats else (cats[0] if cats else "") 

    # Vars 
    csv_path = tk.StringVar(value="") 
    analysis = tk.StringVar(value="ANOVA") 
    posthoc = tk.StringVar(value="None") 
    correction= tk.StringVar(value="None") 
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
    export_xlsx= tk.StringVar(value="") 
    y_label_var = tk.StringVar(value=(load_y_label_history()[0] if load_y_label_history() else "Value")) 
    y_history = load_y_label_history() 
    y_min_var = tk.StringVar(value="") 
    y_max_var = tk.StringVar(value="") 
    current_colors: Dict[str, str] = {} 
    current_hatches: Dict[str, str] = {}        # per-group infill pattern
    current_hatch_colors: Dict[str, str] = {}   # per-group pattern (edge) color
    selected_groups: Dict[str, tk.BooleanVar] = {} 
    factor_a_var = tk.StringVar(value="") 
    factor_b_var = tk.StringVar(value="") 
    series_factor_var = tk.StringVar(value="") 
    x_col_var = tk.StringVar(value="") 
    sig_marker_mode = tk.StringVar(value="p-value") 
    # NEW: font-size controls
    y_label_fontsize_var = tk.StringVar(value="15")
    y_tick_fontsize_var  = tk.StringVar(value="10")
    sig_fontsize_var     = tk.StringVar(value="10")
    adv_show = tk.BooleanVar(value=False) 
    base_gap_factor_var = tk.StringVar(value="0.08") 
    stack_step_factor_var = tk.StringVar(value="0.15") 
    line_height_factor_var= tk.StringVar(value="0.025") 
    collision_k_var = tk.StringVar(value="0.75") 
    top_margin_factor_var = tk.StringVar(value="0.12") 
    show_hists = tk.BooleanVar(value=False) 
    hist_bins = tk.StringVar(value="auto") 
    hist_max = tk.StringVar(value="30") 
    bracket_mode = tk.StringVar(value="Default ref only") 
    selected_pairs_var= tk.StringVar(value="") 
    include_nonsig = tk.BooleanVar(value=False) 
    plot_type = tk.StringVar(value="Bar + scatter") 
    interpret_wide_var = tk.BooleanVar(value=False) 
    prev_bracket_mode = {"value": "Default ref only"} 
    display_to_token: Dict[str, str] = {} 
    token_to_display: Dict[str, str] = {} 

    # Capture last results for automation (global option 2) 
    last_results = {"pairwise_results": [], "alpha": 0.05} 
    def _result_hook(res): 
        try: 
            last_results["pairwise_results"] = res.get("pairwise_results", []) or [] 
            last_results["alpha"] = float(res.get("alpha", alpha.get())) 
        except Exception: 
            pass 
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

    # Start form rows BELOW the toolbar
    r = 1
    main.grid_columnconfigure(0, weight=1)

    # =========================================================
    # Spreadsheet tab (GraphPad-style): empty placeholder with one
    # Import button until a file is loaded; the editor itself only
    # appears after data exists.
    # =========================================================
    sheet_container = ttk.Frame(spreadsheet_tab)
    sheet_container.pack(fill="both", expand=True)

    placeholder = ttk.Frame(sheet_container, padding=(40, 60))
    placeholder.pack(fill="both", expand=True)
    ttk.Label(placeholder,
              text="No data loaded.",
              font=("TkDefaultFont", 14, "bold")).pack(pady=(0, 6))
    ttk.Label(placeholder,
              text="Import a CSV or Excel file to start editing your groups.\n"
                   "Each column becomes one group in the analysis.",
              justify="center",
              foreground="#566275").pack(pady=(0, 18))

    def _apply_from_sheet(dd, df, cats):
        nonlocal data_dict, df_data, categories
        data_dict, df_data = dd, df
        categories = list(data_dict.keys())
        csv_path.set("<from spreadsheet>")
        refresh_groups()
        rebuild_color_rows()

    # The real editor is created on first import so it isn't visible empty.
    embedded_sheet = None

    def _ensure_editor():
        nonlocal embedded_sheet
        if embedded_sheet is not None:
            return embedded_sheet
        placeholder.pack_forget()
        embedded_sheet = SpreadsheetFrame(sheet_container, on_apply=_apply_from_sheet)
        embedded_sheet.pack(fill="both", expand=True)
        return embedded_sheet

    def import_into_spreadsheet():
        path = filedialog.askopenfilename(
            title="Import CSV / Excel",
            filetypes=[
                ("CSV or Excel", "*.csv *.tsv *.xlsx *.xls"),
                ("CSV", "*.csv"),
                ("Excel", "*.xlsx *.xls"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext in {".xlsx", ".xls"}:
                sheet = _pick_excel_sheet_dialog(path)
                if sheet is None:
                    return
                df = pd.read_excel(path, sheet_name=sheet)
            elif ext == ".tsv":
                df = pd.read_csv(path, sep="\t")
            else:
                df = pd.read_csv(path)
        except Exception as e:
            print("Import error:\n", traceback.format_exc())
            messagebox.showerror("Import", f"{type(e).__name__}: {e}")
            return
        dd = {str(c): [x for x in df[c].tolist() if pd.notna(x)] for c in df.columns}
        editor = _ensure_editor()
        editor.load_from_dict(dd)
        csv_path.set(path)
        right_nb.select(spreadsheet_tab)

    ttk.Button(placeholder, text="Import CSV / Excel…",
               command=import_into_spreadsheet).pack()

    # Mirror import button in the left form toolbar area so it's reachable
    # without leaving the Setup view.

    # --- end integrated spreadsheet panel -------------------------------

    # =========================================================
    # Form is split into labelled sections so widgets don't collide.
    # Each section owns its own internal 4-column grid.
    # =========================================================

    def _section(title):
        nonlocal r
        f = ttk.LabelFrame(main, text=title, padding=(8, 6))
        f.grid(row=r, column=0, columnspan=5, sticky="we", pady=(6, 0))
        for c in (1, 3):
            f.grid_columnconfigure(c, weight=1)
        r += 1
        return f

    # NEW: Top Run button (easy to find)
    _run_holder = {"cb": lambda: None}
    top_run = ttk.Frame(main)
    top_run.grid(row=r, column=0, columnspan=5, sticky="we", pady=(4, 0))
    ttk.Button(top_run, text="▶  Run analysis", command=lambda: _run_holder["cb"]()).pack(side="left", padx=4, pady=2)
    r += 1

    # ---- Analysis -------------------------------------------------------
    sec_analysis = _section("Analysis")
    ttk.Label(sec_analysis, text="Primary:").grid(row=0, column=0, sticky="w")
    analysis_cb = ttk.Combobox(
        sec_analysis, textvariable=analysis,
        values=[
            "None",
            "Normality (all)", "ANOVA", "ANOVA (two-way)", "Kruskal",
            "Mann–Whitney (2 groups)", "t_ind_equal", "t_ind_welch",
            "t_paired", "Wilcoxon (paired)", "t_one_sample", "Friedman",
            "Descriptives"
        ],
        width=24, state="readonly"
    )
    analysis_cb.grid(row=0, column=1, sticky="we", padx=(4, 12), pady=2)
    ttk.Label(sec_analysis, text="Post-hoc:").grid(row=0, column=2, sticky="w")
    posthoc_cb = ttk.Combobox(sec_analysis, textvariable=posthoc, values=["None"], width=24, state="readonly")
    posthoc_cb.grid(row=0, column=3, sticky="we", padx=(4, 0), pady=2)

    ttk.Label(sec_analysis, text="Comparison scope:").grid(row=1, column=0, sticky="w")
    scope_cb = ttk.Combobox(sec_analysis, textvariable=scope, values=["vs_ref", "all_pairs"], width=24, state="readonly")
    scope_cb.grid(row=1, column=1, sticky="we", padx=(4, 12), pady=2)
    ttk.Label(sec_analysis, text="Correction:").grid(row=1, column=2, sticky="w")
    ttk.Combobox(sec_analysis, textvariable=correction, values=["None", "bonferroni", "holm", "bh"],
                 width=24, state="readonly").grid(row=1, column=3, sticky="we", padx=(4, 0), pady=2)

    ttk.Label(sec_analysis, text="Significance marker:").grid(row=2, column=0, sticky="w")
    ttk.Combobox(sec_analysis, textvariable=sig_marker_mode, values=["asterisks", "p-value"],
                 width=24, state="readonly").grid(row=2, column=1, sticky="we", padx=(4, 12), pady=2)
    ttk.Label(sec_analysis, text="Alpha:").grid(row=2, column=2, sticky="w")
    ttk.Entry(sec_analysis, textvariable=alpha, width=10).grid(row=2, column=3, sticky="w", padx=(4, 0), pady=2)

    ttk.Label(sec_analysis, text="CI level (0-1):").grid(row=3, column=0, sticky="w")
    ttk.Entry(sec_analysis, textvariable=ci_level, width=10).grid(row=3, column=1, sticky="w", padx=(4, 12), pady=2)
    ttk.Label(sec_analysis, text="One-sample μ:").grid(row=3, column=2, sticky="w")
    ttk.Entry(sec_analysis, textvariable=mu_var, width=10).grid(row=3, column=3, sticky="w", padx=(4, 0), pady=2)
    # NEW: font size controls
    ttk.Label(sec_analysis, text="Y label font size:").grid(row=4, column=0, sticky="w")
    ttk.Entry(sec_analysis, textvariable=y_label_fontsize_var, width=10).grid(row=4, column=1, sticky="w", padx=(4, 12), pady=2)
    ttk.Label(sec_analysis, text="Y tick font size:").grid(row=4, column=2, sticky="w")
    ttk.Entry(sec_analysis, textvariable=y_tick_fontsize_var, width=10).grid(row=4, column=3, sticky="w", padx=(4, 0), pady=2)
    ttk.Label(sec_analysis, text="p-value / asterisk font size:").grid(row=5, column=0, sticky="w")
    ttk.Entry(sec_analysis, textvariable=sig_fontsize_var, width=10).grid(row=5, column=1, sticky="w", padx=(4, 12), pady=2)

    # ---- Plot -----------------------------------------------------------
    sec_plot = _section("Plot")
    ttk.Label(sec_plot, text="Plot type:").grid(row=0, column=0, sticky="w")
    ttk.Combobox(
        sec_plot, textvariable=plot_type, width=24, state="readonly",
        values=[
            "None",
            "Bar + scatter", "Strip",
            "Mean ± CI", "Line ± CI", "Line (means)",
            "Boxplot", "Violin",
            "Area (quartiles stacked)",
            "Lines (series)", "Areas (series)",
            "Regression (series)",
            "Exponential regression (series)",
            "Regression (global)",
            "Exponential regression (global)",
            "Pie chart"
        ]
    ).grid(row=0, column=1, sticky="we", padx=(4, 12), pady=2)
    ttk.Label(sec_plot, text="Y-axis label:").grid(row=0, column=2, sticky="w")
    y_label_cb = ttk.Combobox(sec_plot, textvariable=y_label_var, values=y_history, width=24)
    y_label_cb.grid(row=0, column=3, sticky="we", padx=(4, 0), pady=2)

    ttk.Label(sec_plot, text="Series factor:").grid(row=1, column=0, sticky="w")
    series_factor_cb = ttk.Combobox(sec_plot, textvariable=series_factor_var, values=[], width=24, state="readonly")
    series_factor_cb.grid(row=1, column=1, sticky="we", padx=(4, 12), pady=2)
    ttk.Label(sec_plot, text="X column (numeric):").grid(row=1, column=2, sticky="w")
    x_col_cb = ttk.Combobox(sec_plot, textvariable=x_col_var, values=[], width=24, state="readonly")
    x_col_cb.grid(row=1, column=3, sticky="we", padx=(4, 0), pady=2)

    ttk.Label(sec_plot, text="Y min:").grid(row=2, column=0, sticky="w")
    ttk.Entry(sec_plot, textvariable=y_min_var, width=10).grid(row=2, column=1, sticky="w", padx=(4, 12), pady=2)
    ttk.Label(sec_plot, text="Y max:").grid(row=2, column=2, sticky="w")
    ttk.Entry(sec_plot, textvariable=y_max_var, width=10).grid(row=2, column=3, sticky="w", padx=(4, 0), pady=2)

    ttk.Label(sec_plot, text="Y tick step:").grid(row=3, column=0, sticky="w")
    ttk.Entry(sec_plot, textvariable=tick, width=10).grid(row=3, column=1, sticky="w", padx=(4, 12), pady=2)

    # ---- References & factors ------------------------------------------
    sec_refs = _section("References & factors")
    ttk.Label(sec_refs, text="Default ref (brackets):").grid(row=0, column=0, sticky="w")
    default_ref_cb = ttk.Combobox(sec_refs, textvariable=default_ref_var, values=[], width=24, state="readonly")
    default_ref_cb.grid(row=0, column=1, sticky="we", padx=(4, 12), pady=2)
    ttk.Label(sec_refs, text="Analysis ref:").grid(row=0, column=2, sticky="w")
    ref_cb = ttk.Combobox(sec_refs, textvariable=ref_var, values=[], width=24, state="readonly")
    ref_cb.grid(row=0, column=3, sticky="we", padx=(4, 0), pady=2)

    ttk.Label(sec_refs, text="Exclude (vs_ref):").grid(row=1, column=0, sticky="w")
    exclude_cb = ttk.Combobox(sec_refs, textvariable=exclude_var, values=[], width=24, state="readonly")
    exclude_cb.grid(row=1, column=1, sticky="we", padx=(4, 12), pady=2)

    ttk.Label(sec_refs, text="Factor A (two-way):").grid(row=2, column=0, sticky="w")
    factor_a_cb = ttk.Combobox(sec_refs, textvariable=factor_a_var, values=[], width=24, state="readonly")
    factor_a_cb.grid(row=2, column=1, sticky="we", padx=(4, 12), pady=2)
    ttk.Label(sec_refs, text="Factor B (two-way):").grid(row=2, column=2, sticky="w")
    factor_b_cb = ttk.Combobox(sec_refs, textvariable=factor_b_var, values=[], width=24, state="readonly")
    factor_b_cb.grid(row=2, column=3, sticky="we", padx=(4, 0), pady=2)

    subj_label = ttk.Label(sec_refs, text="Subject column: ✗ (paired/RM disabled)")
    subj_label.grid(row=3, column=0, columnspan=4, sticky="w", pady=(4, 0))

    # ---- Brackets -------------------------------------------------------
    sec_brk = _section("Brackets")
    ttk.Label(sec_brk, text="Bracket source:").grid(row=0, column=0, sticky="w")
    bracket_mode_cb = ttk.Combobox(
        sec_brk, textvariable=bracket_mode,
        values=["Default ref only", "All significant", "All (ignore significance)", "Custom…"],
        width=24, state="readonly"
    )
    bracket_mode_cb.grid(row=0, column=1, sticky="we", padx=(4, 0), pady=2)

    # Custom-pairs panel lives on its OWN row inside the section (no overlap).
    custom_pairs_frame = ttk.LabelFrame(sec_brk, text="Bracket pairs (Custom)", padding=(6, 4))
    custom_pairs_frame.grid(row=1, column=0, columnspan=5, sticky="nsew", pady=(6, 0))
    ttk.Label(custom_pairs_frame, text="Pair:").grid(row=0, column=0, sticky="w")
    pair_combo = ttk.Combobox(custom_pairs_frame, values=[], state="readonly")
    pair_combo.grid(row=0, column=1, sticky="ew", padx=(4, 0))
    add_btn = ttk.Button(custom_pairs_frame, text="Add", width=7)
    add_btn.grid(row=0, column=2, sticky="w", padx=(4, 0))
    ttk.Label(custom_pairs_frame, text="Selected:").grid(row=1, column=0, sticky="nw", pady=(6, 0))
    selected_list = tk.Listbox(custom_pairs_frame, height=8, exportselection=False)
    selected_list.grid(row=1, column=1, rowspan=2, sticky="nsew", pady=(6, 0))
    remove_btn = ttk.Button(custom_pairs_frame, text="Remove", width=7)
    remove_btn.grid(row=1, column=2, sticky="nw", padx=(4, 0), pady=(6, 0))
    clear_btn = ttk.Button(custom_pairs_frame, text="Clear", width=7)
    clear_btn.grid(row=2, column=2, sticky="nw", padx=(4, 0), pady=(2, 0))
    include_nonsig_chk = ttk.Checkbutton(custom_pairs_frame, text="Include non-significant (ignore α)",
                                         variable=include_nonsig)
    include_nonsig_chk.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
    done_btn = ttk.Button(custom_pairs_frame, text="Done", width=7)
    done_btn.grid(row=3, column=2, sticky="e", padx=(4, 0), pady=(6, 0))
    custom_pairs_frame.grid_columnconfigure(1, weight=1)
    custom_pairs_frame.grid_rowconfigure(1, weight=1)
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
        selected_pairs_var.set("".join([tok + ("," if i < len(current)-1 else "") for i, tok in enumerate(current)])) 
        selected_list.delete(0, "end") 
        for tok in current: 
            selected_list.insert("end", token_to_display.get(tok, tok)) 
        y_label_cb['values'] = load_y_label_history() 
        # Ensure there is a BooleanVar for each category (default True) 
        for g in categories: 
            if g not in selected_groups: 
                selected_groups[g] = tk.BooleanVar(value=True) 
        # Try to refresh ordering list if built 
        try: 
            refresh_order_list() 
        except Exception: 
            pass 
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
            if _HAS_SCPH: opts.append("Games–Howell") 
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
    colors_frame = ttk.LabelFrame(main, text="Group colors", padding=(8,6)) 
    colors_frame.grid(row=r, column=0, columnspan=5, sticky="we", pady=(6,0))
    r += 1 
    colors_rows = ttk.Frame(colors_frame) 
    colors_rows.grid(row=0, column=0, sticky="we") 
    colors_frame.grid_columnconfigure(0, weight=1) 
    color_buttons: Dict[str, ttk.Button] = {} 
    # NEW: hide deselected checkbox (affects plotting only) 
    hide_deselected_plot = tk.BooleanVar(value=True) 
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
    def pick_hatch_color_for(group: str):
        cur = current_hatch_colors.get(group, "") or "#000000"
        _, hex_ = colorchooser.askcolor(color=cur, title=f"Choose pattern color for {group}")
        if hex_:
            current_hatch_colors[group] = hex_.upper()
            rebuild_color_rows()
    # Patterns offered to the user. Empty = solid (no infill pattern).
    PATTERN_CHOICES = ["(none)", "/", "//", "\\", "\\\\", "x", "xx", ".", "..",
                       "+", "++", "o", "O", "*", "|", "-"]
    def on_hatch_changed(group: str, value: str):
        current_hatches[group] = "" if value == "(none)" else value
    def rebuild_color_rows(): 
        for w in colors_rows.winfo_children(): 
            w.destroy() 
        color_buttons.clear() 
        # Header row
        ttk.Label(colors_rows, text="", width=2).grid(row=0, column=0)
        ttk.Label(colors_rows, text="Group").grid(row=0, column=1, sticky="w", padx=(0,6))
        ttk.Label(colors_rows, text="Color").grid(row=0, column=2, sticky="w", padx=(0,6))
        ttk.Label(colors_rows, text="Pattern").grid(row=0, column=3, sticky="w", padx=(0,6))
        ttk.Label(colors_rows, text="Pattern color").grid(row=0, column=4, sticky="w", padx=(0,6))
        for i, g in enumerate(categories): 
            row = i + 1
            # checkbox for selection 
            if g not in selected_groups: 
                selected_groups[g] = tk.BooleanVar(value=True) 
            chk = ttk.Checkbutton(colors_rows, variable=selected_groups[g], text="") 
            chk.grid(row=row, column=0, sticky="w", padx=(0,6), pady=2) 
            # label 
            ttk.Label(colors_rows, text=g).grid(row=row, column=1, sticky="w", padx=(0,6), pady=2) 
            # color button 
            hex_color = current_colors.get(g, "") 
            if not (isinstance(hex_color, str) and hex_color.startswith("#") and len(hex_color) in (4, 7)): 
                hex_color = "" 
            btn_text = f"{g}: {hex_color or '(auto)'}" 
            btn = ttk.Button(colors_rows, text=btn_text, command=lambda gr=g: pick_color_for(gr)) 
            btn.grid(row=row, column=2, sticky="w", padx=(0,6), pady=2) 
            color_buttons[g] = btn 
            # pattern (hatch) combobox
            cur_h = current_hatches.get(g, "")
            cur_h_label = cur_h if cur_h else "(none)"
            if cur_h_label not in PATTERN_CHOICES:
                cur_h_label = "(none)"
            h_var = tk.StringVar(value=cur_h_label)
            cb = ttk.Combobox(colors_rows, textvariable=h_var, values=PATTERN_CHOICES,
                              state="readonly", width=8)
            cb.grid(row=row, column=3, sticky="w", padx=(0,6), pady=2)
            cb.bind("<<ComboboxSelected>>",
                    lambda _e, gr=g, var=h_var: on_hatch_changed(gr, var.get()))
            # pattern color button
            hc = current_hatch_colors.get(g, "")
            hc_text = f"{hc}" if hc else "(auto)"
            ttk.Button(colors_rows, text=hc_text,
                       command=lambda gr=g: pick_hatch_color_for(gr)
                       ).grid(row=row, column=4, sticky="w", padx=(0,6), pady=2)
    toolbar_frame = ttk.Frame(colors_frame) 
    toolbar_frame.grid(row=1, column=0, sticky="we", pady=(6,0)) 
    def clear_all_colors(): 
        current_colors.clear() 
        current_hatches.clear()
        current_hatch_colors.clear()
        rebuild_color_rows() 
    def set_default_palette():
        # Fill colors only; leave patterns as the user set them (default = none).
        auto = ensure_colors_for_keys(categories, {})
        current_colors.clear()
        current_colors.update(auto)
        rebuild_color_rows()

    def select_all_groups(): 
        for g in selected_groups: 
            selected_groups[g].set(True) 
        rebuild_color_rows() 
    def select_only_significant(): 
        prs = last_results.get("pairwise_results", []) or [] 
        try: 
            a = float(last_results.get("alpha", alpha.get())) 
        except Exception: 
            a = 0.05 
        if not prs: 
            messagebox.showinfo("Select only significant", "No pairwise results available from the last run.\nRun an analysis with post-hoc first.") 
            return 
        def pval_of(r): 
            p = r.get("p_adj", r.get("p_raw")) 
            try: 
                return float(p) 
            except Exception: 
                return 1.0 
        keep = set() 
        for r in prs: # Always global (Option 2) 
            if pval_of(r) <= a: 
                if r.get("ref"): keep.add(str(r["ref"])) 
                if r.get("group"): keep.add(str(r["group"])) 
        if not keep: 
            messagebox.showinfo("Select only significant", f"No significant groups found at α={a}.") 
            return 
        # Toggle selections 
        for g in categories: 
            if g not in selected_groups: 
                selected_groups[g] = tk.BooleanVar(value=True) 
            selected_groups[g].set(g in keep) 
        rebuild_color_rows() 
    ttk.Button(toolbar_frame, text="Fill with palette", command=set_default_palette).grid(row=0, column=0, padx=(0,6))
    ttk.Checkbutton(toolbar_frame, text="Hide deselected groups in plots", variable=hide_deselected_plot).grid(row=0, column=2, padx=(12,0)) 
    ttk.Button(toolbar_frame, text="Select all groups", command=select_all_groups).grid(row=0, column=4, padx=(12,0)) 
    rebuild_color_rows() 

    # -- Group ordering (rebuilt) --
    group_order_var = tk.StringVar(value="") 
    order_frame = ttk.LabelFrame(main, text="Group ordering", padding=(8, 6)) 
    order_frame.grid(row=r, column=0, columnspan=5, sticky="we", pady=(6, 0))
    r += 1 
    # Listbox with current groups 
    order_list = tk.Listbox(order_frame, height=10, exportselection=False) 
    order_list.grid(row=0, column=0, rowspan=6, sticky="nsew") 
    order_frame.grid_columnconfigure(0, weight=1) 
    order_frame.grid_rowconfigure(0, weight=1) 
    # -- Helpers -- 
    def _listbox_items(): 
        return [order_list.get(i) for i in range(order_list.size())] 
    def apply_from_listbox(): 
        """Write the current visible listbox order to CSV field.""" 
        group_order_var.set(",".join(_listbox_items())) 
    def apply_csv_to_listbox(): 
        """Load CSV field into the listbox; append any new categories at the end.""" 
        csv_raw = (group_order_var.get() or "").strip() 
        if not csv_raw: 
            order_list.delete(0, "end") 
            for g in categories: 
                order_list.insert("end", g) 
            return 
        wanted = [g.strip() for g in csv_raw.split(",") if g.strip()] 
        final = [g for g in wanted if g in categories] + [g for g in categories if g not in wanted] 
        order_list.delete(0, "end") 
        for g in final: 
            order_list.insert("end", g) 
    def refresh_order_list(): 
        """Refresh the listbox. If CSV has an order, honor it first.""" 
        if (group_order_var.get() or "").strip(): 
            apply_csv_to_listbox() 
        else: 
            order_list.delete(0, "end") 
            for g in categories: 
                order_list.insert("end", g) 
    def move_selected(delta: int): 
        """Move the selected row up/down by delta (+1 or -1).""" 
        sel = order_list.curselection() 
        if not sel: 
            return 
        i = int(sel[0]) 
        j = i + delta 
        if j < 0 or j >= order_list.size(): 
            return 
        # Swap items i and j 
        a = order_list.get(i) 
        b = order_list.get(j) 
        order_list.delete(j) 
        order_list.insert(j, a) 
        order_list.delete(i) 
        order_list.insert(i, b) 
        order_list.selection_clear(0, "end") 
        order_list.selection_set(j) 
        order_list.activate(j) 
        # Keep CSV in sync with visible list 
        apply_from_listbox() 
    # Buttons column: Up/Down 
    btns_col = ttk.Frame(order_frame) 
    btns_col.grid(row=0, column=1, sticky="n", padx=(6, 0)) 
    ttk.Button(btns_col, text="↑ Move up", command=lambda: move_selected(-1)).pack(fill="x", pady=(0, 4)) 
    ttk.Button(btns_col, text="↓ Move down", command=lambda: move_selected(+1)).pack(fill="x") 
    # CSV controls 
    csv_col = ttk.Frame(order_frame) 
    csv_col.grid(row=0, column=2, sticky="nwe", padx=(12, 0)) 
    csv_col.grid_columnconfigure(0, weight=1) 
    ttk.Label(csv_col, text="Group order (CSV)").grid(row=0, column=0, sticky="w") 
    csv_entry = ttk.Entry(csv_col, textvariable=group_order_var, width=37) 
    csv_entry.grid(row=1, column=0, sticky="we", pady=(2, 4)) 
    
    ttk.Button(csv_col, text="Load list", command=apply_csv_to_listbox).grid(row=3, column=0, sticky="w") 
    # Optional helpers: sort by mean (based on data_dict) 
    def sort_by_mean(reverse: bool = False): 
        try: 
            means = [(g, float(np.mean(np.asarray(data_dict.get(g, []), dtype=float)))) for g in categories] 
            means_sorted = sorted(means, key=lambda t: (np.isnan(t[1]), t[1]), reverse=reverse) 
            group_order_var.set(",".join([g for g, _ in means_sorted])) 
            apply_csv_to_listbox() 
        except Exception: 
            messagebox.showwarning("Sort by mean", "Could not compute means for sorting.") 
    ttk.Button(csv_col, text="Sort by mean ↑", command=lambda: sort_by_mean(False)).grid(row=4, column=0, sticky="w", pady=(6, 0)) 
    ttk.Button(csv_col, text="Sort by mean ↓", command=lambda: sort_by_mean(True)).grid(row=5, column=0, sticky="w") 
    # Initial fill 
    refresh_order_list() 

    # -- Run / action buttons row --
    btns = ttk.Frame(main) 
    btns.grid(row=r, column=0, columnspan=4, pady=(6,0), sticky="e") 
    def on_run(): 
        save_y_label_to_history(y_label_var.get()) 
        y_label_cb['values'] = load_y_label_history() 
        # Sync listbox → CSV one more time right before building cfg 
        try: 
            apply_from_listbox() 
        except Exception: 
            pass 
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
        if y_min_var.get().strip() != "" and not _is_float(y_min_var.get().strip()): 
            messagebox.showerror("Y-axis", "Y min must be numeric (or leave blank)."); return 
        if y_max_var.get().strip() != "" and not _is_float(y_max_var.get().strip()): 
            messagebox.showerror("Y-axis", "Y max must be numeric (or leave blank)."); return 
        try: 
            hmax = int(float(hist_max.get())) 
            if hmax < 1 or hmax > 30: raise ValueError 
        except Exception: 
            messagebox.showerror("Histograms", "Max histograms must be an integer between 1 and 30."); return 
        # Build selected groups CSV (for plotting hide feature) 
        selected_csv_groups = ",".join([g for g in categories if selected_groups.get(g, tk.BooleanVar(value=True)).get()]) 
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
            "line_height_factor":line_height_factor_var.get().strip(), 
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
            "colors_json": json.dumps(ensure_colors_for_keys(categories, current_colors)), 
            "hatches_json": json.dumps(dict(current_hatches)),
            "hatch_colors_json": json.dumps(dict(current_hatch_colors)),
            "plot_type": plot_type.get().strip(), 
            "ci_level": ci_level.get().strip(), 
            "sig_marker_mode": sig_marker_mode.get().strip(), 
            "y_label_fontsize": y_label_fontsize_var.get().strip(), 
            "y_tick_fontsize":  y_tick_fontsize_var.get().strip(), 
            "sig_fontsize":     sig_fontsize_var.get().strip(), 
            "y_min": y_min_var.get().strip(), 
            "y_max": y_max_var.get().strip(), 
            "group_order": group_order_var.get().strip(), # <— order comes from the mover 
            # NEW 
            "selected_groups": selected_csv_groups, 
            "hide_deselected_plot": "True" if hide_deselected_plot.get() else "False", 
        } 
        execute_analysis(cfg, data_dict, df_data, result_hook=_result_hook) 
    def on_close(): 
        root.destroy() 
    _run_holder["cb"] = on_run

    root.mainloop() 
# Main 
def main(): 
    open_config_gui() 
if __name__ == "__main__": 
    main() 
# === Exponential Regression Additions === 
def run_and_plot_exponential_regression(ax, x_vec, y_vec, color="#2F3B52"): 
    x = np.asarray(x_vec, dtype=float) 
    y = np.asarray(y_vec, dtype=float) 
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0) 
    if mask.sum() < 2: 
        ax.text(0.5,0.5,"Not enough positive values",transform=ax.transAxes,ha='center',va='center') 
        return ax 
    logy = np.log(y[mask]) 
    slope, intercept, r, p, stderr = stats.linregress(x[mask], logy) 
    a = np.exp(intercept); b = slope; r2 = r*r 
    xfit = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 200) 
    yfit = a*np.exp(b*xfit) 
    ax.plot(xfit, yfit, color=color, lw=2, label=f"Exp fit: y={a:.3g}e^({b:.3g}x) R2={r2:.3f}") 
    return ax 
def run_and_plot_exponential_regressions(ax, x_vec, categories, series_means, colors): 
    for series, y in series_means.items(): 
        x = np.asarray(x_vec); y = np.asarray(y) 
        mask = np.isfinite(x) & np.isfinite(y) & (y>0) 
        if mask.sum()<2: continue 
        logy = np.log(y[mask]) 
        slope, intercept, r, p, stderr = stats.linregress(x[mask], logy) 
        a = np.exp(intercept); b = slope; r2 = r*r 
        xfit = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 200) 
        yfit = a*np.exp(b*xfit) 
        col = colors.get(series,"#2F3B52") 
        ax.plot(xfit, yfit, color=col, lw=2, label=f"{series} Exp R2={r2:.3f}") 
    ax.legend(loc='best') 
    return ax
