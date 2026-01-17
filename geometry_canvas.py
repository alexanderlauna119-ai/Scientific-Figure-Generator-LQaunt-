
# geometry_canvas.py
import tkinter as tk
from tkinter import ttk
import numpy as np
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.text import Text
from collections import OrderedDict
import os
import csv

class InteractivePlane(tk.Toplevel):
    # ------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------
    def __init__(self, master=None, title="Geometry Canvas"):
        super().__init__(master)
        self.title(title)

        # --- Open in full window / maximized (cross-platform best effort)
        try:
            self.state("zoomed")  # Windows/macOS (where supported)
        except Exception:
            try:
                self.attributes("-zoomed", True)  # Many Linux builds
            except Exception:
                self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")

        # --- FIGURE
        self.fig, self.ax = plt.subplots(figsize=(9.5, 7))
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, linestyle="--", alpha=0.3)
        self.ax.axhline(0, color="#444")
        self.ax.axvline(0, color="#444")
        # NEW: keep x/y limits fixed when new artists are added
        self.ax.set_autoscale_on(False)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side="left", fill="both", expand=True)

        # --- SIDEBAR (OBJECT INSPECTOR)
        self.sidebar = ttk.Frame(self)
        self.sidebar.pack(side="right", fill="y")
        ttk.Label(
            self.sidebar, text="Object Inspector",
            font=("Segoe UI", 12, "bold")
        ).pack(pady=4, anchor="w")

        # TreeView (with its own scrollbar)
        self.tree_frame = ttk.Frame(self.sidebar)
        self.tree_frame.pack(fill="x")
        self.tree = ttk.Treeview(
            self.tree_frame,
            columns=("type", "summary"),
            show="headings",
            height=18
        )
        self.tree.heading("type", text="Type")
        self.tree.heading("summary", text="Summary")
        self.tree.column("type", width=90, anchor="w")
        self.tree.column("summary", width=260, anchor="w")
        self.tree_scroll = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.tree_scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree_scroll.grid(row=0, column=1, sticky="ns")
        self.tree_frame.columnconfigure(0, weight=1)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        ttk.Label(
            self.sidebar, text="Properties",
            font=("Segoe UI", 11, "bold")
        ).pack(pady=(10, 4), anchor="w")

        # --- SCROLLABLE PROPERTY GRID (Canvas + inner Frame + Scrollbar)
        self.prop_container = ttk.Frame(self.sidebar)
        self.prop_container.pack(fill="both", expand=True)
        self.prop_canvas = tk.Canvas(self.prop_container, borderwidth=0, highlightthickness=0)
        self.prop_vsb = ttk.Scrollbar(self.prop_container, orient="vertical", command=self.prop_canvas.yview)
        self.prop_canvas.configure(yscrollcommand=self.prop_vsb.set)
        self.prop_inner = ttk.Frame(self.prop_canvas)
        self.prop_inner.bind(
            "<Configure>",
            lambda e: self.prop_canvas.configure(scrollregion=self.prop_canvas.bbox("all"))
        )
        self.prop_canvas.create_window((0, 0), window=self.prop_inner, anchor="nw")
        self.prop_canvas.grid(row=0, column=0, sticky="nsew")
        self.prop_vsb.grid(row=0, column=1, sticky="ns")
        self.prop_container.rowconfigure(0, weight=1)
        self.prop_container.columnconfigure(0, weight=1)

        # Optional: mouse wheel scrolling over property grid
        self.prop_canvas.bind("<Enter>", lambda e: self._bind_prop_mousewheel(True))
        self.prop_canvas.bind("<Leave>", lambda e: self._bind_prop_mousewheel(False))

        self.prop_entries = {}          # for numeric/text props
        self.prop_color_controls = {}   # for color props

        # --- STATUS BAR
        self.status = tk.StringVar(
            value="Left-click: point \nRight-click: text \nWheel: zoom \nSpace: pan \nShift: snap"
        )
        ttk.Label(self, textvariable=self.status).pack(fill="x")

        # --- INTERNAL STORAGE
        self._points = []
        self._labels = []
        self._segments = []
        self._polygons = []
        self._curves = []  # regression/function curves (Line2D)
        self._selected = []
        self._last_styles = {}
        self._drag = None
        self._snapping = False
        self._panning = False
        self._pan_start = None

        # modes & helpers
        self._connect_mode = False
        self._pending_point = None
        self._polygon_mode = False
        self._polygon_pts = []
        self._poly_preview = None  # Line2D used for polygon rubber-band
        self._line_mode = False
        self._line_start = None
        self._line_preview = None  # Line2D used for line rubber-band
        self._point_mode = False   # Click-to-place simple point
        self._text_mode = False    # Add Text (mode)
        self._selection_mode = False
        self._sel_rect = None
        self._sel_start = None

        # --- REGISTRY FOR INSPECTOR
        self.object_counter = {
            "point": 0,
            "segment": 0,
            "circle": 0,
            "rectangle": 0,
            "polygon": 0,
            "text": 0,
            "curve": 0,
        }
        self.registry = {}  # id → {type, ref, props}

        # --- COLOR PALETTE
        self.COLOR_PALETTE = OrderedDict([
            ("tab:blue", "#1f77b4"),
            ("tab:orange", "#ff7f0e"),
            ("tab:green", "#2ca02c"),
            ("tab:red", "#d62728"),
            ("tab:purple", "#9467bd"),
            ("tab:brown", "#8c564b"),
            ("tab:pink", "#e377c2"),
            ("tab:gray", "#7f7f7f"),
            ("tab:olive", "#bcbd22"),
            ("tab:cyan", "#17becf"),
            ("black", "#000000"),
            ("white", "#ffffff"),
            ("gold", "#ffd700"),
            ("dodgerblue", "#1e90ff"),
            ("limegreen", "#32cd32"),
            ("crimson", "#dc143c"),
            ("indigo", "#4b0082"),
            ("teal", "#008080"),
        ])

        # --- EVENT CONNECTIONS
        cid = self.canvas.mpl_connect
        cid("button_press_event", self._on_press)
        cid("button_release_event", self._on_release)
        cid("motion_notify_event", self._on_motion)
        cid("scroll_event", self._on_scroll)
        cid("key_press_event", self._on_key)

        # Keep point sizes constant: update on each draw
        self.fig.canvas.mpl_connect("draw_event", lambda e: self._update_point_radii())

        self._build_menu()

    # ------------------------------------------------------------
    # SCROLL UTILS
    # ------------------------------------------------------------
    def _bind_prop_mousewheel(self, bind):
        if bind:
            self.prop_canvas.bind_all("<MouseWheel>", self._on_prop_mousewheel)  # Windows/macOS
            self.prop_canvas.bind_all("<Button-4>", self._on_prop_mousewheel_linux)  # Linux
            self.prop_canvas.bind_all("<Button-5>", self._on_prop_mousewheel_linux)
        else:
            self.prop_canvas.unbind_all("<MouseWheel>")
            self.prop_canvas.unbind_all("<Button-4>")
            self.prop_canvas.unbind_all("<Button-5>")

    def _on_prop_mousewheel(self, event):
        if event.delta:
            self.prop_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_prop_mousewheel_linux(self, event):
        if event.num == 4:
            self.prop_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.prop_canvas.yview_scroll(1, "units")

    # ------------------------------------------------------------
    # SIDEBAR REGISTRY METHODS
    # ------------------------------------------------------------
    def _new_id(self, kind):
        self.object_counter[kind] += 1
        return f"{kind[:2].upper()}{self.object_counter[kind]}"

    def _register_object(self, oid, kind, ref, props):
        self.registry[oid] = {"type": kind, "ref": ref, "props": props}
        self._refresh_object_list()

    def _make_summary(self, oid):
        d = self.registry[oid]
        k = d["type"]
        p = d["props"]
        if k == "point":
            return f"({p['x']:.2f}, {p['y']:.2f})"
        if k == "segment":
            return f"({p['x1']:.2f},{p['y1']:.2f}) → ({p['x2']:.2f},{p['y2']:.2f})"
        if k == "circle":
            return f"center=({p['x']:.2f},{p['y']:.2f}) r={p['r']:.2f}"
        if k == "rectangle":
            return f"x={p['x']}, y={p['y']}, w={p['w']}, h={p['h']}"
        if k == "polygon":
            return f"{len(p['points'])} pts"
        if k == "text":
            return p["text"]
        if k == "curve":
            return p.get("info", p.get("model", "curve"))
        return ""

    def _refresh_object_list(self):
        selected = self.tree.selection()
        keep_sel = selected[0] if selected else None
        for row in self.tree.get_children():
            self.tree.delete(row)
        for oid, data in self.registry.items():
            self.tree.insert("", "end", iid=oid, values=(data["type"], self._make_summary(oid)))
        if keep_sel and keep_sel in self.registry:
            self.tree.selection_set(keep_sel)

    # ------------------------------------------------------------
    # COLOR PALETTE UI (scrollable pop-up)
    # ------------------------------------------------------------
    def _ask_color_from_palette(self, title="Choose color", initial=None):
        top = tk.Toplevel(self)
        top.title(title)
        top.transient(self)
        top.grab_set()

        outer = ttk.Frame(top)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_config(_):
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=bbox)
            canvas.itemconfig(inner_id, width=canvas.winfo_width())

        inner.bind("<Configure>", _on_inner_config)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(inner_id, width=canvas.winfo_width()))

        chosen = {"hex": None}

        def to_hex(val):
            if not val:
                return None
            if val.startswith("#") and len(val) in (4, 7):
                return val
            return self.COLOR_PALETTE.get(val, None)

        def choose(val):
            chosen["hex"] = to_hex(val) if not val.startswith("#") else val
            top.destroy()

        pad, row, col, max_cols = 6, 0, 0, 6
        for name, hexv in self.COLOR_PALETTE.items():
            cell = tk.Frame(inner, width=30, height=24, bg=hexv, relief="ridge", bd=1, cursor="hand2")
            cell.grid(row=row, column=col, padx=pad, pady=pad)
            lbl = tk.Label(inner, text=name, font=("Segoe UI", 8))
            lbl.grid(row=row + 1, column=col, padx=pad, pady=(0, pad))
            cell.bind("<Button-1>", lambda e, val=hexv: choose(val))
            lbl.bind("<Button-1>", lambda e, val=hexv: choose(val))
            col += 1
            if col >= max_cols:
                col = 0
                row += 2

        def _bind_wheel(bind: bool):
            if bind:
                canvas.bind_all("<MouseWheel>", _on_mousewheel)
                canvas.bind_all("<Button-4>", _on_mousewheel_linux)
                canvas.bind_all("<Button-5>", _on_mousewheel_linux)
            else:
                canvas.unbind_all("<MouseWheel>")
                canvas.unbind_all("<Button-4>")
                canvas.unbind_all("<Button-5>")

        def _on_mousewheel(event):
            if event.delta:
                canvas.yview_scroll(int(-event.delta / 120), "units")

        def _on_mousewheel_linux(event):
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")

        canvas.bind("<Enter>", lambda e: _bind_wheel(True))
        canvas.bind("<Leave>", lambda e: _bind_wheel(False))

        top.geometry("520x420")
        self.update_idletasks()
        try:
            x = self.winfo_rootx() + (self.winfo_width() // 2) - (top.winfo_reqwidth() // 2)
            y = self.winfo_rooty() + (self.winfo_height() // 2) - (top.winfo_reqheight() // 2)
            top.geometry(f"+{x}+{y}")
        except Exception:
            pass

        top.wait_window()
        return chosen["hex"]

    def _make_color_row(self, parent, label_text, initial_hex, apply_callback):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label_text, width=16).pack(side="left")
        swatch = tk.Label(row, width=4, relief="groove", bg=initial_hex if initial_hex else "#ffffff")
        swatch.pack(side="left", padx=(0, 6))

        def pick():
            chosen = self._ask_color_from_palette(title=f"Choose {label_text}", initial=initial_hex)
            if chosen:
                swatch.config(bg=chosen)
                apply_callback(chosen)

        ttk.Button(row, text="Choose…", command=pick).pack(side="left")
        return {"swatch": swatch}

    # ------------------------------------------------------------
    # PROPERTY GRID HANDLING
    # ------------------------------------------------------------
    def _on_tree_select(self, event):
        items = self.tree.selection()
        if not items:
            return
        oid = items[0]
        data = self.registry[oid]
        props = data["props"]

        for w in self.prop_inner.winfo_children():
            w.destroy()
        self.prop_entries.clear()
        self.prop_color_controls.clear()

        ttk.Label(
            self.prop_inner, text=f"{data['type'].upper()} Properties",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 6))

        color_keys_by_kind = {
            "point": ("facecolor", "edgecolor"),
            "segment": ("color",),
            "circle": ("edgecolor",),
            "rectangle": ("edgecolor",),
            "polygon": ("facecolor", "edgecolor"),
            "curve": ("color",),
        }
        color_keys = color_keys_by_kind.get(data["type"], ())

        for key, val in props.items():
            if key in color_keys:
                def make_apply(k):
                    def apply_color(hexv):
                        props[k] = hexv
                        self._apply_properties(data["type"], data["ref"], props)
                        self._refresh_object_list()
                        self.canvas.draw_idle()
                    return apply_color

                ctrl = self._make_color_row(self.prop_inner, key, str(val), make_apply(key))
                self.prop_color_controls[key] = ctrl

            elif key in ("lw",):
                row = ttk.Frame(self.prop_inner)
                row.pack(fill="x", pady=2)
                ttk.Label(row, text=key, width=15).pack(side="left")
                e = ttk.Entry(row)
                e.pack(side="left", fill="x", expand=True)
                e.insert(0, str(val))
                e.bind("<KeyRelease>", lambda ev, oid=oid, k=key: self._live_update(oid, k))
                self.prop_entries[key] = e
            else:
                if key in ("info", "model"):
                    row = ttk.Frame(self.prop_inner); row.pack(fill="x", pady=2)
                    ttk.Label(row, text=f"{key}: ", width=10).pack(side="left")
                    ttk.Label(row, text=str(val)).pack(side="left", fill="x", expand=True)
                else:
                    row = ttk.Frame(self.prop_inner)
                    row.pack(fill="x", pady=2)
                    ttk.Label(row, text=key, width=15).pack(side="left")
                    e = ttk.Entry(row)
                    e.pack(side="left", fill="x", expand=True)
                    e.insert(0, str(val))
                    e.bind("<KeyRelease>", lambda ev, oid=oid, k=key: self._live_update(oid, k))
                    self.prop_entries[key] = e

        self._highlight_from_tree(oid)

    def _highlight_from_tree(self, oid):
        self._clear_selection()
        obj = self.registry[oid]["ref"]
        self._selected = [obj]
        self._apply_selection_style(obj)
        self.canvas.draw_idle()

    def _live_update(self, oid, key):
        entry = self.prop_entries[key]
        raw = entry.get()
        obj = self.registry[oid]
        props = obj["props"]
        ref = obj["ref"]
        if key in ("lw",):
            try:
                val = float(raw)
                props[key] = val
            except:
                props[key] = raw
        else:
            props[key] = raw

        self._apply_properties(obj["type"], ref, props)
        self._refresh_object_list()
        self.canvas.draw_idle()

    def _apply_properties(self, kind, ref, p):
        if kind == "point":
            ref.center = (p["x"], p["y"])
            if "facecolor" in p:
                try: ref.set_facecolor(p["facecolor"])
                except Exception: pass
            if "edgecolor" in p:
                try: ref.set_edgecolor(p["edgecolor"])
                except Exception: pass
            if "px_radius" in p:
                try: ref._px_radius = float(p["px_radius"])
                except Exception: pass
            self._update_point_radii()
            self._update_segments()

        elif kind == "segment":
            ref.set_data([p["x1"], p["x2"]], [p["y1"], p["y2"]])
            if "color" in p:
                try: ref.set_color(p["color"])
                except Exception: pass
            if "lw" in p:
                try: ref.set_linewidth(float(p["lw"]))
                except Exception: pass

        elif kind == "circle":
            ref.center = (p["x"], p["y"])
            ref.set_radius(p["r"])
            if "edgecolor" in p:
                try: ref.set_edgecolor(p["edgecolor"])
                except Exception: pass
            if "lw" in p:
                try: ref.set_linewidth(float(p["lw"]))
                except Exception: pass

        elif kind == "rectangle":
            ref.set_x(p["x"]); ref.set_y(p["y"])
            ref.set_width(p["w"]); ref.set_height(p["h"])
            if "edgecolor" in p:
                try: ref.set_edgecolor(p["edgecolor"])
                except Exception: pass
            if "lw" in p:
                try: ref.set_linewidth(float(p["lw"]))
                except Exception: pass

        elif kind == "polygon":
            pts = p["points"]
            ref.set_xy(pts)
            if "facecolor" in p:
                try: ref.set_facecolor(p["facecolor"])
                except Exception: pass
            if "edgecolor" in p:
                try: ref.set_edgecolor(p["edgecolor"])
                except Exception: pass
            if "alpha" in p:
                try: ref.set_alpha(float(p["alpha"]))
                except Exception: pass
            if "lw" in p:
                try: ref.set_linewidth(float(p["lw"]))
                except Exception: pass

        elif kind == "curve":
            if "color" in p:
                try: ref.set_color(p["color"])
                except Exception: pass
            if "lw" in p:
                try: ref.set_linewidth(float(p["lw"]))
                except Exception: pass

        elif kind == "text":
            ref.set_position((p["x"], p["y"]))
            ref.set_text(p["text"])

    # ------------------------------------------------------------
    # PIXEL-STABLE POINT SIZE HELPERS
    # ------------------------------------------------------------
    def _pixels_to_data_radius(self, px):
        bb = self.ax.get_window_extent()
        w_px, h_px = bb.width, bb.height
        (x1, x2) = self.ax.get_xlim()
        (y1, y2) = self.ax.get_ylim()
        dx_per_px = (x2 - x1) / max(w_px, 1)
        dy_per_px = (y2 - y1) / max(h_px, 1)
        return 0.5 * (dx_per_px + dy_per_px) * px

    def _update_point_radii(self):
        for p in self._points:
            px = getattr(p, "_px_radius", 6)
            try:
                p.set_radius(self._pixels_to_data_radius(px))
            except Exception:
                pass

    # ------------------------------------------------------------
    # MENU CREATION
    # ------------------------------------------------------------
    def _export(self, ext):
        from tkinter import filedialog
        file = filedialog.asksaveasfilename(defaultextension=f'.{ext}', filetypes=[(ext.upper(), f'*.{ext}')])
        if not file:
            return
        self.fig.savefig(file, dpi=300, bbox_inches='tight')
        self.status.set(f'Exported: {file}')

    def _recolor_selected(self):
        if not self._selected:
            self.status.set("Select an object first.")
            return
        chosen = self._ask_color_from_palette(title="Choose color")
        if not chosen:
            return
        a = self._selected[0]
        found_oid = None
        for oid, data in self.registry.items():
            if data["ref"] is a:
                found_oid = oid
                break
        if not found_oid:
            return

        data = self.registry[found_oid]
        t = data["type"]
        if t in ("segment", "curve"):
            data["props"]["color"] = chosen
            try: a.set_color(chosen)
            except Exception: pass
        elif t == "point":
            data["props"]["facecolor"] = chosen
            try: a.set_facecolor(chosen)
            except Exception: pass
        elif t in ("circle", "rectangle", "polygon"):
            data["props"]["edgecolor"] = chosen
            try: a.set_edgecolor(chosen)
            except Exception: pass

        self._refresh_object_list()
        self.canvas.draw_idle()

    def _build_menu(self):
        from tkinter import filedialog
        menu = tk.Menu(self)
        self.config(menu=menu)

        # FILE MENU
        mf_file = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label='File', menu=mf_file)
        mf_file.add_command(label='Export as PNG', command=lambda: self._export('png'))
        mf_file.add_command(label='Export as JPEG', command=lambda: self._export('jpg'))
        mf_file.add_command(label='Export as TIFF', command=lambda: self._export('tif'))
        mf_file.add_separator()
        mf_file.add_command(label='Import Points (CSV/Excel)…', command=self._import_points_dialog)

        # EDIT MENU
        me = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Edit", menu=me)
        me.add_command(label="Delete Selected", command=self._delete_selected)
        me.add_command(label="Recolor Selected…", command=self._recolor_selected)

        # SHAPES MENU
        ms = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Shapes", menu=ms)
        ms.add_command(label="Add Point by Coordinates…", command=self._dialog_add_point)
        ms.add_command(label="Add Line (Length + Angle)…", command=self._dialog_add_line)
        ms.add_command(label="Add Circle…", command=self._dialog_circle)
        ms.add_command(label="Add Rectangle…", command=self._dialog_rectangle)
        ms.add_command(label="Draw Polygon (mode)", command=self._toggle_polygon_mode)
        ms.add_command(label="Draw Line (mode)", command=self._toggle_line_mode)
        ms.add_command(label="Plot Point (mode)", command=self._toggle_point_mode)
        ms.add_command(label="Add Text (mode)", command=self._toggle_text_mode)
        ms.add_separator()
        ms.add_command(label="Selection Mode (toggle)", command=self._toggle_selection_mode)
        ms.add_separator()
        ms.add_command(label="Connect Points (toggle)", command=self._toggle_connect)
        ms.add_command(label="Clear Segments", command=self._clear_segments)
        ms.add_command(label="Clear Polygons", command=self._clear_polygons)

        # ANALYSIS / REGRESSION MENU
        ma = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Analysis", menu=ma)

        # Basic regressions
        mr_basic = tk.Menu(ma, tearoff=0)
        ma.add_cascade(label="Regression (basic)", menu=mr_basic)
        mr_basic.add_command(label="Linear (y = a + b·x)", command=self._regression_linear)
        mr_basic.add_command(label="Polynomial…", command=self._regression_polynomial)
        mr_basic.add_command(label="Exponential (y = a·e^{b·x})", command=self._regression_exponential)
        mr_basic.add_command(label="Logarithmic (y = a + b·ln x)", command=self._regression_logarithmic)
        mr_basic.add_command(label="Power-law (y = a·x^b)", command=self._regression_power)

        # sklearn-based regressions (optional)
        mr_ml = tk.Menu(ma, tearoff=0)
        ma.add_cascade(label="Regression (scikit-learn)", menu=mr_ml)
        mr_ml.add_command(label="Ridge", command=lambda: self._regression_sklearn("ridge"))
        mr_ml.add_command(label="Lasso", command=lambda: self._regression_sklearn("lasso"))
        mr_ml.add_command(label="ElasticNet", command=lambda: self._regression_sklearn("elasticnet"))
        mr_ml.add_separator()
        mr_ml.add_command(label="SVR (RBF)", command=lambda: self._regression_sklearn("svr"))
        mr_ml.add_command(label="Decision Tree", command=lambda: self._regression_sklearn("tree"))
        mr_ml.add_command(label="Random Forest", command=lambda: self._regression_sklearn("forest"))
        mr_ml.add_command(label="KNN", command=lambda: self._regression_sklearn("knn"))

        ma.add_separator()
        ma.add_command(label="Run Regression by Name(s)…", command=self._regression_by_names_prompt)
        ma.add_command(label="Clear Fits", command=self._clear_curves)
        ma.add_separator()
        ma.add_command(label="Measure Segment Length", command=self._measure_segment_length)
        ma.add_command(label="Compute Polygon Area", command=self._polygon_area)

        # FUNCTIONS MENU (plot typed f(x))
        mf = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Functions", menu=mf)
        mf.add_command(label="Plot Function…", command=self._plot_function_dialog)

    # ------------------------------------------------------------
    # PROMPT DIALOG (generic)
    # ------------------------------------------------------------
    def _prompt(self, title, fields):
        win = tk.Toplevel(self)
        win.title(title)
        vars = {}
        for f in fields:
            ttk.Label(win, text=f).pack(padx=10, pady=(6, 0))
            v = tk.StringVar()
            ttk.Entry(win, textvariable=v).pack(padx=10, pady=(0, 6))
            vars[f] = v
        out = {"vals": None}

        def ok():
            out["vals"] = {k: v.get() for k, v in vars.items()}
            win.destroy()

        ttk.Button(win, text="OK", command=ok).pack(pady=8)
        win.grab_set()
        win.wait_window()
        return out["vals"]

    # ------------------------------------------------------------
    # ADDING OBJECTS (dialogs + helpers)
    # ------------------------------------------------------------
    def _dialog_add_point(self):
        win = tk.Toplevel(self)
        win.title("Add Point")
        entries = {}
        for label in ("x", "y", "px_radius (pixels)"):
            row = ttk.Frame(win); row.pack(fill="x", padx=10, pady=(6, 0))
            ttk.Label(row, text=label, width=18).pack(side="left")
            var = tk.StringVar()
            ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
            entries[label] = var
        entries["px_radius (pixels)"].set("6")

        facecolor = {"hex": "#1f77b4"}
        self._make_color_row(win, "facecolor", facecolor["hex"], lambda h: facecolor.__setitem__("hex", h))
        edgecolor = {"hex": "#333333"}
        self._make_color_row(win, "edgecolor", edgecolor["hex"], lambda h: edgecolor.__setitem__("hex", h))

        out = {"ok": False}

        def ok():
            out["ok"] = True
            win.destroy()

        ttk.Button(win, text="OK", command=ok).pack(pady=10)
        win.grab_set(); win.wait_window()
        if not out["ok"]:
            return

        try:
            x = float(entries["x"].get()); y = float(entries["y"].get())
            px_radius = float(entries["px_radius (pixels)"].get() or 6)
        except Exception:
            return

        self._add_point(x, y, px_radius=px_radius, facecolor=facecolor["hex"], edgecolor=edgecolor["hex"])

    def _add_point(self, x, y, px_radius=6, facecolor="#1f77b4", edgecolor="#333"):
        p = Circle((x, y), radius=0.05, facecolor=facecolor, edgecolor=edgecolor)
        p._px_radius = px_radius
        self.ax.add_patch(p)
        self._points.append(p)
        oid = self._new_id("point")
        self._register_object(oid, "point", p,
                              {"x": x, "y": y, "px_radius": px_radius, "facecolor": facecolor, "edgecolor": edgecolor})
        self._update_point_radii()
        self.canvas.draw_idle()
        return p

    def _dialog_add_line(self):
        win = tk.Toplevel(self); win.title("Add Line")
        entries = {}
        for label in ("x1", "y1", "Length", "Angle (deg)", "lw"):
            row = ttk.Frame(win); row.pack(fill="x", padx=10, pady=(6, 0))
            ttk.Label(row, text=label, width=18).pack(side="left")
            var = tk.StringVar(); ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
            entries[label] = var
        entries["lw"].set("2")
        color = {"hex": "#000000"}
        self._make_color_row(win, "color", color["hex"], lambda h: color.__setitem__("hex", h))

        out = {"ok": False}
        def ok():
            out["ok"] = True; win.destroy()
        ttk.Button(win, text="OK", command=ok).pack(pady=10)
        win.grab_set(); win.wait_window()
        if not out["ok"]:
            return

        try:
            x1 = float(entries["x1"].get()); y1 = float(entries["y1"].get())
            L = float(entries["Length"].get())
            ang = np.radians(float(entries["Angle (deg)"].get()))
            lw = float(entries["lw"].get() or 2.0)
        except Exception:
            return

        x2 = x1 + L * np.cos(ang); y2 = y1 + L * np.sin(ang)
        line, = self.ax.plot([x1, x2], [y1, y2], lw=lw, color=color["hex"])
        self._segments.append(line)
        oid = self._new_id("segment")
        self._register_object(oid, "segment", line,
                              {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "color": color["hex"], "lw": lw})
        self.canvas.draw_idle()

    def _dialog_circle(self):
        win = tk.Toplevel(self); win.title("Add Circle")
        entries = {}
        for label in ("x", "y", "radius", "lw"):
            row = ttk.Frame(win); row.pack(fill="x", padx=10, pady=(6, 0))
            ttk.Label(row, text=label, width=18).pack(side="left")
            var = tk.StringVar(); ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
            entries[label] = var
        entries["lw"].set("2")
        edgecolor = {"hex": "#008000"}
        self._make_color_row(win, "edgecolor", edgecolor["hex"], lambda h: edgecolor.__setitem__("hex", h))

        out = {"ok": False}
        def ok(): out["ok"] = True; win.destroy()
        ttk.Button(win, text="OK", command=ok).pack(pady=10)
        win.grab_set(); win.wait_window()
        if not out["ok"]:
            return

        try:
            x = float(entries["x"].get()); y = float(entries["y"].get())
            r = float(entries["radius"].get()); lw = float(entries["lw"].get() or 2.0)
        except Exception:
            return

        c = Circle((x, y), r, fill=False, lw=lw, edgecolor=edgecolor["hex"])
        self.ax.add_patch(c); self._polygons.append(c)
        oid = self._new_id("circle")
        self._register_object(oid, "circle", c,
                              {"x": x, "y": y, "r": r, "edgecolor": edgecolor["hex"], "lw": lw})
        self.canvas.draw_idle()

    def _dialog_rectangle(self):
        win = tk.Toplevel(self); win.title("Add Rectangle")
        entries = {}
        for label in ("x", "y", "width", "height", "lw"):
            row = ttk.Frame(win); row.pack(fill="x", padx=10, pady=(6, 0))
            ttk.Label(row, text=label, width=18).pack(side="left")
            var = tk.StringVar(); ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
            entries[label] = var
        entries["lw"].set("2")
        edgecolor = {"hex": "#800080"}
        self._make_color_row(win, "edgecolor", edgecolor["hex"], lambda h: edgecolor.__setitem__("hex", h))

        out = {"ok": False}
        def ok(): out["ok"] = True; win.destroy()
        ttk.Button(win, text="OK", command=ok).pack(pady=10)
        win.grab_set(); win.wait_window()
        if not out["ok"]:
            return

        try:
            x = float(entries["x"].get()); y = float(entries["y"].get())
            w = float(entries["width"].get()); h = float(entries["height"].get())
            lw = float(entries["lw"].get() or 2.0)
        except Exception:
            return

        r = Rectangle((x, y), w, h, fill=False, lw=lw, edgecolor=edgecolor["hex"])
        self.ax.add_patch(r); self._polygons.append(r)
        oid = self._new_id("rectangle")
        self._register_object(oid, "rectangle", r,
                              {"x": x, "y": y, "w": w, "h": h, "edgecolor": edgecolor["hex"], "lw": lw})
        self.canvas.draw_idle()

    # ------------------------------------------------------------
    # MODE MANAGEMENT (Mutual Exclusivity)
    # ------------------------------------------------------------
    def _deactivate_modes(self, except_=None):
        """Turn off all interactive modes except the one named in 'except_'."""
        modes = {
            "polygon": ("_polygon_mode", self._clear_poly_preview),
            "line": ("_line_mode", self._clear_line_preview),
            "point": ("_point_mode", lambda: None),
            "text": ("_text_mode", lambda: None),
            "selection": ("_selection_mode", lambda: None),
        }
        for name, (flag, cleanup) in modes.items():
            if name == except_:
                continue
            if getattr(self, flag, False):
                setattr(self, flag, False)
                if name == "line":
                    self._line_start = None
                if name == "polygon":
                    self._polygon_pts = []
                cleanup()

    # ------------------------------------------------------------
    # POLYGON MODE + PREVIEW
    # ------------------------------------------------------------
    def _toggle_polygon_mode(self):
        turning_on = not self._polygon_mode
        if turning_on:
            self._deactivate_modes(except_="polygon")
            self._polygon_pts = []
            self._ensure_poly_preview()
            self._update_poly_preview()
            self.status.set("Polygon mode: click to add points, right-click or Enter to finish.")
            self.canvas.draw_idle()
        else:
            self._clear_poly_preview()
            self.status.set("Polygon mode off.")
        self._polygon_mode = turning_on

    def _ensure_poly_preview(self):
        if self._poly_preview is None:
            (self._poly_preview,) = self.ax.plot(
                [], [], linestyle="--", color="tab:gray", lw=1.5, alpha=0.8,
                marker="o", markersize=3, markerfacecolor="tab:gray"
            )

    def _update_poly_preview(self, mouse_xy=None):
        if self._poly_preview is None:
            return
        pts = list(self._polygon_pts)
        if mouse_xy is not None:
            pts.append(mouse_xy)
        xs = [p[0] for p in pts] if pts else []
        ys = [p[1] for p in pts] if pts else []
        self._poly_preview.set_data(xs, ys)

    def _clear_poly_preview(self):
        if self._poly_preview is not None:
            try: self._poly_preview.remove()
            except Exception: pass
        self._poly_preview = None
        self.canvas.draw_idle()

    def _finish_polygon(self):
        if len(self._polygon_pts) >= 3:
            xs = [p[0] for p in self._polygon_pts]
            ys = [p[1] for p in self._polygon_pts]
            pts = list(zip(xs, ys))
            face_hex = self._ask_color_from_palette(title="Choose polygon facecolor") or "#ffa500"
            edge_hex = self._ask_color_from_palette(title="Choose polygon edgecolor") or "#000000"
            poly = Polygon(pts, closed=True, facecolor=face_hex, alpha=0.35, edgecolor=edge_hex)
            self.ax.add_patch(poly); self._polygons.append(poly)
            oid = self._new_id("polygon")
            self._register_object(oid, "polygon", poly,
                                  {"points": pts, "facecolor": face_hex, "edgecolor": edge_hex, "alpha": 0.35, "lw": 1.5})
            self._polygon_pts = []; self._polygon_mode = False
            self.canvas.draw_idle(); self.status.set("Polygon created.")
            self._clear_poly_preview()

    # ------------------------------------------------------------
    # LINE MODE + PREVIEW + SNAP
    # ------------------------------------------------------------
    def _toggle_line_mode(self):
        turning_on = not self._line_mode
        if turning_on:
            self._deactivate_modes(except_="line")
            self._line_start = None
            self._ensure_line_preview()
            self._update_line_preview(None, None)
            self.status.set("Line mode: left-click start (snaps), move, left-click end (snaps). Right-click cancels, Esc to exit.")
            self.canvas.draw_idle()
        else:
            self._clear_line_preview()
            self.status.set("Line mode off.")
        self._line_mode = turning_on

    def _ensure_line_preview(self):
        if self._line_preview is None:
            (self._line_preview,) = self.ax.plot(
                [], [], linestyle="--", color="tab:gray", lw=1.5, alpha=0.9
            )

    def _update_line_preview(self, start, current):
        if self._line_preview is None:
            return
        if start is None or current is None:
            self._line_preview.set_data([], [])
        else:
            self._line_preview.set_data([start[0], current[0]], [start[1], current[1]])

    def _clear_line_preview(self):
        if self._line_preview is not None:
            try: self._line_preview.remove()
            except Exception: pass
        self._line_preview = None
        self.canvas.draw_idle()

    def _finish_line(self, start, end):
        if start is None or end is None:
            return
        seg_color = self._ask_color_from_palette(title="Choose segment color") or "#000000"
        lw = 2.0
        line, = self.ax.plot([start[0], end[0]], [start[1], end[1]], lw=lw, color=seg_color)
        self._segments.append(line)
        oid = self._new_id("segment")
        self._register_object(oid, "segment", line,
                              {"x1": start[0], "y1": start[1], "x2": end[0], "y2": end[1], "color": seg_color, "lw": lw})
        self._update_segments()
        self.canvas.draw_idle()

    # ------------------------------------------------------------
    # PLOT POINT MODE
    # ------------------------------------------------------------
    def _toggle_point_mode(self):
        turning_on = not self._point_mode
        if turning_on:
            self._deactivate_modes(except_="point")
            self._line_start = None
            self._clear_line_preview()
            self.status.set("Point mode: left-click to place points. Esc to exit.")
        else:
            self.status.set("Point mode off.")
        self._point_mode = turning_on

    # ------------------------------------------------------------
    # ADD TEXT MODE
    # ------------------------------------------------------------
    def _toggle_text_mode(self):
        turning_on = not self._text_mode
        if turning_on:
            self._deactivate_modes(except_="text")
            self.status.set("Text mode: left-click to place text. Esc to exit.")
        else:
            self.status.set("Text mode off.")
        self._text_mode = turning_on

    # ------------------------------------------------------------
    # CONNECTING POINTS, SEGMENTS, DETECTION
    # ------------------------------------------------------------
    def _toggle_connect(self):
        self._connect_mode = not self._connect_mode
        if self._connect_mode:
            self.status.set("Connect mode ON: click points to create segments.")
        else:
            self.status.set("Connect mode OFF.")

    def _connect_points_direct(self, p1, p2, color="#000000", lw=2.0):
        line, = self.ax.plot(
            [p1.center[0], p2.center[0]],
            [p1.center[1], p2.center[1]],
            lw=lw, color=color
        )
        self._segments.append(line)
        oid = self._new_id("segment")
        self._register_object(
            oid, "segment", line,
            {
                "x1": p1.center[0], "y1": p1.center[1],
                "x2": p2.center[0], "y2": p2.center[1],
                "color": color, "lw": lw
            }
        )

    def _detect_polygon_closure(self):
        if len(self._segments) < 3:
            return None
        pts = []
        used = set()
        for s in self._segments:
            (x1, x2), (y1, y2) = s.get_xdata(), s.get_ydata()
            a = (x1, y1); b = (x2, y2)
            if a not in used:
                pts.append(a); used.add(a)
            pts.append(b)
            if len(pts) >= 3 and np.allclose(pts[0], pts[-1], atol=1e-6):
                return pts
        return None

    def _connect_points(self, p1, p2):
        seg_color = self._ask_color_from_palette(title="Choose segment color") or "#000000"
        self._connect_points_direct(p1, p2, color=seg_color)
        pts = self._detect_polygon_closure()
        if pts:
            face_hex = self._ask_color_from_palette(title="Choose polygon facecolor") or "#ffa500"
            edge_hex = self._ask_color_from_palette(title="Choose polygon edgecolor") or "#000000"
            poly = Polygon(pts, closed=True, facecolor=face_hex, alpha=0.35, edgecolor=edge_hex)
            self.ax.add_patch(poly)
            self._polygons.append(poly)
            oid = self._new_id("polygon")
            self._register_object(oid, "polygon", poly,
                                  {"points": pts, "facecolor": face_hex, "edgecolor": edge_hex, "alpha": 0.35, "lw": 1.5})
            self.canvas.draw_idle()

    # ------------------------------------------------------------
    # HIT TESTING & SELECTION
    # ------------------------------------------------------------
    def _hit_test(self, x, y, tol=0.15):
        # 1) Points
        best = None
        bestd = tol * tol
        for p in self._points:
            px, py = p.center
            d = (px - x) ** 2 + (py - y) ** 2
            if d < bestd:
                best = p; bestd = d

        # 2) Segments
        min_px = 6.0
        best_line = None
        best_line_dist = None
        for s in self._segments:
            (x1, x2), (y1, y2) = s.get_xdata(), s.get_ydata()
            dpx = self._dist_to_segment_px(x, y, x1, y1, x2, y2)
            if dpx is not None and (best_line_dist is None or dpx < best_line_dist):
                best_line_dist = dpx
                best_line = s
        if best_line is not None and best_line_dist <= min_px:
            return best_line

        # 3) Curves
        best_curve = None
        best_curve_dist = None
        for c in self._curves:
            xs = np.asarray(c.get_xdata()); ys = np.asarray(c.get_ydata())
            if xs.size < 2:
                continue
            local_best = None
            for i in range(xs.size - 1):
                dpx = self._dist_to_segment_px(x, y, xs[i], ys[i], xs[i+1], ys[i+1])
                if dpx is not None:
                    if local_best is None or dpx < local_best:
                        local_best = dpx
                if local_best is not None and local_best <= min_px:
                    break
            if local_best is not None and (best_curve_dist is None or local_best < best_curve_dist):
                best_curve_dist = local_best
                best_curve = c
        if best_curve is not None and best_curve_dist <= min_px:
            return best_curve

        # 4) Labels
        for t in self._labels:
            tx, ty = t.get_position()
            d = (tx - x) ** 2 + (ty - y) ** 2
            if d < bestd:
                best = t; bestd = d

        # 5) Polygons/Circles/Rectangles by contains_point
        disp_pt = self.ax.transData.transform((x, y))
        for shp in reversed(self._polygons):
            try:
                if hasattr(shp, "contains_point") and shp.contains_point(disp_pt, radius=5):
                    return shp
            except Exception:
                pass

        return best

    def _dist_to_segment_px(self, x, y, x1, y1, x2, y2):
        p = np.array(self.ax.transData.transform((x, y)))
        a = np.array(self.ax.transData.transform((x1, y1)))
        b = np.array(self.ax.transData.transform((x2, y2)))
        ab = b - a
        ap = p - a
        denom = np.dot(ab, ab)
        if denom == 0:
            return float(np.linalg.norm(ap))
        t = np.clip(np.dot(ap, ab) / denom, 0.0, 1.0)
        closest = a + t * ab
        return float(np.linalg.norm(p - closest))

    # ------------------------------------------------------------
    # SELECTION & STYLING
    # ------------------------------------------------------------
    def _clear_selection(self):
        for a, st in self._last_styles.items():
            try:
                if isinstance(a, Circle):
                    a.set_edgecolor(st["edge"]); a.set_linewidth(st["lw"])
                elif isinstance(a, (Rectangle, Polygon)):
                    a.set_edgecolor(st["edge"]); a.set_linewidth(st["lw"])
                elif isinstance(a, Text):
                    a.set_bbox(st["bbox"])
                elif a in self._segments or a in self._curves:
                    a.set_color(st["edge"]); a.set_linewidth(st["lw"])
            except Exception:
                pass
        self._selected.clear(); self._last_styles.clear()

        if not self.tree.selection():
            for w in self.prop_inner.winfo_children():
                w.destroy()
            self.prop_entries.clear()
            self.prop_color_controls.clear()

    def _apply_selection_style(self, a):
        if isinstance(a, Circle):
            self._last_styles[a] = {"edge": a.get_edgecolor(), "lw": a.get_linewidth()}
            a.set_edgecolor("red"); a.set_linewidth(2.5)
        elif isinstance(a, (Rectangle, Polygon)):
            self._last_styles[a] = {"edge": a.get_edgecolor(), "lw": a.get_linewidth()}
            a.set_edgecolor("red"); a.set_linewidth(2.5)
        elif isinstance(a, Text):
            self._last_styles[a] = {"bbox": a.get_bbox_patch()}
            a.set_bbox(dict(facecolor="yellow", edgecolor="red"))
        elif a in self._segments or a in self._curves:
            self._last_styles[a] = {"edge": a.get_color(), "lw": a.get_linewidth()}
            a.set_color("red"); a.set_linewidth(2.5)

    def _select_single(self, a):
        self._clear_selection()
        self._selected = [a]
        self._apply_selection_style(a)
        self.canvas.draw_idle()

    # ------------------------------------------------------------
    # DRAGGING / MOVING
    # ------------------------------------------------------------
    def _start_drag(self, artist, x, y):
        if isinstance(artist, Circle):
            cx, cy = artist.center
        elif isinstance(artist, Rectangle):
            cx, cy = artist.get_x(), artist.get_y()
        elif isinstance(artist, Polygon):
            cx, cy = artist.get_xy()[0]
        elif isinstance(artist, Text):
            cx, cy = artist.get_position()
        elif artist in self._segments or artist in self._curves:
            cx, cy = x, y
        else:
            cx, cy = x, y
        self._drag = {"artist": artist, "offset": (cx - x, cy - y)}

    def _drag_selected(self, dx, dy):
        for a in self._selected:
            if isinstance(a, Circle):
                cx, cy = a.center; a.center = (cx + dx, cy + dy)
                self._update_registry_from_circle(a)
            elif isinstance(a, Rectangle):
                a.set_x(a.get_x() + dx); a.set_y(a.get_y() + dy)
                self._update_registry_from_rectangle(a)
            elif isinstance(a, Polygon):
                xy = a.get_xy(); xy[:, 0] += dx; xy[:, 1] += dy
                a.set_xy(xy); self._update_registry_from_polygon(a)
            elif isinstance(a, Text):
                tx, ty = a.get_position(); a.set_position((tx + dx, ty + dy))
                self._update_registry_from_text(a)
            elif a in self._segments:
                xs, ys = a.get_xdata(), a.get_ydata()
                xs = np.array(xs) + dx; ys = np.array(ys) + dy
                a.set_data(xs, ys); self._update_registry_from_segment(a)
            elif a in self._curves:
                xs, ys = a.get_xdata(), a.get_ydata()
                xs = np.array(xs) + dx; ys = np.array(ys) + dy
                a.set_data(xs, ys); self._update_registry_from_curve(a)
        self._update_segments()

    # registry update helpers
    def _update_registry_from_circle(self, c):
        for oid, data in self.registry.items():
            if data["ref"] is c and data["type"] == "circle":
                x, y = c.center; data["props"]["x"] = x; data["props"]["y"] = y
            if data["ref"] is c and data["type"] == "point":
                x, y = c.center; data["props"]["x"] = x; data["props"]["y"] = y

    def _update_registry_from_rectangle(self, r):
        for oid, data in self.registry.items():
            if data["ref"] is r and data["type"] == "rectangle":
                data["props"]["x"] = r.get_x(); data["props"]["y"] = r.get_y()

    def _update_registry_from_polygon(self, p):
        for oid, data in self.registry.items():
            if data["ref"] is p and data["type"] == "polygon":
                data["props"]["points"] = list(map(tuple, p.get_xy()))

    def _update_registry_from_text(self, t):
        for oid, data in self.registry.items():
            if data["ref"] is t and data["type"] == "text":
                data["props"]["x"], data["props"]["y"] = t.get_position()

    def _update_registry_from_segment(self, s):
        for oid, data in self.registry.items():
            if data["ref"] is s and data["type"] == "segment":
                x = s.get_xdata(); y = s.get_ydata()
                data["props"]["x1"], data["props"]["x2"] = x
                data["props"]["y1"], data["props"]["y2"] = y

    def _update_registry_from_curve(self, c):
        for oid, data in self.registry.items():
            if data["ref"] is c and data["type"] == "curve":
                data["props"]["xdata"] = list(map(float, c.get_xdata()))
                data["props"]["ydata"] = list(map(float, c.get_ydata()))

    # ------------------------------------------------------------
    # EVENT HANDLERS (PRESS, MOTION, RELEASE)
    # ------------------------------------------------------------
    def _on_press(self, e):
        if not e.inaxes:
            return
        x, y = e.xdata, e.ydata

        # selection rectangle highest priority
        if self._selection_mode:
            if e.button == 1:
                self._begin_selection_rect(e)
            return

        # Text mode
        if self._text_mode:
            if e.button == 1:
                v = self._prompt("Add text", ["Text"])
                if v and v["Text"]:
                    self._add_label(
                        x, y, v["Text"],
                        bbox=dict(facecolor="white", edgecolor="black", alpha=0.8)
                    )
                self.status.set("Text added. Click to add more, Esc to exit.")
            return

        # Polygon mode
        if self._polygon_mode:
            if e.button == 3:
                self._finish_polygon(); return
            elif e.button == 1:
                self._polygon_pts.append((x, y))
                self._ensure_poly_preview()
                self._update_poly_preview(mouse_xy=(x, y))
                self.status.set(f"Polygon mode: {len(self._polygon_pts)} point(s). Right-click or Enter to finish.")
                self.canvas.draw_idle()
            return

        # Line mode
        if self._line_mode:
            if e.button == 3:
                if self._line_start is not None:
                    self._line_start = None
                    self._update_line_preview(None, None); self.canvas.draw_idle()
                    self.status.set("Line mode: start canceled. Left-click start again, or Esc to exit.")
                else:
                    self._toggle_line_mode()
                return
            if e.button == 1:
                sx, sy = self._snap_to_point(x, y)
                if self._line_start is None:
                    self._line_start = (sx, sy)
                    self._ensure_line_preview()
                    self._update_line_preview(self._line_start, (sx, sy))
                    self.status.set("Line mode: move mouse, left-click to set end (snaps).")
                    self.canvas.draw_idle()
                else:
                    ex, ey = self._snap_to_point(x, y)
                    end = (ex, ey)
                    self._finish_line(self._line_start, end)
                    self._line_start = None
                    self._update_line_preview(None, None)
                    self.status.set("Line created. Left-click to start another, right-click cancels, Esc to exit.")
                return

        # Point mode
        if self._point_mode:
            if e.button == 1:
                self._add_point(x, y, px_radius=6, facecolor="#1f77b4", edgecolor="#333333")
                self.status.set("Point placed. Click to add more, Esc to exit.")
            return

        # Right click: add label
        if e.button == 3:
            v = self._prompt("Add label", ["Label"])
            if v and v["Label"]:
                self._add_label(
                    x, y, v["Label"],
                    bbox=dict(facecolor="white", edgecolor="black", alpha=0.8)
                )
            return

        # Middle: pan
        if e.button == 2:
            self._panning = True
            self._pan_start = (e.x, e.y, self.ax.get_xlim(), self.ax.get_ylim())
            return

        # Connect mode
        hit = self._hit_test(x, y)
        if self._connect_mode:
            if isinstance(hit, Circle) and hit in self._points:
                if self._pending_point is None:
                    self._pending_point = hit
                else:
                    self._connect_points(self._pending_point, hit)
                    self._pending_point = None
            return

        # Normal click
        if hit:
            if hit not in self._selected:
                self._select_single(hit)
            self._start_drag(hit, x, y)

    def _on_motion(self, e):
        if not e.inaxes:
            return
        self.status.set(f"x={e.xdata:.2f}, y={e.ydata:.2f}")

        if self._polygon_mode:
            self._ensure_poly_preview()
            self._update_poly_preview(mouse_xy=(e.xdata, e.ydata))
            self.canvas.draw_idle()

        if self._line_mode and self._line_start is not None:
            self._ensure_line_preview()
            ex, ey = self._snap_to_point(e.xdata, e.ydata)
            self._update_line_preview(self._line_start, (ex, ey))
            self.canvas.draw_idle()

        if self._panning:
            x0, y0, (x1, x2), (y1, y2) = self._pan_start
            dx, dy = e.x - x0, e.y - y0
            w = x2 - x1; h = y2 - y1
            mx = -dx * w / self.canvas_widget.winfo_width()
            my = dy * h / self.canvas_widget.winfo_height()
            self.ax.set_xlim(x1 + mx, x2 + mx)
            self.ax.set_ylim(y1 + my, y2 + my)
            self._update_point_radii()
            self.canvas.draw_idle()
            return

        if self._selection_mode and self._sel_start:
            self._update_selection_rect(e)
            return

        if not self._drag:
            return

        artist = self._drag["artist"]
        ox, oy = self._drag["offset"]
        nx = e.xdata + ox; ny = e.ydata + oy

        if self._snapping:
            if abs(ox) > abs(oy):
                if isinstance(artist, Circle): ny = artist.center[1]
                elif isinstance(artist, Text): ny = artist.get_position()[1]
            else:
                if isinstance(artist, Circle): nx = artist.center[0]
                elif isinstance(artist, Text): nx = artist.get_position()[0]

        if isinstance(artist, Circle):
            dx = nx - artist.center[0]; dy = ny - artist.center[1]
        elif isinstance(artist, Rectangle):
            dx = nx - artist.get_x(); dy = ny - artist.get_y()
        elif isinstance(artist, Polygon):
            xy = artist.get_xy()[0]; dx = nx - xy[0]; dy = ny - xy[1]
        elif isinstance(artist, Text):
            tx, ty = artist.get_position(); dx = nx - tx; dy = ny - ty
        elif artist in self._segments or artist in self._curves:
            dx = nx - e.xdata; dy = ny - e.ydata
        else:
            dx = dy = 0

        self._drag_selected(dx, dy)
        self.canvas.draw_idle()

    def _on_release(self, e):
        if self._panning:
            self._panning = False
        if self._selection_mode and self._sel_rect:
            self._finalize_selection_rect()
        self._drag = None
        self._snapping = False

    def _on_scroll(self, e):
        if e.inaxes != self.ax:
            return
        x, y = e.xdata, e.ydata
        x1, x2 = self.ax.get_xlim(); y1, y2 = self.ax.get_ylim()
        scale = 1.1 if e.button == "down" else 1/1.1
        w = (x2 - x1) * scale; h = (y2 - y1) * scale
        rx = (x - x1) / (x2 - x1); ry = (y - y1) / (y2 - y1)
        self.ax.set_xlim(x - rx * w, x + (1 - rx) * w)
        self.ax.set_ylim(y - ry * h, y + (1 - ry) * h)
        self._update_point_radii()
        self.canvas.draw_idle()

    def _on_key(self, e):
        if e.key == "shift":
            self._snapping = True
        elif e.key in ("delete", "backspace"):
            self._delete_selected()
        elif e.key == "escape":
            self._deactivate_modes(except_=None)
            if self._sel_rect:
                try:
                    self._sel_rect.remove()
                except Exception:
                    pass
                self._sel_rect = None
                self._sel_start = None
            self.status.set("All modes canceled.")
            self.canvas.draw_idle()
        elif e.key == "enter":
            if self._polygon_mode:
                self._finish_polygon()

    # ------------------------------------------------------------
    # SELECTION RECTANGLE LOGIC
    # ------------------------------------------------------------
    def _begin_selection_rect(self, event):
        self._sel_start = (event.xdata, event.ydata)
        if self._sel_rect:
            try: self._sel_rect.remove()
            except Exception: pass
        self._sel_rect = Rectangle((event.xdata, event.ydata), 0, 0, fill=False, edgecolor="black", linewidth=1.5)
        self.ax.add_patch(self._sel_rect)
        self.canvas.draw_idle()

    def _update_selection_rect(self, event):
        if not self._sel_rect or not self._sel_start:
            return
        x0, y0 = self._sel_start
        x1, y1 = event.xdata, event.ydata
        if x1 is None or y1 is None:
            return
        self._sel_rect.set_x(min(x0, x1)); self._sel_rect.set_y(min(y0, y1))
        self._sel_rect.set_width(abs(x1 - x0)); self._sel_rect.set_height(abs(y1 - y0))
        self.canvas.draw_idle()

    def _finalize_selection_rect(self):
        if not self._sel_rect:
            return
        x = self._sel_rect.get_x(); y = self._sel_rect.get_y()
        w = self._sel_rect.get_width(); h = self._sel_rect.get_height()

        def inside(pt): return x <= pt[0] <= x + w and y <= pt[1] <= y + h

        self._clear_selection()
        for p in self._points:
            if inside(p.center): self._add_to_selection(p)
        for t in self._labels:
            if inside(t.get_position()): self._add_to_selection(t)
        for s in self._segments:
            xs, ys = s.get_xdata(), s.get_ydata()
            if inside((xs[0], ys[0])) or inside((xs[1], ys[1])): self._add_to_selection(s)
        for c in self._curves:
            xs = np.asarray(c.get_xdata()); ys = np.asarray(c.get_ydata())
            if xs.size and ys.size:
                if xs.min() >= x and xs.max() <= x + w and ys.min() >= y and ys.max() <= y + h:
                    self._add_to_selection(c)
        for shp in self._polygons:
            pts = shp.get_path().vertices
            pts = shp.get_transform().transform(pts)
            pts_data = self.ax.transData.inverted().transform(pts)
            xs = pts_data[:, 0]; ys = pts_data[:, 1]
            if xs.min() >= x and xs.max() <= x + w and ys.min() >= y and ys.max() <= y + h:
                self._add_to_selection(shp)

        try: self._sel_rect.remove()
        except Exception: pass
        self._sel_rect = None; self._sel_start = None
        self.canvas.draw_idle()

    def _add_to_selection(self, a):
        if a in self._selected:
            return
        self._selected.append(a)
        self._apply_selection_style(a)
        self.canvas.draw_idle()

    # ------------------------------------------------------------
    # TOGGLES, CLEARING, MEASUREMENTS
    # ------------------------------------------------------------
    def _toggle_selection_mode(self):
        turning_on = not self._selection_mode
        if turning_on:
            self._deactivate_modes(except_="selection")
            self._line_start = None
            self._clear_line_preview()
            self._clear_poly_preview()
            self._sel_rect = None
            self._sel_start = None
            self.status.set("Selection mode ON. Drag to select.")
        else:
            if self._sel_rect:
                try: self._sel_rect.remove()
                except Exception: pass
            self._sel_rect = None
            self._sel_start = None
            self.status.set("Selection mode OFF.")
        self._selection_mode = turning_on
        self.canvas.draw_idle()

    def _clear_segments(self):
        for s in self._segments:
            try: s.remove()
            except Exception: pass
        self._segments.clear()
        for oid in list(self.registry.keys()):
            if self.registry[oid]["type"] == "segment":
                del self.registry[oid]
        self._refresh_object_list()
        self.canvas.draw_idle()

    def _clear_polygons(self):
        for p in self._polygons:
            try: p.remove()
            except Exception: pass
        self._polygons.clear()
        for oid in list(self.registry.keys()):
            if self.registry[oid]["type"] in ("polygon", "circle", "rectangle"):
                del self.registry[oid]
        self._refresh_object_list()
        self.canvas.draw_idle()

    def _clear_curves(self):
        for c in self._curves:
            try: c.remove()
            except Exception: pass
        self._curves.clear()
        for oid in list(self.registry.keys()):
            if self.registry[oid]["type"] == "curve":
                del self.registry[oid]
        self._refresh_object_list()
        self.canvas.draw_idle()
        self.status.set("All fits cleared.")

    def _add_label(self, x, y, text, **kwargs):
        t = self.ax.text(x, y, text, **kwargs)
        self._labels.append(t)
        oid = self._new_id("text")
        self._register_object(
            oid, "text", t,
            {"x": x, "y": y, "text": text}
        )
        return t

    def _measure_segment_length(self):
        if len(self._selected) != 1:
            self.status.set("Select one segment."); return
        s = self._selected[0]
        if s not in self._segments:
            self.status.set("Not a segment."); return
        (x1, x2), (y1, y2) = s.get_xdata(), s.get_ydata()
        d = np.hypot(x2 - x1, y2 - y1)
        self.status.set(f"Length: {d:.4f}")
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        self._add_label(
            mx, my, f"Length: {d:.4f}",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.8)
        )
        self.canvas.draw_idle()

    def _polygon_area(self):
        for p in reversed(self._polygons):
            if isinstance(p, Polygon):
                pts = p.get_xy()
                if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
                    pts = pts[:-1]
                xs = pts[:, 0]; ys = pts[:, 1]
                area = 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
                self.status.set(f"Polygon area: {area:.4f}")
                cross = xs * np.roll(ys, -1) - np.roll(xs, -1) * ys
                A = np.sum(cross) / 2.0
                if abs(A) > 1e-12:
                    cx = np.sum((xs + np.roll(xs, -1)) * cross) / (6.0 * A)
                    cy = np.sum((ys + np.roll(ys, -1)) * cross) / (6.0 * A)
                else:
                    cx = (xs.min() + xs.max()) / 2.0
                    cy = (ys.min() + ys.max()) / 2.0
                self._add_label(
                    cx, cy, f"Area: {area:.4f}",
                    bbox=dict(facecolor="white", edgecolor="black", alpha=0.8)
                )
                self.canvas.draw_idle()
                return
        self.status.set("No polygon found.")

    # ------------------------------------------------------------
    # MISC. COMPUTATIONS & SNAP
    # ------------------------------------------------------------
    def _update_segments(self):
        for s in self._segments:
            xs = s.get_xdata(); ys = s.get_ydata()
            p1 = self._nearest_point(xs[0], ys[0])
            p2 = self._nearest_point(xs[1], ys[1])
            if p1 and p2:
                s.set_data([p1.center[0], p2.center[0]],
                           [p1.center[1], p2.center[1]])

    def _nearest_point(self, x, y, tol=0.15):
        for p in self._points:
            px, py = p.center
            if (px-x)**2 + (py-y)**2 < tol**2:
                return p
        return None

    def _snap_to_point(self, x, y, tol=0.15):
        p = self._nearest_point(x, y, tol=tol)
        if p is not None:
            return p.center
        return (x, y)

    # ------------------------------------------------------------
    # DELETE SELECTED
    # ------------------------------------------------------------
    def _delete_selected(self):
        for a in self._selected:
            for oid, data in list(self.registry.items()):
                if data["ref"] is a:
                    del self.registry[oid]
            try:
                if isinstance(a, Circle):
                    if a in self._points: self._points.remove(a)
                    if a in self._polygons: self._polygons.remove(a)
                    a.remove()
                elif isinstance(a, Text):
                    if a in self._labels: self._labels.remove(a)
                    a.remove()
                elif a in self._segments:
                    self._segments.remove(a); a.remove()
                elif a in self._polygons:
                    self._polygons.remove(a); a.remove()
                elif a in self._curves:
                    self._curves.remove(a); a.remove()
            except Exception:
                pass
        self._clear_selection()
        self._refresh_object_list()
        self.canvas.draw_idle()

    # ============================================================
    # ========== REGRESSION TOOLBOX ==============================
    # ============================================================
    def _get_xy_points(self):
        """Return (x, y) as numpy arrays from selected circles if ≥2 else from all points. Sorted by x."""
        sel_pts = [a for a in self._selected if isinstance(a, Circle)]
        if len(sel_pts) >= 2:
            xs = np.array([p.center[0] for p in sel_pts], dtype=float)
            ys = np.array([p.center[1] for p in sel_pts], dtype=float)
            source = f"{len(sel_pts)} selected point(s)"
        else:
            if len(self._points) < 2:
                self.status.set("Need at least 2 points for regression.")
                return None, None, None
            xs = np.array([p.center[0] for p in self._points], dtype=float)
            ys = np.array([p.center[1] for p in self._points], dtype=float)
            source = f"all {len(self._points)} points"
        idx = np.argsort(xs)
        return xs[idx], ys[idx], source

    def _choose_color_and_plot(self, x_fit, y_fit, info, model="curve", lw=2.0):
        color = self._ask_color_from_palette(title=f"Choose color for {model}") or "#000000"
        line, = self.ax.plot(x_fit, y_fit, color=color, lw=lw)
        self._curves.append(line)
        oid = self._new_id("curve")
        props = {"model": model, "info": info, "color": color, "lw": lw,
                 "xdata": list(map(float, x_fit)), "ydata": list(map(float, y_fit))}
        self._register_object(oid, "curve", line, props)
        self.canvas.draw_idle()

    # --- Basic regressions
    def _regression_linear(self):
        x, y, src = self._get_xy_points()
        if x is None: return
        b, a = np.polyfit(x, y, 1)  # y ≈ a + b x
        yhat = a + b * x
        info = f"Linear: y = {a:.4g} + {b:.4g}·x"
        self._choose_color_and_plot(x, yhat, info, model="linear")
        self.status.set(f"{info} (using {src})")

    def _regression_polynomial(self, degree=None):
        x, y, src = self._get_xy_points()
        if x is None: return
        if degree is None:
            vals = self._prompt("Polynomial degree", ["degree"])
            if not vals or not vals["degree"]:
                return
            try:
                degree = int(vals["degree"])
                if degree < 1: raise ValueError
            except Exception:
                self.status.set("Degree must be an integer ≥ 1.")
                return
        coef = np.polyfit(x, y, degree)
        poly = np.poly1d(coef)
        x_fit = np.linspace(float(x.min()), float(x.max()), 400)
        y_fit = poly(x_fit)
        terms = " + ".join([f"{c:.3g}·x^{k}" for k, c in zip(range(degree, -1, -1), coef)])
        info = f"Poly deg {degree}: y = {terms}"
        self._choose_color_and_plot(x_fit, y_fit, info, model=f"poly{degree}")
        self.status.set(f"{info} (using {src})")

    def _regression_exponential(self):
        x, y, src = self._get_xy_points()
        if x is None: return
        if np.any(y <= 0):
            self.status.set("Exponential fit requires all y > 0.")
            return
        Y = np.log(y)  # ln y = ln a + b x
        b, ln_a = np.polyfit(x, Y, 1)
        a = np.exp(ln_a)
        x_fit = np.linspace(float(x.min()), float(x.max()), 400)
        y_fit = a * np.exp(b * x_fit)
        info = f"Exponential: y = {a:.4g}·e^({b:.4g}·x)"
        self._choose_color_and_plot(x_fit, y_fit, info, model="exponential")
        self.status.set(f"{info} (using {src})")

    def _regression_logarithmic(self):
        x, y, src = self._get_xy_points()
        if x is None: return
        if np.any(x <= 0):
            self.status.set("Logarithmic fit requires all x > 0.")
            return
        X = np.log(x)  # y = a + b ln x
        b, a = np.polyfit(X, y, 1)
        x_fit = np.linspace(float(x.min()), float(x.max()), 400)
        y_fit = a + b * np.log(x_fit)
        info = f"Logarithmic: y = {a:.4g} + {b:.4g}·ln x"
        self._choose_color_and_plot(x_fit, y_fit, info, model="logarithmic")
        self.status.set(f"{info} (using {src})")

    def _regression_power(self):
        x, y, src = self._get_xy_points()
        if x is None: return
        if np.any(x <= 0) or np.any(y <= 0):
            self.status.set("Power-law fit requires all x > 0 and y > 0.")
            return
        X = np.log(x); Y = np.log(y)  # ln y = ln a + b ln x
        b, ln_a = np.polyfit(X, Y, 1)
        a = np.exp(ln_a)
        x_fit = np.linspace(float(x.min()), float(x.max()), 400)
        y_fit = a * np.power(x_fit, b)
        info = f"Power-law: y = {a:.4g}·x^{b:.4g}"
        self._choose_color_and_plot(x_fit, y_fit, info, model="power")
        self.status.set(f"{info} (using {src})")

    # --- scikit-learn regressions (optional)
    def _regression_sklearn(self, kind):
        try:
            if kind in ("ridge", "lasso", "elasticnet"):
                from sklearn.linear_model import Ridge, Lasso, ElasticNet
            elif kind == "svr":
                from sklearn.svm import SVR
            elif kind == "tree":
                from sklearn.tree import DecisionTreeRegressor
            elif kind == "forest":
                from sklearn.ensemble import RandomForestRegressor
            elif kind == "knn":
                from sklearn.neighbors import KNeighborsRegressor
        except Exception:
            self.status.set("scikit-learn not available. Install with: pip install scikit-learn")
            return

        x, y, src = self._get_xy_points()
        if x is None: return
        X = x.reshape(-1, 1)

        if kind == "ridge":
            alpha = 1.0
            try:
                vals = self._prompt("Ridge α (leave blank for 1.0)", ["alpha"])
                if vals and vals["alpha"]:
                    alpha = float(vals["alpha"])
            except: pass
            model = Ridge(alpha=alpha, random_state=0)
            model_name = f"Ridge(α={alpha})"

        elif kind == "lasso":
            alpha = 0.1
            try:
                vals = self._prompt("Lasso α (leave blank for 0.1)", ["alpha"])
                if vals and vals["alpha"]:
                    alpha = float(vals["alpha"])
            except: pass
            model = Lasso(alpha=alpha, random_state=0, max_iter=10000)
            model_name = f"Lasso(α={alpha})"

        elif kind == "elasticnet":
            alpha, l1 = 0.1, 0.5
            try:
                vals = self._prompt("ElasticNet α, l1_ratio (default 0.1, 0.5)", ["alpha", "l1_ratio"])
                if vals and vals["alpha"]:
                    alpha = float(vals["alpha"])
                if vals and vals["l1_ratio"]:
                    l1 = float(vals["l1_ratio"])
            except: pass
            model = ElasticNet(alpha=alpha, l1_ratio=l1, random_state=0, max_iter=10000)
            model_name = f"ElasticNet(α={alpha}, l1={l1})"

        elif kind == "svr":
            model = SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1)
            model_name = "SVR(RBF)"

        elif kind == "tree":
            model = DecisionTreeRegressor(random_state=0)
            model_name = "DecisionTree"

        elif kind == "forest":
            model = RandomForestRegressor(n_estimators=200, random_state=0)
            model_name = "RandomForest"

        elif kind == "knn":
            k = 5
            try:
                vals = self._prompt("KNN k (default 5)", ["k"])
                if vals and vals["k"]:
                    k = int(vals["k"])
            except: pass
            model = KNeighborsRegressor(n_neighbors=k)
            model_name = f"KNN(k={k})"

        else:
            self.status.set("Unsupported model.")
            return

        try:
            model.fit(X, y)
        except Exception as ex:
            self.status.set(f"Fitting failed: {ex}")
            return

        x_fit = np.linspace(float(x.min()), float(x.max()), 400).reshape(-1, 1)
        try:
            y_fit = model.predict(x_fit)
        except Exception as ex:
            self.status.set(f"Prediction failed: {ex}")
            return
        x_fit = x_fit.ravel()
        info = f"{model_name}"
        self._choose_color_and_plot(x_fit, y_fit, info, model=model_name)
        self.status.set(f"{model_name} fit (using {src})")

    # --- Run regression(s) by typing names
    def _regression_by_names_prompt(self):
        vals = self._prompt(
            "Run Regression(s) by Name",
            ["models (comma-separated): e.g. linear, poly3, exponential, ridge"]
        )
        if not vals:
            return
        raw = vals.get("models (comma-separated): e.g. linear, poly3, exponential, ridge", "")
        self._run_regressions_from_string(raw)

    def _run_regressions_from_string(self, s: str):
        if not s:
            self.status.set("No model names provided.")
            return
        tokens = [t.strip() for t in s.split(",") if t.strip()]
        ran = 0
        for tok in tokens:
            self._run_single_model_token(tok)
            ran += 1
        if ran:
            self.status.set(f"Ran {ran} model(s): {', '.join(tokens)}")

    def _run_single_model_token(self, tok: str):
        t = tok.strip().lower()

        # Polynomial variants
        degree = None
        if t.startswith("poly"):
            suffix = t[4:].strip()
            if suffix.isdigit():
                degree = int(suffix)
                return self._regression_polynomial(degree=degree)
            else:
                return self._regression_polynomial()
        if t.startswith("polynomial"):
            for sep in (" ", ":", "="):
                if sep in t:
                    try:
                        deg = int(t.split(sep)[1])
                        degree = deg
                    except:
                        pass
                    break
            return self._regression_polynomial(degree=degree)

        # Basic names
        if t in ("linear",):
            return self._regression_linear()
        if t in ("exp", "exponential"):
            return self._regression_exponential()
        if t in ("log", "logarithmic"):
            return self._regression_logarithmic()
        if t in ("power", "powerlaw", "power-law"):
            return self._regression_power()

        # scikit-learn names/aliases
        alias = {
            "ridge": "ridge",
            "lasso": "lasso",
            "elasticnet": "elasticnet",
            "svr": "svr",
            "support vector": "svr",
            "decision tree": "tree",
            "tree": "tree",
            "random forest": "forest",
            "forest": "forest",
            "knn": "knn",
            "k-nearest": "knn",
        }
        if t in alias:
            return self._regression_sklearn(alias[t])

        self.status.set(f"Unknown model: '{tok}'. Supported: linear, polyN, exponential, logarithmic, power, ridge, lasso, elasticnet, svr, tree, forest, knn")

    # ============================================================
    # ===== Import Points (CSV/Excel) ============================
    # ============================================================
    def _import_points_dialog(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Import Points",
            filetypes=[
                ("All supported", "*.csv;*.xlsx;*.xls"),
                ("CSV", "*.csv"),
                ("Excel workbook", "*.xlsx"),
                ("Excel 97-2003", "*.xls"),
                ("All files", "*.*"),
            ]
        )
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".csv":
                n = self._load_points_from_csv(path)
            elif ext in (".xlsx", ".xls"):
                n = self._load_points_from_excel(path, ext)
            else:
                self.status.set("Unsupported file type. Use CSV, XLSX, or XLS.")
                return
            self.canvas.draw_idle()
            self.status.set(f"Imported {n} point(s) from {os.path.basename(path)}")
        except Exception as ex:
            self.status.set(f"Import failed: {ex}")

    def _load_points_from_csv(self, path):
        count = 0
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return 0

        def is_float(s):
            try:
                float(s); return True
            except:
                return False

        header = None
        start_idx = 0
        if not all(is_float(c) for c in rows[0][:2]):
            header = [c.strip().lower() for c in rows[0]]
            start_idx = 1
        x_idx = y_idx = None
        if header:
            for i, name in enumerate(header):
                if name == "x": x_idx = i
                elif name == "y": y_idx = i
        if x_idx is None or y_idx is None:
            x_idx, y_idx = 0, 1

        for r in rows[start_idx:]:
            if len(r) <= max(x_idx, y_idx):
                continue
            try:
                x = float(r[x_idx]); y = float(r[y_idx])
            except:
                continue
            self._add_point(x, y, px_radius=6, facecolor="#1f77b4", edgecolor="#333333")
            count += 1
        return count

    def _load_points_from_excel(self, path, ext):
        try:
            import pandas as pd
        except Exception:
            raise RuntimeError("pandas is required for Excel import. Install with: pip install pandas openpyxl (xlsx) or pip install pandas xlrd (xls)")
        engine = None
        if ext == ".xlsx":
            engine = "openpyxl"
        elif ext == ".xls":
            engine = "xlrd"
        try:
            df = pd.read_excel(path, engine=engine)
        except Exception as ex:
            raise RuntimeError(f"Failed to read Excel. Ensure the right engine is installed. Details: {ex}")
        if df.empty:
            return 0
        lower_cols = {c.lower(): c for c in df.columns}
        x_col = lower_cols.get("x")
        y_col = lower_cols.get("y")
        if x_col is None or y_col is None:
            if len(df.columns) < 2:
                return 0
            x_col, y_col = df.columns[:2]
        x = pd.to_numeric(df[x_col], errors="coerce")
        y = pd.to_numeric(df[y_col], errors="coerce")
        mask = (~x.isna()) & (~y.isna())
        xs = x[mask].to_numpy(dtype=float)
        ys = y[mask].to_numpy(dtype=float)
        count = 0
        for xv, yv in zip(xs, ys):
            self._add_point(float(xv), float(yv), px_radius=6, facecolor="#1f77b4", edgecolor="#333333")
            count += 1
        return count

    # ============================================================
    # ===== Plot arbitrary f(x) (safe-eval) ======================
    # ============================================================
    def _plot_function_dialog(self):
        """Open a dialog where the user types f(x), x-range, and options; then plot."""
        win = tk.Toplevel(self)
        win.title("Plot Function f(x)")
        win.transient(self)
        win.grab_set()

        frm = ttk.Frame(win); frm.pack(fill="both", expand=True, padx=12, pady=10)
        ttk.Label(frm, text="f(x) =").grid(row=0, column=0, sticky="w")
        expr_var = tk.StringVar(value="sin(x)")
        expr_ent = ttk.Entry(frm, textvariable=expr_var, width=38)
        expr_ent.grid(row=0, column=1, columnspan=3, sticky="ew", pady=(0, 8))

        ttk.Label(frm, text="x min").grid(row=1, column=0, sticky="w")
        xmin_var = tk.StringVar(value="-10")
        ttk.Entry(frm, textvariable=xmin_var, width=10).grid(row=1, column=1, sticky="w", padx=(0, 12))
        ttk.Label(frm, text="x max").grid(row=1, column=2, sticky="w")
        xmax_var = tk.StringVar(value="10")
        ttk.Entry(frm, textvariable=xmax_var, width=10).grid(row=1, column=3, sticky="w")

        ttk.Label(frm, text="samples").grid(row=2, column=0, sticky="w", pady=(8, 0))
        n_var = tk.StringVar(value="400")
        ttk.Entry(frm, textvariable=n_var, width=10).grid(row=2, column=1, sticky="w", padx=(0, 12), pady=(8, 0))

        deg_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Use degrees for trig (sin/cos/tan)", variable=deg_var).grid(
            row=2, column=2, columnspan=2, sticky="w", pady=(8, 0)
        )

        hint = (
            "Examples:\n"
            "  linear  (or linear(a,b))\n"
            "  exponential  (or exponential(a,b))\n"
            "  logarithmic  (or logarithmic(a,b))\n"
            "  power(a,b) / pow(a,b)  # plots a*x^b\n"
            "  pow()  # defaults to x\n"
            "  sin, cos, tan  (cos -> cos(x); sin(2) -> sin(2*x))\n"
            "  exp  or exp()  -> exp(x);  ln  or ln()  -> ln(x)\n"
            "  2x + 3, exp(0.5x), ln(x), sin(2x), cos(x/3)\n"
            "Notes: implicit multiplication like 2x, x2, 2(x+1), (x+1)(x-1) is supported."
        )
        ttk.Label(frm, text=hint, foreground="#555").grid(row=3, column=0, columnspan=4, sticky="w", pady=(8, 0))

        btns = ttk.Frame(win); btns.pack(fill="x", padx=12, pady=(8, 12))
        out = {"ok": False}

        def do_plot():
            try:
                expr = expr_var.get().strip()
                xmin = float(xmin_var.get())
                xmax = float(xmax_var.get())
                n = int(n_var.get())
                if n < 10: n = 10
                if n > 20000: n = 20000
                if xmin >= xmax:
                    self.status.set("x min must be < x max")
                    return
            except Exception as ex:
                self.status.set(f"Inputs invalid: {ex}")
                return

            # Expand friendly keywords (linear/exponential/logarithmic/power/pow/exp/ln/trig shorthands)
            expr_expanded = self._expand_function_keyword(expr)

            x = np.linspace(xmin, xmax, n)
            try:
                y = self._evaluate_function_expression(expr_expanded, x, use_degrees=deg_var.get())
                y = np.array(y, dtype=float)
                y[~np.isfinite(y)] = np.nan
            except Exception as ex:
                self.status.set(f"Function error: {ex}")
                return

            info = f"f(x) = {expr_expanded}"
            self._choose_color_and_plot(x, y, info, model="function")
            self.status.set(f"Plotted {info}")
            out["ok"] = True
            win.destroy()

        ttk.Button(btns, text="Plot", command=do_plot).pack(side="left")
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="right")

        for c in range(4):
            frm.grid_columnconfigure(c, weight=1)
        expr_ent.focus_set()
        win.wait_window()

    # --- Implicit multiplication insertion (FIXED to not break function calls)
    def _insert_implicit_multiplication(self, s: str) -> str:
        """
        Insert '*' for common implicit multiplication cases, without breaking function calls.
        - number variable : 2x    -> 2*x
        - variable number : x2    -> x*2
        - number '('     : 2(x)   -> 2*(x)
        - variable '('   : x(y)   -> x*(y)  (but DO NOT change func calls like exp(x))
        - ')' number / ')' letter : )2, )x  -> )*2, )*x
        - ')('           :        -> ')*('
        - number function: 5sin(x)-> 5*sin(x)
        """
        import re

        # 1) number followed by letter: 2x -> 2*x
        s = re.sub(r'(\d)\s*([A-Za-z])', r'\1*\2', s)

        # 2) letter followed by number: x2 -> x*2
        s = re.sub(r'([A-Za-z])\s*(\d)', r'\1*\2', s)

        # 3) number followed by '(' : 2(x+1) -> 2*(x+1)
        s = re.sub(r'(\d)\s*(\()', r'\1*\2', s)

        # 4) SAFE variable '(' : ONLY for single-letter variables (x or y) before '('
        #    x(y+1) -> x*(y+1) ; but exp(x) stays exp(x)
        s = re.sub(r'\b([xy])\s*(\()', r'\1*\2', s, flags=re.IGNORECASE)

        # 5) ')' followed by number: )2 -> )*2
        s = re.sub(r'(\))\s*(\d)', r'\1*\2', s)

        # 6) ')' followed by letter: )x -> )*x
        s = re.sub(r'(\))\s*([A-Za-z])', r'\1*\2', s)

        # 7) ')(' -> ')*('
        s = re.sub(r'(\))\s*(\()', r'\1*\2', s)

        # 8) number followed by known function names (sin,cos,tan,exp,ln,log, hyperbolic & inverse): 5sin(x) -> 5*sin(x)
        s = re.sub(r'(\d)\s*(sin|cos|tan|exp|ln|log|sinh|cosh|tanh|asin|acos|atan)\s*(\()',
                   r'\1*\2\3', s, flags=re.IGNORECASE)

        return s

    def _expand_function_keyword(self, s: str) -> str:
        """
        Expand friendly keywords into concrete f(x) expressions.

        Supported forms (case-insensitive):
          - "linear"                   -> "x"
          - "linear(a,b)"              -> "a + b*x"
          - "exponential"              -> "exp(x)"             # a=1, b=1
          - "exponential(a,b)"         -> "a*exp(b*x)"
          - "logarithmic"              -> "ln(x)"              # a=0, b=1
          - "logarithmic(a,b)"         -> "a + b*ln(x)"
          - "power(a,b)" / "pow(a,b)"  -> "a * x^b"            # a=1, b=1 if omitted
          - "pow()"                    -> "x"                  # default when no args
          - "sin" / "cos" / "tan"      -> "sin(x)" / ...
          - "sin(k)"                   -> "sin(k*x)"           # same for cos/tan
          - "exp" or "exp()"           -> "exp(x)"
          - "ln"  or "ln()"            -> "ln(x)"
        """
        import re

        t = s.strip().lower()
        if t.startswith("y="):
            t = t[2:].strip()

        # Simple shorthands for exp/ln
        if t == "exp" or re.fullmatch(r"exp\s*\(\s*\)", t):
            return "exp(x)"
        if t == "ln" or re.fullmatch(r"ln\s*\(\s*\)", t):
            return "ln(x)"

        # Trig shorthand
        simple_trig = ("sin", "cos", "tan")
        for name in simple_trig:
            if t == name:
                return f"{name}(x)"
            m = re.fullmatch(rf"{name}\s*\(\s*([^\(\)]*)\s*\)", t)
            if m:
                arg = m.group(1).strip()
                if arg and ('x' not in arg) and all(ch not in arg for ch in '()[]{}'):
                    return f"{name}(({arg})*x)"
                return f"{name}({arg})"

        # Linear
        if t == "linear":
            return "x"
        m = re.fullmatch(r"linear\s*\(\s*([^\)]*)\s*\)", t)
        if m:
            args = m.group(1).strip()
            a = b = None
            if args:
                parts = [p.strip() for p in args.split(",") if p.strip()]
                for p in parts:
                    if "=" in p:
                        k, v = [q.strip() for q in p.split("=", 1)]
                        if k == "a": a = v
                        if k == "b": b = v
                    else:
                        if a is None: a = p
                        elif b is None: b = p
            a = a or "0"
            b = b or "1"
            return f"({a}) + ({b})*x"

        # Exponential keyword (parameterized)
        if t == "exponential":
            return "exp(x)"
        m = re.fullmatch(r"exponential\s*\(\s*([^\)]*)\s*\)", t)
        if m:
            args = m.group(1).strip()
            a = b = None
            if args:
                parts = [p.strip() for p in args.split(",") if p.strip()]
                for p in parts:
                    if "=" in p:
                        k, v = [q.strip() for q in p.split("=", 1)]
                        if k == "a": a = v
                        if k == "b": b = v
                    else:
                        if a is None: a = p
                        elif b is None: b = p
            a = a or "1"
            b = b or "1"
            return f"({a})*exp(({b})*x)"

        # Logarithmic keyword
        if t == "logarithmic":
            return "ln(x)"
        m = re.fullmatch(r"logarithmic\s*\(\s*([^\)]*)\s*\)", t)
        if m:
            args = m.group(1).strip()
            a = b = None
            if args:
                parts = [p.strip() for p in args.split(",") if p.strip()]
                for p in parts:
                    if "=" in p:
                        k, v = [q.strip() for q in p.split("=", 1)]
                        if k == "a": a = v
                        if k == "b": b = v
                    else:
                        if a is None: a = p
                        elif b is None: b = p
            a = a or "0"
            b = b or "1"
            return f"({a}) + ({b})*ln(x)"

        # power / pow (with defaults)
        m_empty = re.fullmatch(r"(power|pow)\s*\(\s*\)", t)
        if m_empty:
            if m_empty.group(1) == "pow":
                return "x"  # pow() => x
            # power() left unchanged (no default)

        m = re.fullmatch(r"(power|pow)\s*\(\s*([^\)]*)\s*\)", t)
        if m:
            args = m.group(2).strip()
            a = b = None
            if args:
                parts = [p.strip() for p in args.split(",") if p.strip()]
                for p in parts:
                    if "=" in p:
                        k, v = [q.strip() for q in p.split("=", 1)]
                        if k == "a": a = v
                        if k == "b": b = v
                    else:
                        if a is None: a = p
                        elif b is None: b = p
            a = a or "1"
            b = b or "1"
            return f"({a}) * x^{b}"  # '^' -> '**' in evaluator

        # Nothing matched; return original
        return s

    def _evaluate_function_expression(self, expr: str, x: np.ndarray, use_degrees: bool = False):
        """
        Safely evaluate a user-provided f(x) with a restricted math namespace (NumPy).
        - Supports implicit multiplication (2x, x2, 2(x+1), (x+1)(x-1), 5sin(x), …).
        - Accepts '^' for exponent (auto converted to '**').
        - Accepts 'ln' as alias for natural log.
        - If use_degrees=True, trig functions receive degrees (converted to radians).
        """
        import numpy as _np

        ns = {
            "x": x,
            "pi": _np.pi,
            "e": _np.e,
            "abs": _np.abs,
            "sqrt": _np.sqrt,
            "pow": _np.power,       # NOTE: built-in pow for arrays
            "sign": _np.sign,
            "clip": _np.clip,
            "where": _np.where,
            "min": _np.minimum,
            "max": _np.maximum,
            "round": _np.round,
            "floor": _np.floor,
            "ceil": _np.ceil,
            "exp": _np.exp,
            "log": _np.log,
            "log10": _np.log10,
            "sin": _np.sin,
            "cos": _np.cos,
            "tan": _np.tan,
            "sinh": _np.sinh,
            "cosh": _np.cosh,
            "tanh": _np.tanh,
            "asin": _np.arcsin,
            "acos": _np.arccos,
            "atan": _np.arctan,
        }

        if use_degrees:
            ns.update({
                "sin": lambda v: _np.sin(_np.deg2rad(v)),
                "cos": lambda v: _np.cos(_np.deg2rad(v)),
                "tan": lambda v: _np.tan(_np.deg2rad(v)),
                "asin": lambda v: _np.rad2deg(_np.arcsin(v)),
                "acos": lambda v: _np.rad2deg(_np.arccos(v)),
                "atan": lambda v: _np.rad2deg(_np.arctan(v)),
            })

        s = expr.strip()
        if s.lower().startswith("y="):
            s = s[2:].strip()

        # Insert implicit multiplication before other rewrites (FIXED)
        s = self._insert_implicit_multiplication(s)

        # '^' -> '**' and 'ln(' -> 'log('
        s = s.replace("^", "**")
        s = s.replace("ln(", "log(")

        try:
            y = eval(s, {"__builtins__": {}}, ns)
        except NameError as ex:
            raise ValueError(f"Unknown name in expression: {ex}")
        except Exception:
            raise
        return y


# ------------------------------------------------------------
# MAIN APPLICATION
# ------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    app = InteractivePlane(root)
    root.mainloop()
