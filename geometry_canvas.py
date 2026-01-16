
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.text import Text
from collections import OrderedDict


class InteractivePlane(tk.Toplevel):
    # -------------------------------
    # INITIALIZATION
    # -------------------------------
    def __init__(self, master=None, title="Geometry Canvas"):
        super().__init__(master)
        self.title(title)
        self.geometry("1500x900")

        # --- FIGURE ---------------------------------------------------------
        self.fig, self.ax = plt.subplots(figsize=(9.5, 7))
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, linestyle="--", alpha=0.3)
        self.ax.axhline(0, color="#444")
        self.ax.axvline(0, color="#444")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side="left", fill="both", expand=True)

        # --- SIDEBAR (OBJECT INSPECTOR) ------------------------------------
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

        # --- SCROLLABLE PROPERTY GRID (Canvas + inner Frame + Scrollbar) ---
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
        self.prop_entries = {}        # for numeric/text props
        self.prop_color_controls = {} # for color props: {key: {"var": tk.StringVar, "swatch": tk.Label, "button": ttk.Button}}

        # --- STATUS BAR -----------------------------------------------------
        self.status = tk.StringVar(
            value="Left-click: point \n Right-click: text \n Wheel: zoom \n Space: pan \n Shift: snap"
        )
        ttk.Label(self, textvariable=self.status).pack(fill="x")

        # --- INTERNAL STORAGE ----------------------------------------------
        self._points = []
        self._labels = []
        self._functions = []
        self._segments = []
        self._polygons = []
        self._selected = []
        self._last_styles = {}
        self._drag = None
        self._snapping = False
        self._panning = False
        self._pan_start = None
        self._connect_mode = False
        self._pending_point = None
        self._polygon_mode = False
        self._polygon_pts = []
        self._selection_mode = False
        self._sel_rect = None
        self._sel_start = None
        self._angle_mode = False
        self._angle_points = []
        self._angle_cid = None

        # --- REGISTRY FOR INSPECTOR ----------------------------------------
        self.object_counter = {
            "point": 0,
            "segment": 0,
            "circle": 0,
            "rectangle": 0,
            "polygon": 0,
            "text": 0,
            "function": 0,
        }
        self.registry = {}  # id → {type, ref, props}

        # --- COLOR PALETTE --------------------------------------------------
        # A compact, visually distinct palette (Matplotlib "tab" colors + a few extras)
        self.COLOR_PALETTE = OrderedDict([
            ("tab:blue",   "#1f77b4"),
            ("tab:orange", "#ff7f0e"),
            ("tab:green",  "#2ca02c"),
            ("tab:red",    "#d62728"),
            ("tab:purple", "#9467bd"),
            ("tab:brown",  "#8c564b"),
            ("tab:pink",   "#e377c2"),
            ("tab:gray",   "#7f7f7f"),
            ("tab:olive",  "#bcbd22"),
            ("tab:cyan",   "#17becf"),
            ("black",      "#000000"),
            ("white",      "#ffffff"),
            ("gold",       "#ffd700"),
            ("dodgerblue", "#1e90ff"),
            ("limegreen",  "#32cd32"),
            ("crimson",    "#dc143c"),
            ("indigo",     "#4b0082"),
            ("teal",       "#008080"),
        ])

        # --- EVENT CONNECTIONS ---------------------------------------------
        cid = self.canvas.mpl_connect
        cid("button_press_event", self._on_press)
        cid("button_release_event", self._on_release)
        cid("motion_notify_event", self._on_motion)
        cid("scroll_event", self._on_scroll)
        cid("key_press_event", self._on_key)

        # Keep point sizes constant: update on each draw
        self.fig.canvas.mpl_connect("draw_event", lambda e: self._update_point_radii())

        self._build_menu()

    # ----------------------------------------------------------------------
    # SCROLL UTILS
    # ----------------------------------------------------------------------
    def _bind_prop_mousewheel(self, bind):
        # Windows / macOS wheel
        if bind:
            self.prop_canvas.bind_all("<MouseWheel>", self._on_prop_mousewheel)
            # Linux wheel
            self.prop_canvas.bind_all("<Button-4>", self._on_prop_mousewheel_linux)
            self.prop_canvas.bind_all("<Button-5>", self._on_prop_mousewheel_linux)
        else:
            self.prop_canvas.unbind_all("<MouseWheel>")
            self.prop_canvas.unbind_all("<Button-4>")
            self.prop_canvas.unbind_all("<Button-5>")

    def _on_prop_mousewheel(self, event):
        # On Windows/Mac, event.delta is ±120 per notch
        if event.delta:
            self.prop_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_prop_mousewheel_linux(self, event):
        if event.num == 4:
            self.prop_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.prop_canvas.yview_scroll(1, "units")

    # ----------------------------------------------------------------------
    # SIDEBAR REGISTRY METHODS
    # ----------------------------------------------------------------------
    def _new_id(self, kind):
        self.object_counter[kind] += 1
        return f"{kind[:2].upper()}{self.object_counter[kind]}"

    def _register_object(self, oid, kind, ref, props):
        self.registry[oid] = {
            "type": kind,
            "ref": ref,
            "props": props
        }
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
        if k == "function":
            expr = p.get("expr", "")
            xmi = p.get("x_min", None)
            xma = p.get("x_max", None)
            if xmi is not None and xma is not None:
                return f"{expr} [{xmi:.2f}, {xma:.2f}]"
            return expr
        return ""

    def _refresh_object_list(self):
        selected = self.tree.selection()
        keep_sel = selected[0] if selected else None
        for row in self.tree.get_children():
            self.tree.delete(row)
        for oid, data in self.registry.items():
            self.tree.insert(
                "", "end",
                iid=oid,
                values=(data["type"], self._make_summary(oid))
            )
        if keep_sel and keep_sel in self.registry:
            self.tree.selection_set(keep_sel)

    # ----------------------------------------------------------------------
    # COLOR PALETTE UI
    # ----------------------------------------------------------------------
    def _ask_color_from_palette(self, title="Choose color", initial=None):
        """
        Show a simple palette popup. Returns hex (like #aabbcc) or None.
        """
        top = tk.Toplevel(self)
        top.title(title)
        top.transient(self)
        top.grab_set()

        chosen = {"hex": None}

        # resolve name -> hex (accepts either)
        def to_hex(val):
            if not val:
                return None
            if val.startswith("#") and len(val) in (4, 7):
                return val
            # palette lookup
            return self.COLOR_PALETTE.get(val, None)

        def choose(val):
            chosen["hex"] = to_hex(val) if not val.startswith("#") else val
            top.destroy()

        pad = 6
        row = 0
        col = 0
        max_cols = 6

        # Optional "No color" row (if needed, but we keep it simple)
        # ttk.Button(top, text="None", command=lambda: choose(None)).grid(row=row, column=0, columnspan=max_cols, sticky="ew", padx=8, pady=(8, 2))
        # row += 1

        # Build grid of swatches
        for name, hexv in self.COLOR_PALETTE.items():
            cell = tk.Frame(top, width=30, height=24, bg=hexv, relief="ridge", bd=1, cursor="hand2")
            cell.grid(row=row, column=col, padx=pad, pady=pad)
            # tooltip label (name under)
            lbl = tk.Label(top, text=name, font=("Segoe UI", 8))
            lbl.grid(row=row+1, column=col, padx=pad, pady=(0, pad))
            cell.bind("<Button-1>", lambda e, val=hexv: choose(val))
            lbl.bind("<Button-1>",  lambda e, val=hexv: choose(val))

            col += 1
            if col >= max_cols:
                col = 0
                row += 2  # because we used two rows per swatch

        # Preselect/preview if initial provided (not strictly necessary)

        # Center popup relative to parent
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
        """
        Build one "color" control row in a dialog:
        - label
        - swatch (a small colored square)
        - "Choose…" button (opens palette)
        Returns a dictionary with vars for later reads if needed.
        """
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

    # ----------------------------------------------------------------------
    # PROPERTY GRID HANDLING
    # ----------------------------------------------------------------------
    def _on_tree_select(self, event):
        items = self.tree.selection()
        if not items:
            return
        oid = items[0]
        data = self.registry[oid]
        props = data["props"]

        # clear current properties (from the scrollable inner frame)
        for w in self.prop_inner.winfo_children():
            w.destroy()
        self.prop_entries.clear()
        self.prop_color_controls.clear()

        # Heading
        ttk.Label(
            self.prop_inner, text=f"{data['type'].upper()} Properties",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 6))

        # Which keys are colors for each kind?
        color_keys_by_kind = {
            "point":     ("facecolor", "edgecolor"),
            "segment":   ("color",),
            "circle":    ("edgecolor",),
            "rectangle": ("edgecolor",),
            "polygon":   ("facecolor", "edgecolor"),
            "function":  ("color",),
        }
        color_keys = color_keys_by_kind.get(data["type"], ())

        # Build entries (color keys get palette; others get Entry)
        for key, val in props.items():
            if key in color_keys:
                # Use swatch + choose button; update props & artist immediately
                def make_apply(k):
                    def apply_color(hexv):
                        props[k] = hexv
                        self._apply_properties(data["type"], data["ref"], props)
                        self._refresh_object_list()
                        self.canvas.draw_idle()
                    return apply_color

                ctrl = self._make_color_row(self.prop_inner, key, str(val), make_apply(key))
                self.prop_color_controls[key] = ctrl
            else:
                # Regular numeric/text prop
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
        # highlight selected object visually
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

        if key == "points":
            # polygons
            try:
                parsed = eval(raw, {"__builtins__": {}}, {})
                if isinstance(parsed, (list, tuple)):
                    props[key] = list(map(tuple, parsed))
                else:
                    props[key] = parsed
            except Exception:
                props[key] = raw
        else:
            # try float else keep as string
            try:
                val = float(raw)
                props[key] = val
            except:
                props[key] = raw

        self._apply_properties(obj["type"], ref, props)
        self._refresh_object_list()
        self.canvas.draw_idle()

    def _apply_properties(self, kind, ref, p):
        """Applies updated properties to the actual canvas objects."""
        if kind == "point":
            # position
            ref.center = (p["x"], p["y"])
            # visual
            if "facecolor" in p:
                try:
                    ref.set_facecolor(p["facecolor"])
                except Exception:
                    pass
            if "edgecolor" in p:
                try:
                    ref.set_edgecolor(p["edgecolor"])
                except Exception:
                    pass
            if "px_radius" in p:
                try:
                    ref._px_radius = float(p["px_radius"])
                except Exception:
                    pass
                self._update_point_radii()
            self._update_segments()

        elif kind == "segment":
            ref.set_data([p["x1"], p["x2"]], [p["y1"], p["y2"]])
            if "color" in p:
                try:
                    ref.set_color(p["color"])
                except Exception:
                    pass
            if "lw" in p:
                try:
                    ref.set_linewidth(float(p["lw"]))
                except Exception:
                    pass

        elif kind == "circle":
            ref.center = (p["x"], p["y"])
            ref.set_radius(p["r"])
            if "edgecolor" in p:
                try:
                    ref.set_edgecolor(p["edgecolor"])
                except Exception:
                    pass
            if "lw" in p:
                try:
                    ref.set_linewidth(float(p["lw"]))
                except Exception:
                    pass

        elif kind == "rectangle":
            ref.set_x(p["x"])
            ref.set_y(p["y"])
            ref.set_width(p["w"])
            ref.set_height(p["h"])
            if "edgecolor" in p:
                try:
                    ref.set_edgecolor(p["edgecolor"])
                except Exception:
                    pass
            if "lw" in p:
                try:
                    ref.set_linewidth(float(p["lw"]))
                except Exception:
                    pass

        elif kind == "polygon":
            pts = p["points"]
            ref.set_xy(pts)
            if "facecolor" in p:
                try:
                    ref.set_facecolor(p["facecolor"])
                except Exception:
                    pass
            if "edgecolor" in p:
                try:
                    ref.set_edgecolor(p["edgecolor"])
                except Exception:
                    pass
            if "alpha" in p:
                try:
                    ref.set_alpha(float(p["alpha"]))
                except Exception:
                    pass
            if "lw" in p:
                try:
                    ref.set_linewidth(float(p["lw"]))
                except Exception:
                    pass

        elif kind == "text":
            ref.set_position((p["x"], p["y"]))
            ref.set_text(p["text"])

        elif kind == "function":
            # Use user-selected span if present
            x1 = p.get("x_min", self.ax.get_xlim()[0])
            x2 = p.get("x_max", self.ax.get_xlim()[1])
            # Ensure correct order
            if isinstance(x1, (int, float)) and isinstance(x2, (int, float)) and x1 > x2:
                x1, x2 = x2, x1
                p["x_min"], p["x_max"] = x1, x2
                # also update the entries text to reflect swap
                if "x_min" in self.prop_entries:
                    self.prop_entries["x_min"].delete(0, "end")
                    self.prop_entries["x_min"].insert(0, str(x1))
                if "x_max" in self.prop_entries:
                    self.prop_entries["x_max"].delete(0, "end")
                    self.prop_entries["x_max"].insert(0, str(x2))
            xs = np.linspace(x1, x2, 800)
            try:
                ys = eval(p["expr"], {"x": xs, "np": np})
            except Exception:
                return
            ref.set_data(xs, ys)
            if "color" in p:
                try:
                    ref.set_color(p["color"])
                except Exception:
                    pass
            if "lw" in p:
                try:
                    ref.set_linewidth(float(p["lw"]))
                except Exception:
                    pass

    # ----------------------------------------------------------------------
    # PIXEL-STABLE POINT SIZE HELPERS
    # ----------------------------------------------------------------------
    def _pixels_to_data_radius(self, px):
        """Convert a radius in pixels to a radius in data units near the current view."""
        bb = self.ax.get_window_extent()
        w_px, h_px = bb.width, bb.height
        (x1, x2) = self.ax.get_xlim()
        (y1, y2) = self.ax.get_ylim()
        # data per pixel in each axis
        dx_per_px = (x2 - x1) / max(w_px, 1)
        dy_per_px = (y2 - y1) / max(h_px, 1)
        # average to keep circle looking round
        return 0.5 * (dx_per_px + dy_per_px) * px

    def _update_point_radii(self):
        """Update all point circle radii so they remain constant in screen pixels."""
        for p in self._points:
            px = getattr(p, "_px_radius", 6)
            try:
                p.set_radius(self._pixels_to_data_radius(px))
            except Exception:
                pass

    # ----------------------------------------------------------------------
    # MENU CREATION
    # ----------------------------------------------------------------------
    def _export(self, ext):
        from tkinter import filedialog
        file = filedialog.asksaveasfilename(defaultextension=f'.{ext}', filetypes=[(ext.upper(), f'*.{ext}')])
        if not file:
            return
        self.fig.savefig(file, dpi=300, bbox_inches='tight')
        self.status.set(f'Exported: {file}')

    def _recolor_selected(self):
        """Quick palette for the currently selected object."""
        if not self._selected:
            self.status.set("Select an object first.")
            return

        chosen = self._ask_color_from_palette(title="Choose color")
        if not chosen:
            return

        a = self._selected[0]
        # find registry entry
        found_oid = None
        for oid, data in self.registry.items():
            if data["ref"] is a:
                found_oid = oid
                break
        if not found_oid:
            return
        data = self.registry[found_oid]
        t = data["type"]

        # update registry props and artist
        if t in ("segment", "function"):
            data["props"]["color"] = chosen
            try:
                a.set_color(chosen)
            except Exception:
                pass
        elif t == "point":
            data["props"]["facecolor"] = chosen
            try:
                a.set_facecolor(chosen)
            except Exception:
                pass
        elif t in ("circle", "rectangle", "polygon"):
            data["props"]["edgecolor"] = chosen
            try:
                a.set_edgecolor(chosen)
            except Exception:
                pass
        # if polygon also has facecolor you want to change, you can add a second recolor flow

        self._refresh_object_list()
        self.canvas.draw_idle()

    def _build_menu(self):
        menu = tk.Menu(self)
        self.config(menu=menu)

        # FILE MENU
        mf_file = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label='File', menu=mf_file)
        mf_file.add_command(label='Export as PNG', command=lambda: self._export('png'))
        mf_file.add_command(label='Export as JPEG', command=lambda: self._export('jpg'))
        mf_file.add_command(label='Export as TIFF', command=lambda: self._export('tif'))

        # EDIT MENU
        me = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Edit", menu=me)
        me.add_command(label="Delete Selected", command=self._delete_selected)
        me.add_command(label="Recolor Selected…", command=self._recolor_selected)

        # FUNCTIONS MENU
        mf = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Functions", menu=mf)
        mf.add_command(label="Add Linear Function", command=self._dialog_linear)
        mf.add_command(label="Add Exponential Function", command=self._dialog_exponential)
        mf.add_command(label="Add Generic Function", command=self._dialog_generic)
        mf.add_separator()
        mf.add_command(label="Clear All Functions", command=self._clear_functions)

        # SHAPES MENU
        ms = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Shapes", menu=ms)
        ms.add_command(label="Add Point by Coordinates…", command=self._dialog_add_point)
        ms.add_command(label="Add Line (Length + Angle)…", command=self._dialog_add_line)
        ms.add_command(label="Add Circle…", command=self._dialog_circle)
        ms.add_command(label="Add Rectangle…", command=self._dialog_rectangle)
        ms.add_command(label="Draw Polygon (mode)", command=self._toggle_polygon_mode)
        ms.add_separator()
        ms.add_command(label="Selection Mode (toggle)", command=self._toggle_selection_mode)
        ms.add_separator()
        ms.add_command(label="Connect Points (toggle)", command=self._toggle_connect)
        ms.add_command(label="Clear Segments", command=self._clear_segments)
        ms.add_command(label="Clear Polygons", command=self._clear_polygons)
        ms.add_separator()
        ms.add_command(label="Measure Segment Length", command=self._measure_segment_length)
        ms.add_command(label="Segment Slope", command=self._segment_slope)
        ms.add_command(label="Compute Polygon Area", command=self._polygon_area)
        ms.add_command(label="Measure Angle", command=self._measure_angle)

    # ----------------------------------------------------------------------
    # PROMPT DIALOG + with Palette rows for color fields
    # ----------------------------------------------------------------------
    def _prompt(self, title, fields):
        """
        Generic prompt for numeric/text fields. (Used where there are no color fields.)
        """
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

    # ----------------------------------------------------------------------
    # ADDING OBJECTS (Dialogs now use palette for colors)
    # ----------------------------------------------------------------------
    def _dialog_add_point(self):
        # Custom dialog with palette for face/edge color and numeric px_radius
        win = tk.Toplevel(self)
        win.title("Add Point")

        # x, y, px_radius
        entries = {}
        for label in ("x", "y", "px_radius (pixels)"):
            row = ttk.Frame(win); row.pack(fill="x", padx=10, pady=(6, 0))
            ttk.Label(row, text=label, width=18).pack(side="left")
            var = tk.StringVar()
            ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
            entries[label] = var
        entries["px_radius (pixels)"].set("6")

        # facecolor
        facecolor = {"hex": "#1f77b4"}  # default
        self._make_color_row(win, "facecolor", facecolor["hex"], lambda h: facecolor.__setitem__("hex", h))
        # edgecolor
        edgecolor = {"hex": "#333333"}
        self._make_color_row(win, "edgecolor", edgecolor["hex"], lambda h: edgecolor.__setitem__("hex", h))

        out = {"ok": False}

        def ok():
            out["ok"] = True
            win.destroy()

        ttk.Button(win, text="OK", command=ok).pack(pady=10)
        win.grab_set()
        win.wait_window()
        if not out["ok"]:
            return
        try:
            x = float(entries["x"].get())
            y = float(entries["y"].get())
            px_radius = float(entries["px_radius (pixels)"].get() or 6)
        except Exception:
            return
        self._add_point(x, y, px_radius=px_radius, facecolor=facecolor["hex"], edgecolor=edgecolor["hex"])

    def _add_point(self, x, y, px_radius=6, facecolor="#1f77b4", edgecolor="#333"):
        # Radius will be stabilized by pixel → data conversion.
        p = Circle((x, y), radius=0.05, facecolor=facecolor, edgecolor=edgecolor)
        p._px_radius = px_radius  # store desired pixel radius
        self.ax.add_patch(p)
        self._points.append(p)
        oid = self._new_id("point")
        self._register_object(
            oid, "point", p,
            {"x": x, "y": y, "px_radius": px_radius, "facecolor": facecolor, "edgecolor": edgecolor}
        )
        self._update_point_radii()  # ensure correct radius immediately
        self.canvas.draw_idle()
        return p

    def _dialog_add_line(self):
        # Custom dialog with numeric fields + color palette + linewidth
        win = tk.Toplevel(self)
        win.title("Add Line")

        entries = {}
        for label in ("x1", "y1", "Length", "Angle (deg)", "lw"):
            row = ttk.Frame(win); row.pack(fill="x", padx=10, pady=(6, 0))
            ttk.Label(row, text=label, width=18).pack(side="left")
            var = tk.StringVar()
            ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
            entries[label] = var
        entries["lw"].set("2")

        # color via palette
        color = {"hex": "#000000"}
        self._make_color_row(win, "color", color["hex"], lambda h: color.__setitem__("hex", h))

        out = {"ok": False}
        def ok():
            out["ok"] = True
            win.destroy()

        ttk.Button(win, text="OK", command=ok).pack(pady=10)
        win.grab_set()
        win.wait_window()
        if not out["ok"]:
            return

        try:
            x1 = float(entries["x1"].get())
            y1 = float(entries["y1"].get())
            L = float(entries["Length"].get())
            ang = np.radians(float(entries["Angle (deg)"].get()))
            lw = float(entries["lw"].get() or 2.0)
        except Exception:
            return
        x2 = x1 + L * np.cos(ang)
        y2 = y1 + L * np.sin(ang)
        line, = self.ax.plot([x1, x2], [y1, y2], lw=lw, color=color["hex"])
        self._segments.append(line)
        oid = self._new_id("segment")
        self._register_object(
            oid, "segment", line,
            {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "color": color["hex"], "lw": lw}
        )
        self.canvas.draw_idle()

    def _dialog_circle(self):
        win = tk.Toplevel(self)
        win.title("Add Circle")
        entries = {}
        for label in ("x", "y", "radius", "lw"):
            row = ttk.Frame(win); row.pack(fill="x", padx=10, pady=(6, 0))
            ttk.Label(row, text=label, width=18).pack(side="left")
            var = tk.StringVar()
            ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
            entries[label] = var
        entries["lw"].set("2")

        edgecolor = {"hex": "#008000"}  # green-ish default
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
            x = float(entries["x"].get())
            y = float(entries["y"].get())
            r = float(entries["radius"].get())
            lw = float(entries["lw"].get() or 2.0)
        except Exception:
            return
        c = Circle((x, y), r, fill=False, lw=lw, edgecolor=edgecolor["hex"])
        self.ax.add_patch(c)
        self._polygons.append(c)
        oid = self._new_id("circle")
        self._register_object(
            oid, "circle", c,
            {"x": x, "y": y, "r": r, "edgecolor": edgecolor["hex"], "lw": lw}
        )
        self.canvas.draw_idle()

    def _dialog_rectangle(self):
        win = tk.Toplevel(self)
        win.title("Add Rectangle")
        entries = {}
        for label in ("x", "y", "width", "height", "lw"):
            row = ttk.Frame(win); row.pack(fill="x", padx=10, pady=(6, 0))
            ttk.Label(row, text=label, width=18).pack(side="left")
            var = tk.StringVar()
            ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
            entries[label] = var
        entries["lw"].set("2")

        edgecolor = {"hex": "#800080"}  # purple-ish default
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
            x = float(entries["x"].get())
            y = float(entries["y"].get())
            w = float(entries["width"].get())
            h = float(entries["height"].get())
            lw = float(entries["lw"].get() or 2.0)
        except Exception:
            return
        r = Rectangle((x, y), w, h, fill=False, lw=lw, edgecolor=edgecolor["hex"])
        self.ax.add_patch(r)
        self._polygons.append(r)
        oid = self._new_id("rectangle")
        self._register_object(
            oid, "rectangle", r,
            {"x": x, "y": y, "w": w, "h": h, "edgecolor": edgecolor["hex"], "lw": lw}
        )
        self.canvas.draw_idle()

    # ----------------------------------------------------------------------
    # FUNCTION DIALOGS
    # ----------------------------------------------------------------------
    def _dialog_linear(self):
        v = self._prompt("Linear Function", ["m", "b"])
        if not v:
            return
        try:
            m = float(v["m"])
            b = float(v["b"])
        except Exception:
            return
        expr = f"{m}*x+{b}"
        self._plot_function(expr)

    def _dialog_exponential(self):
        v = self._prompt("Exponential Function", ["a", "b"])
        if not v:
            return
        try:
            a = float(v["a"])
            b = float(v["b"])
        except Exception:
            return
        expr = f"{a}*np.exp({b}*x)"
        self._plot_function(expr)

    def _dialog_generic(self):
        v = self._prompt("Generic Function", ["y = f(x)"])
        if not v:
            return
        expr = v["y = f(x)"]
        self._plot_function(expr)

    def _plot_function(self, expr):
        x1, x2 = self.ax.get_xlim()
        xs = np.linspace(x1, x2, 600)
        try:
            ys = eval(expr, {"np": np, "x": xs})
        except Exception:
            return

        # Ask color via palette
        chosen = self._ask_color_from_palette(title="Choose function color") or None
        if chosen is None:
            # fall back to cycler if none chosen
            line, = self.ax.plot(xs, ys, lw=2)
            color = line.get_color()
        else:
            line, = self.ax.plot(xs, ys, lw=2, color=chosen)
            color = chosen

        self._functions.append(line)
        oid = self._new_id("function")
        self._register_object(
            oid, "function", line,
            {
                "expr": expr,
                "x_min": x1,
                "x_max": x2,
                "color": color,
                "lw": 2
            }
        )
        self.canvas.draw_idle()

    def _clear_functions(self):
        for l in self._functions:
            try:
                l.remove()
            except Exception:
                pass
        self._functions.clear()
        # remove functions from registry:
        for oid in list(self.registry.keys()):
            if self.registry[oid]["type"] == "function":
                del self.registry[oid]
        self._refresh_object_list()
        self.canvas.draw_idle()

    # ----------------------------------------------------------------------
    # LABELS (REGISTERED)
    # ----------------------------------------------------------------------
    def _add_label(self, x, y, text):
        t = self.ax.text(x, y, text)
        self._labels.append(t)
        oid = self._new_id("text")
        self._register_object(
            oid, "text", t,
            {"x": x, "y": y, "text": text}
        )
        return t

    # ----------------------------------------------------------------------
    # POLYGON MODE
    # ----------------------------------------------------------------------
    def _toggle_polygon_mode(self):
        self._polygon_mode = not self._polygon_mode
        self._polygon_pts = []
        if self._polygon_mode:
            self.status.set("Polygon mode: click to add points, right-click or Enter to finish.")
        else:
            self.status.set("Polygon mode off.")

    def _finish_polygon(self):
        if len(self._polygon_pts) >= 3:
            xs = [p[0] for p in self._polygon_pts]
            ys = [p[1] for p in self._polygon_pts]
            pts = list(zip(xs, ys))

            # Ask for face & edge colors
            face_hex = self._ask_color_from_palette(title="Choose polygon facecolor") or "#ffa500"  # orange
            edge_hex = self._ask_color_from_palette(title="Choose polygon edgecolor") or "#000000"  # black

            poly = Polygon(pts, closed=True, facecolor=face_hex, alpha=0.35, edgecolor=edge_hex)
            self.ax.add_patch(poly)
            self._polygons.append(poly)
            oid = self._new_id("polygon")
            self._register_object(oid, "polygon", poly,
                                  {"points": pts, "facecolor": face_hex, "edgecolor": edge_hex, "alpha": 0.35, "lw": 1.5})
            self._polygon_pts = []
            self._polygon_mode = False
            self.canvas.draw_idle()
            self.status.set("Polygon created.")

    # ----------------------------------------------------------------------
    # CONNECTING POINTS, SEGMENTS, DETECTION
    # ----------------------------------------------------------------------
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
                "x1": p1.center[0],
                "y1": p1.center[1],
                "x2": p2.center[0],
                "y2": p2.center[1],
                "color": color,
                "lw": lw
            }
        )

    def _detect_polygon_closure(self):
        if len(self._segments) < 3:
            return None
        pts = []
        used = set()
        for s in self._segments:
            xs = s.get_xdata()
            ys = s.get_ydata()
            a = (xs[0], ys[0])
            b = (xs[1], ys[1])
            if a not in used:
                pts.append(a)
                used.add(a)
            pts.append(b)
            if len(pts) >= 3 and np.allclose(pts[0], pts[-1], atol=1e-6):
                return pts
        return None

    def _connect_points(self, p1, p2):
        # Ask color via palette for the new segment
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

    # ----------------------------------------------------------------------
    # HIT TESTING & SELECTION
    # ----------------------------------------------------------------------
    def _hit_test(self, x, y, tol=0.15):
        best = None
        bestd = tol * tol
        # points
        for p in self._points:
            px, py = p.center
            d = (px - x)**2 + (py - y)**2
            if d < bestd:
                best = p; bestd = d
        # labels
        for t in self._labels:
            tx, ty = t.get_position()
            d = (tx - x)**2 + (ty - y)**2
            if d < bestd:
                best = t; bestd = d
        # segments endpoints (simple proximity)
        for s in self._segments:
            (x1, x2), (y1, y2) = s.get_xdata(), s.get_ydata()
            d = min((x1 - x)**2 + (y1 - y)**2,
                    (x2 - x)**2 + (y2 - y)**2)
            if d < bestd:
                best = s; bestd = d
        # polygons (inside hit)
        disp_pt = self.ax.transData.transform((x, y))
        for shp in reversed(self._polygons):
            if hasattr(shp, "contains_point") and shp.contains_point(disp_pt):
                best = shp
                break
        return best

    # ----------------------------------------------------------------------
    # SELECTION & STYLING
    # ----------------------------------------------------------------------
    def _clear_selection(self):
        for a, st in self._last_styles.items():
            try:
                if isinstance(a, Circle):
                    a.set_edgecolor(st["edge"])
                    a.set_linewidth(st["lw"])
                elif isinstance(a, (Rectangle, Polygon)):
                    a.set_edgecolor(st["edge"])
                    a.set_linewidth(st["lw"])
                elif isinstance(a, Text):
                    a.set_bbox(st["bbox"])
                elif a in self._segments:
                    a.set_color(st["edge"])
                    a.set_linewidth(st["lw"])
            except Exception:
                pass
        self._selected.clear()
        self._last_styles.clear()
        # Clear property grid if nothing selected in TreeView either
        if not self.tree.selection():
            for w in self.prop_inner.winfo_children():
                w.destroy()
            self.prop_entries.clear()
            self.prop_color_controls.clear()

    def _apply_selection_style(self, a):
        if isinstance(a, Circle):
            self._last_styles[a] = {
                "edge": a.get_edgecolor(),
                "lw": a.get_linewidth()
            }
            a.set_edgecolor("red")
            a.set_linewidth(2.5)
        elif isinstance(a, (Rectangle, Polygon)):
            self._last_styles[a] = {
                "edge": a.get_edgecolor(),
                "lw": a.get_linewidth()
            }
            a.set_edgecolor("red")
            a.set_linewidth(2.5)
        elif isinstance(a, Text):
            self._last_styles[a] = {"bbox": a.get_bbox_patch()}
            a.set_bbox(dict(facecolor="yellow", edgecolor="red"))
        elif a in self._segments:
            self._last_styles[a] = {
                "edge": a.get_color(),
                "lw": a.get_linewidth()
            }
            a.set_color("red")
            a.set_linewidth(2.5)

    def _select_single(self, a):
        self._clear_selection()
        self._selected = [a]
        self._apply_selection_style(a)
        self.canvas.draw_idle()

    # ----------------------------------------------------------------------
    # DRAGGING / MOVING
    # ----------------------------------------------------------------------
    def _start_drag(self, artist, x, y):
        if isinstance(artist, Circle):
            cx, cy = artist.center
        elif isinstance(artist, Rectangle):
            cx, cy = artist.get_x(), artist.get_y()
        elif isinstance(artist, Polygon):
            cx, cy = artist.get_xy()[0]
        elif isinstance(artist, Text):
            cx, cy = artist.get_position()
        elif artist in self._segments:
            cx, cy = x, y
        else:
            cx, cy = x, y
        self._drag = {
            "artist": artist,
            "offset": (cx - x, cy - y)
        }

    def _drag_selected(self, dx, dy):
        """Apply dragging to selected objects."""
        for a in self._selected:
            if isinstance(a, Circle):
                cx, cy = a.center
                a.center = (cx + dx, cy + dy)
                self._update_registry_from_circle(a)
            elif isinstance(a, Rectangle):
                a.set_x(a.get_x() + dx)
                a.set_y(a.get_y() + dy)
                self._update_registry_from_rectangle(a)
            elif isinstance(a, Polygon):
                xy = a.get_xy()
                xy[:, 0] += dx
                xy[:, 1] += dy
                a.set_xy(xy)
                self._update_registry_from_polygon(a)
            elif isinstance(a, Text):
                tx, ty = a.get_position()
                a.set_position((tx + dx, ty + dy))
                self._update_registry_from_text(a)
            elif a in self._segments:
                xs, ys = a.get_xdata(), a.get_ydata()
                xs = np.array(xs) + dx
                ys = np.array(ys) + dy
                a.set_data(xs, ys)
                self._update_registry_from_segment(a)
            self._update_segments()

    # registry update helpers
    def _update_registry_from_circle(self, c):
        for oid, data in self.registry.items():
            if data["ref"] is c and data["type"] == "circle":
                x, y = c.center
                data["props"]["x"] = x
                data["props"]["y"] = y
            if data["ref"] is c and data["type"] == "point":
                x, y = c.center
                data["props"]["x"] = x
                data["props"]["y"] = y

    def _update_registry_from_rectangle(self, r):
        for oid, data in self.registry.items():
            if data["ref"] is r and data["type"] == "rectangle":
                data["props"]["x"] = r.get_x()
                data["props"]["y"] = r.get_y()

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
                x = s.get_xdata()
                y = s.get_ydata()
                data["props"]["x1"], data["props"]["x2"] = x
                data["props"]["y1"], data["props"]["y2"] = y

    # ----------------------------------------------------------------------
    # EVENT HANDLERS (PRESS, MOTION, RELEASE)
    # ----------------------------------------------------------------------
    def _on_press(self, e):
        if not e.inaxes:
            return
        x, y = e.xdata, e.ydata

        # RIGHT click → add label
        if e.button == 3:
            v = self._prompt("Add label", ["Label"])
            if v and v["Label"]:
                self._add_label(x, y, v["Label"])
            return

        # MIDDLE → pan
        if e.button == 2:
            self._panning = True
            self._pan_start = (
                e.x, e.y,
                self.ax.get_xlim(),
                self.ax.get_ylim()
            )
            return

        # Selection rectangle mode
        if self._selection_mode:
            self._begin_selection_rect(e)
            return

        # Polygon mode
        if self._polygon_mode:
            self._polygon_pts.append((x, y))
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
        else:
            return

    def _on_motion(self, e):
        if not e.inaxes:
            return
        self.status.set(f"x={e.xdata:.2f}, y={e.ydata:.2f}")

        # panning
        if self._panning:
            x0, y0, (x1, x2), (y1, y2) = self._pan_start
            dx, dy = e.x - x0, e.y - y0
            w = x2 - x1
            h = y2 - y1
            mx = -dx * w / self.canvas_widget.winfo_width()
            my = dy * h / self.canvas_widget.winfo_height()
            self.ax.set_xlim(x1 + mx, x2 + mx)
            self.ax.set_ylim(y1 + my, y2 + my)
            self._update_point_radii()
            self.canvas.draw_idle()
            return

        # selection rectangle
        if self._selection_mode and self._sel_start:
            self._update_selection_rect(e)
            return

        # dragging
        if not self._drag:
            return
        artist = self._drag["artist"]
        ox, oy = self._drag["offset"]
        nx = e.xdata + ox
        ny = e.ydata + oy

        # snapping
        if self._snapping:
            if abs(ox) > abs(oy):
                # lock y
                if isinstance(artist, Circle):
                    ny = artist.center[1]
                elif isinstance(artist, Text):
                    ny = artist.get_position()[1]
            else:
                # lock x
                if isinstance(artist, Circle):
                    nx = artist.center[0]
                elif isinstance(artist, Text):
                    nx = artist.get_position()[0]

        # compute movement
        if isinstance(artist, Circle):
            dx = nx - artist.center[0]
            dy = ny - artist.center[1]
        elif isinstance(artist, Rectangle):
            dx = nx - artist.get_x()
            dy = ny - artist.get_y()
        elif isinstance(artist, Polygon):
            xy = artist.get_xy()[0]
            dx = nx - xy[0]
            dy = ny - xy[1]
        elif isinstance(artist, Text):
            tx, ty = artist.get_position()
            dx = nx - tx
            dy = ny - ty
        elif artist in self._segments:
            dx = nx - e.xdata
            dy = ny - e.ydata
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
        x1, x2 = self.ax.get_xlim()
        y1, y2 = self.ax.get_ylim()
        scale = 1.1 if e.button == "down" else 1/1.1
        w = (x2 - x1) * scale
        h = (y2 - y1) * scale
        rx = (x - x1) / (x2 - x1)
        ry = (y - y1) / (y2 - y1)
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
            # cancel modes
            if self._polygon_mode:
                self._polygon_mode = False
                self._polygon_pts = []
                self.status.set("Polygon mode canceled.")
            if self._angle_mode:
                self._disconnect_angle_handler()
                self.status.set("Angle mode canceled.")
            if self._selection_mode:
                if self._sel_rect:
                    try:
                        self._sel_rect.remove()
                    except Exception:
                        pass
                self._sel_rect = None
                self._sel_start = None
                self.canvas.draw_idle()
        elif e.key == "enter":
            if self._polygon_mode:
                self._finish_polygon()

    # ----------------------------------------------------------------------
    # SELECTION RECTANGLE LOGIC
    # ----------------------------------------------------------------------
    def _begin_selection_rect(self, event):
        self._sel_start = (event.xdata, event.ydata)
        if self._sel_rect:
            try:
                self._sel_rect.remove()
            except Exception:
                pass
        self._sel_rect = Rectangle(
            (event.xdata, event.ydata),
            0, 0,
            fill=False, edgecolor="black",
            linewidth=1.5
        )
        self.ax.add_patch(self._sel_rect)
        self.canvas.draw_idle()

    def _update_selection_rect(self, event):
        if not self._sel_rect or not self._sel_start:
            return
        x0, y0 = self._sel_start
        x1, y1 = event.xdata, event.ydata
        if x1 is None or y1 is None:
            return
        self._sel_rect.set_x(min(x0, x1))
        self._sel_rect.set_y(min(y0, y1))
        self._sel_rect.set_width(abs(x1 - x0))
        self._sel_rect.set_height(abs(y1 - y0))
        self.canvas.draw_idle()

    def _finalize_selection_rect(self):
        if not self._sel_rect:
            return
        x = self._sel_rect.get_x()
        y = self._sel_rect.get_y()
        w = self._sel_rect.get_width()
        h = self._sel_rect.get_height()

        def inside(pt):
            return x <= pt[0] <= x + w and y <= pt[1] <= y + h

        self._clear_selection()
        # points
        for p in self._points:
            if inside(p.center):
                self._add_to_selection(p)
        # labels
        for t in self._labels:
            if inside(t.get_position()):
                self._add_to_selection(t)
        # segments
        for s in self._segments:
            xs, ys = s.get_xdata(), s.get_ydata()
            if inside((xs[0], ys[0])) or inside((xs[1], ys[1])):
                self._add_to_selection(s)
        # polygons
        for shp in self._polygons:
            pts = shp.get_path().vertices
            pts = shp.get_transform().transform(pts)
            pts_data = self.ax.transData.inverted().transform(pts)
            xs = pts_data[:, 0]
            ys = pts_data[:, 1]
            if xs.min() >= x and xs.max() <= x + w and ys.min() >= y and ys.max() <= y + h:
                self._add_to_selection(shp)

        try:
            self._sel_rect.remove()
        except Exception:
            pass
        self._sel_rect = None
        self._sel_start = None
        self.canvas.draw_idle()

    def _add_to_selection(self, a):
        if a in self._selected:
            return
        self._selected.append(a)
        self._apply_selection_style(a)
        self.canvas.draw_idle()

    # ----------------------------------------------------------------------
    # TOGGLES, CLEARING, MEASUREMENTS
    # ----------------------------------------------------------------------
    def _toggle_selection_mode(self):
        self._selection_mode = not self._selection_mode
        self._clear_selection()
        if self._selection_mode:
            self.status.set("Selection mode ON. Drag to select.")
        else:
            self.status.set("Selection mode OFF.")
        self.canvas.draw_idle()

    def _clear_segments(self):
        for s in self._segments:
            try:
                s.remove()
            except Exception:
                pass
        self._segments.clear()
        # Remove segments from registry
        for oid in list(self.registry.keys()):
            if self.registry[oid]["type"] == "segment":
                del self.registry[oid]
        self._refresh_object_list()
        self.canvas.draw_idle()

    def _clear_polygons(self):
        for p in self._polygons:
            try:
                p.remove()
            except Exception:
                pass
        self._polygons.clear()
        # Remove circles/rectangles/polygons from registry
        for oid in list(self.registry.keys()):
            if self.registry[oid]["type"] in ("polygon", "circle", "rectangle"):
                del self.registry[oid]
        self._refresh_object_list()
        self.canvas.draw_idle()

    def _measure_segment_length(self):
        if len(self._selected) != 1:
            self.status.set("Select one segment.")
            return
        s = self._selected[0]
        if s not in self._segments:
            self.status.set("Not a segment.")
            return
        (x1, x2), (y1, y2) = s.get_xdata(), s.get_ydata()
        d = np.hypot(x2 - x1, y2 - y1)
        self.status.set(f"Length: {d:.4f}")

    def _segment_slope(self):
        if len(self._selected) != 1:
            self.status.set("Select one segment.")
            return
        s = self._selected[0]
        if s not in self._segments:
            self.status.set("Not a segment.")
            return
        (x1, x2), (y1, y2) = s.get_xdata(), s.get_ydata()
        if x2 == x1:
            self.status.set("Slope undefined (vertical line).")
        else:
            m = (y2 - y1) / (x2 - x1)
            self.status.set(f"Slope: {m:.4f}")

    def _polygon_area(self):
        # compute for the most recent Polygon
        for p in reversed(self._polygons):
            if isinstance(p, Polygon):
                pts = p.get_xy()
                xs = pts[:, 0]
                ys = pts[:, 1]
                area = 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
                self.status.set(f"Polygon area: {area:.4f}")
                return
        self.status.set("No polygon found.")

    def _measure_angle(self):
        if self._angle_mode:
            return
        self._angle_mode = True
        self._angle_points = []
        self.status.set("Angle mode: click 3 points. Angle at point 2. ESC to cancel.")

        def handler(event):
            if not event.inaxes:
                return
            hit = self._hit_test(event.xdata, event.ydata)
            if isinstance(hit, Circle) and hit in self._points:
                self._angle_points.append(hit)
                if len(self._angle_points) == 3:
                    p1, p2, p3 = [np.array(p.center) for p in self._angle_points]
                    v1 = p1 - p2
                    v2 = p3 - p2
                    n1 = np.linalg.norm(v1)
                    n2 = np.linalg.norm(v2)
                    if n1 == 0 or n2 == 0:
                        self.status.set("Angle undefined.")
                    else:
                        cosang = np.dot(v1, v2) / (n1 * n2)
                        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
                        self.status.set(f"Angle: {ang:.2f}°")
                    self._disconnect_angle_handler()
                    self.canvas.draw_idle()

        self._angle_cid = self.canvas.mpl_connect("button_press_event", handler)

    def _disconnect_angle_handler(self):
        if self._angle_cid is not None:
            try:
                self.canvas.mpl_disconnect(self._angle_cid)
            except Exception:
                pass
            self._angle_cid = None
        self._angle_mode = False
        self._angle_points = []

    # ----------------------------------------------------------------------
    # DELETE SELECTED
    # ----------------------------------------------------------------------
    def _delete_selected(self):
        for a in self._selected:
            # delete from registry
            for oid, data in list(self.registry.items()):
                if data["ref"] is a:
                    del self.registry[oid]
            try:
                if isinstance(a, Circle):
                    if a in self._points:
                        self._points.remove(a)
                    if a in self._polygons:  # circles are stored there for selection
                        self._polygons.remove(a)
                    a.remove()
                elif isinstance(a, Text):
                    if a in self._labels:
                        self._labels.remove(a)
                    a.remove()
                elif a in self._segments:
                    self._segments.remove(a)
                    a.remove()
                elif a in self._polygons:
                    self._polygons.remove(a)
                    a.remove()
            except Exception:
                pass
        self._clear_selection()
        self._refresh_object_list()
        self.canvas.draw_idle()

    # ----------------------------------------------------------------------
    # MISC. COMPUTATIONS
    # ----------------------------------------------------------------------
    def _update_segments(self):
        """When points move, update their connected segments."""
        for s in self._segments:
            xs = s.get_xdata()
            ys = s.get_ydata()
            p1 = self._nearest_point(xs[0], ys[0])
            p2 = self._nearest_point(xs[1], ys[1])
            if p1 and p2:
                s.set_data([p1.center[0], p2.center[0]],
                           [p1.center[1], p2.center[1]])

    def _nearest_point(self, x, y):
        for p in self._points:
            px, py = p.center
            if (px-x)**2 + (py-y)**2 < 0.15**2:
                return p
        return None


# ----------------------------------------------------------------------
# MAIN APPLICATION
# ----------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    app = InteractivePlane(root)
    root.mainloop()
