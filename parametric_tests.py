
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont
import numpy as np
# --- Optional dependencies used for computations ---
# scipy is used for distributions and t-tests
from scipy import stats  # pip install scipy

DEFAULT_DECIMALS = 6

# ---------------------- Formatting helpers (decimal comma) ----------------------
def parse_number(s: str) -> float:
    """Parse a number string robustly, accepting comma or dot decimals."""
    if s is None:
        return 0.0
    s = str(s).strip()
    if s == '' or s.lower() in {'nan', 'none'}:
        return 0.0
    s = s.replace('\u00A0', ' ').replace(' ', '')
    has_comma = ',' in s
    has_dot = '.' in s
    if has_comma and has_dot:
        last_comma = s.rfind(',')
        last_dot = s.rfind('.')
        if last_comma > last_dot:
            s = s.replace('.', '')
            s = s.replace(',', '.')
        else:
            s = s.replace(',', '')
    elif has_comma and not has_dot:
        s = s.replace(',', '.')
    try:
        return float(s)
    except Exception:
        import re
        s2 = re.sub(r'[^0-9\.\-]', '', s)
        return float(s2) if s2 not in {'', '-', '.'} else 0.0

class Formatter:
    def __init__(self, decimals=DEFAULT_DECIMALS):
        self.decimals = int(decimals)
    def fmt(self, x, places=None):
        p = self.decimals if places is None else int(places)
        try:
            val = float(x)
        except Exception:
            return str(x)
        return f"{val:.{p}f}".replace('.', ',')
    def array_to_str(self, arr: np.ndarray, places=None):
        p = self.decimals if places is None else int(places)
        def f(v):
            return self.fmt(v, p)
        return np.array2string(
            np.asarray(arr, dtype=float),
            formatter={'float_kind': lambda v: f(v)},
            max_line_width=160
        )

# --------------------------- Scrollable container ---------------------------
class ScrollableFrame(ttk.Frame):
    """A reusable vertical scrollable container for any tab content."""
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.canvas.pack(side='left', fill='both', expand=True)
        self.vsb.pack(side='right', fill='y')

        # The inner frame that will hold real content
        self.scrollable_frame = ttk.Frame(self.canvas)
        self._win_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw')

        # Resize canvas/scrollregion when inner changes
        self.scrollable_frame.bind('<Configure>', self._on_frame_configure)
        self.canvas.bind('<Configure>', self._on_canvas_configure)

        # Mouse wheel support (Windows/Linux/Mac)
        self._bind_mousewheel()

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def _on_canvas_configure(self, event):
        # Make inner frame width match canvas width
        self.canvas.itemconfigure(self._win_id, width=event.width)

    def _bind_mousewheel(self):
        # Windows and Linux
        self.canvas.bind_all('<MouseWheel>', self._on_mousewheel)
        # macOS (older events)
        self.canvas.bind_all('<Button-4>', lambda e: self.canvas.yview_scroll(-1, 'units'))
        self.canvas.bind_all('<Button-5>', lambda e: self.canvas.yview_scroll(1, 'units'))

    def _on_mousewheel(self, event):
        # event.delta is typically multiples of 120 on Windows
        delta = -1 * int(event.delta / 120) if event.delta != 0 else 0
        self.canvas.yview_scroll(delta, 'units')

# --------------------------- Reusable grid editor ---------------------------
class SheetEditor(ttk.Frame):
    """Lightweight grid editor for small numeric tables (1 or 2 columns)."""
    def __init__(self, parent, rows=10, cols=1, lock_size=False, col_headers=None):
        super().__init__(parent)
        self.lock_size = bool(lock_size)
        self._rows = max(1, int(rows))
        self._cols = max(1, int(cols))
        self._cells = []
        self._mono = tkfont.Font(
            family='Consolas' if 'Consolas' in tkfont.families() else 'Courier New',
            size=10
        )
        # Toolbar
        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 4))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        ttk.Label(bar, text='Rows:').pack(side='left')
        self.rows_var = tk.IntVar(value=self._rows)
        self.cols_var = tk.IntVar(value=self._cols)
        self.spin_r = ttk.Spinbox(
            bar, from_=1, to=2000, width=5, textvariable=self.rows_var,
            command=self._apply_size, state=('disabled' if self.lock_size else 'normal')
        )
        self.spin_r.pack(side='left', padx=(4, 10))
        ttk.Label(bar, text='Cols:').pack(side='left')
        self.spin_c = ttk.Spinbox(
            bar, from_=1, to=10, width=4, textvariable=self.cols_var,
            command=self._apply_size, state=('disabled' if self.lock_size else 'normal')
        )
        self.spin_c.pack(side='left', padx=(4, 10))
        ttk.Button(bar, text='Paste', command=self.paste_block).pack(side='left', padx=(0, 6))
        ttk.Button(bar, text='Copy', command=self.copy_block).pack(side='left', padx=(0, 12))
        ttk.Button(bar, text='Load CSV…', command=self.load_csv).pack(side='left', padx=(0, 6))
        ttk.Button(bar, text='Load Excel…', command=self.load_excel).pack(side='left', padx=(0, 6))
        ttk.Button(bar, text='Clear', command=self.clear).pack(side='left', padx=(0, 6))

        # Grid container with headers
        grid = ttk.Frame(self)
        grid.grid(row=2, column=0, sticky='nsew')
        self._grid = grid
        grid.columnconfigure(0, weight=0)
        for j in range(1, self._cols + 1):
            grid.columnconfigure(j, weight=1)
        for i in range(1, self._rows + 1):
            grid.rowconfigure(i, weight=0)

        self._hdr_bg = '#f2f2f7'
        self._col_headers = list(col_headers) if col_headers else None
        self._make_headers()
        self._build_cells()

        # Selection state
        self._sel = (0, 0)
        self._select_cell(0, 0)

        # Shortcuts
        self.bind_all('<Control-c>', lambda e: self.copy_block(), add='+')
        self.bind_all('<Control-C>', lambda e: self.copy_block(), add='+')
        self.bind_all('<Control-v>', lambda e: self.paste_block(), add='+')
        self.bind_all('<Control-V>', lambda e: self.paste_block(), add='+')
        self.bind_all('<Control-l>', lambda e: self.load_csv(), add='+')
        self.bind_all('<Control-L>', lambda e: self.load_csv(), add='+')
        self.bind_all('<Control-e>', lambda e: self.load_excel(), add='+')
        self.bind_all('<Control-E>', lambda e: self.load_excel(), add='+')
        self.bind_all('<Control-k>', lambda e: self.clear(), add='+')
        self.bind_all('<Control-K>', lambda e: self.clear(), add='+')

    def _make_headers(self):
        for child in list(self._grid.children.values()):
            child.destroy()
        # corner
        hdr_tl = ttk.Label(self._grid, text='', background=self._hdr_bg)
        hdr_tl.grid(row=0, column=0, sticky='nsew')
        # columns
        for c in range(self._cols):
            txt = self._col_headers[c] if self._col_headers and c < len(self._col_headers) else f"C{c+1}"
            l = ttk.Label(self._grid, text=txt, background=self._hdr_bg, anchor='center')
            l.grid(row=0, column=c + 1, sticky='nsew', padx=1, pady=1)
        # rows
        for r in range(self._rows):
            l = ttk.Label(self._grid, text=str(r + 1), background=self._hdr_bg, anchor='e')
            l.grid(row=r + 1, column=0, sticky='nsew', padx=1, pady=1)

    def _build_cells(self):
        for child in list(self._grid.children.values()):
            info = child.grid_info()
            if info and int(info.get('row', 0)) > 0 and int(info.get('column', 0)) > 0:
                child.destroy()
        self._cells = []
        for r in range(self._rows):
            row = []
            for c in range(self._cols):
                e = ttk.Entry(self._grid, width=12, justify='center', font=self._mono)
                e.grid(row=r + 1, column=c + 1, padx=1, pady=1, sticky='nsew')
                e.insert(0, '')
                e.bind('<Button-1>', lambda ev, rr=r, cc=c: self._select_cell(rr, cc))
                e.bind('<Shift-Button-1>', lambda ev, rr=r, cc=c: self._select_cell(rr, cc))
                e.bind('<Return>', lambda ev, rr=r, cc=c: (self._move_focus(rr + 1, cc), 'break'))  # Excel: Enter moves down
                e.bind('<KP_Enter>', lambda ev, rr=r, cc=c: (self._move_focus(rr + 1, cc), 'break'))
                e.bind('<Tab>', lambda ev, rr=r, cc=c: (self._move_focus(rr, cc + 1), 'break'))
                e.bind('<ISO_Left_Tab>', lambda ev, rr=r, cc=c: (self._move_focus(rr, cc - 1), 'break'))
                e.bind('<Shift-Tab>', lambda ev, rr=r, cc=c: (self._move_focus(rr, cc - 1), 'break'))
                e.bind('<Up>', lambda ev, rr=r, cc=c: (self._move_focus(rr - 1, cc), 'break'))
                e.bind('<Down>', lambda ev, rr=r, cc=c: (self._move_focus(rr + 1, cc), 'break'))
                e.bind('<Left>', lambda ev, rr=r, cc=c: (self._move_focus(rr, cc - 1), 'break'))
                e.bind('<Right>', lambda ev, rr=r, cc=c: (self._move_focus(rr, cc + 1), 'break'))
                row.append(e)
            self._cells.append(row)

    def _apply_size(self):
        if self.lock_size:
            return
        r = max(1, int(self.rows_var.get()))
        c = max(1, int(self.cols_var.get()))
        if r != self._rows or c != self._cols:
            self._rows, self._cols = r, c
            self._make_headers()
            self._build_cells()
            self._select_cell(0, 0)

    def _select_cell(self, r, c):
        r = max(0, min(self._rows - 1, r))
        c = max(0, min(self._cols - 1, c))
        self._sel = (r, c)
        self._cells[r][c].focus_set()
        self._cells[r][c].selection_range(0, 'end')

    def _move_focus(self, r, c):
        r = max(0, min(self._rows - 1, r))
        c = max(0, min(self._cols - 1, c))
        self._select_cell(r, c)

    def read_column(self, idx: int):
        """Read a single column as a numeric vector, ignoring blanks."""
        idx = int(idx)
        if not (0 <= idx < self._cols):
            raise ValueError("Column index out of range.")
        vals = []
        for i in range(self._rows):
            s = self._cells[i][idx].get().strip()
            if s == '':
                continue
            vals.append(parse_number(s))
        if len(vals) == 0:
            raise ValueError("No numeric data in the selected column.")
        return np.asarray(vals, dtype=float)

    def read_two_columns(self):
        if self._cols < 2:
            raise ValueError("Grid has fewer than 2 columns.")
        a = self.read_column(0)
        b = self.read_column(1)
        # Align by length if needed (truncate to min length)
        n = min(len(a), len(b))
        if n == 0:
            raise ValueError("No paired data found.")
        return a[:n], b[:n]

    def clear(self):
        for i in range(self._rows):
            for j in range(self._cols):
                e = self._cells[i][j]
                e.delete(0, 'end')

    def paste_block(self):
        """Paste cells copied from Excel/Sheets (tab-separated). Falls back to CSV/space."""
        try:
            raw = self.clipboard_get()
        except Exception:
            messagebox.showwarning('Paste', 'Nothing on clipboard.')
            return

        # Normalize line endings
        raw = raw.replace('\r\n', '\n').replace('\r', '\n')
        lines = [ln for ln in raw.strip().split('\n') if ln.strip()]
        data = []

        # Prefer Excel/Sheets format: TAB-separated
        if any('\t' in ln for ln in lines):
            for ln in lines:
                parts = ln.split('\t')
                data.append([p for p in parts])
        else:
            # Fallbacks: CSV (comma) or whitespace-separated
            for ln in lines:
                if ',' in ln:
                    parts = ln.split(',')
                else:
                    parts = ln.split()
                data.append([p.strip() for p in parts])

        if not data:
            messagebox.showwarning('Paste', 'Clipboard does not look like a table.')
            return

        # Expand grid if needed (only if not locked)
        if not self.lock_size:
            need_rows = max(self._rows, len(data))
            need_cols = max(self._cols, max(len(r) for r in data))
            self.rows_var.set(need_rows)
            self.cols_var.set(need_cols)
            self._apply_size()

        r0, c0 = self._sel
        for i, row in enumerate(data):
            ri = r0 + i
            if ri >= self._rows:
                break
            for j, val in enumerate(row):
                cj = c0 + j
                if cj >= self._cols:
                    break
                e = self._cells[ri][cj]
                e.delete(0, 'end')
                e.insert(0, val)

    def copy_block(self):
        # Copy whole grid as TAB-separated table (Excel-friendly)
        buf_lines = []
        for i in range(self._rows):
            buf_lines.append('\t'.join(self._cells[i][j].get() for j in range(self._cols)))
        txt = '\n'.join(buf_lines)
        self.clipboard_clear()
        self.clipboard_append(txt)
        try:
            self.update()
        except Exception:
            pass
        messagebox.showinfo('Copy', 'Grid copied to clipboard (tab-separated).')

    def load_csv(self):
        path = filedialog.askopenfilename(
            title='Load CSV',
            filetypes=[('CSV', '*.csv'), ('All files', '*.*')]
        )
        if not path:
            return
        try:
            import pandas as pd
            df = pd.read_csv(path, header=None, dtype=str)
            values = df.values
            if not self.lock_size:
                self.rows_var.set(max(self._rows, values.shape[0]))
                self.cols_var.set(max(self._cols, values.shape[1]))
                self._apply_size()
            for i in range(min(self._rows, values.shape[0])):
                for j in range(min(self._cols, values.shape[1])):
                    e = self._cells[i][j]
                    e.delete(0, 'end')
                    e.insert(0, str(values[i, j]))
        except Exception as e:
            messagebox.showerror('CSV', f'Could not load CSV: {e}')

    def load_excel(self):
        path = filedialog.askopenfilename(
            title='Load Excel',
            filetypes=[('Excel', '*.xlsx *.xls'), ('All files', '*.*')]
        )
        if not path:
            return
        try:
            import pandas as pd
            ext = path.lower().rsplit('.', 1)[-1]
            if ext == 'xlsx':
                df = pd.read_excel(path, header=None, engine='openpyxl', dtype=str)
            elif ext == 'xls':
                df = pd.read_excel(path, header=None, engine='xlrd', dtype=str)
            else:
                df = pd.read_excel(path, header=None, dtype=str)
            values = df.values
            if not self.lock_size:
                self.rows_var.set(max(self._rows, values.shape[0]))
                self.cols_var.set(max(self._cols, values.shape[1]))
                self._apply_size()
            for i in range(min(self._rows, values.shape[0])):
                for j in range(min(self._cols, values.shape[1])):
                    e = self._cells[i][j]
                    e.delete(0, 'end')
                    e.insert(0, str(values[i, j]))
        except Exception as e:
            messagebox.showerror('Excel', f'Could not load Excel: {e}')

# ---------------- Utility: CI helpers and effect sizes ----------------
def t_ci_mean(mean, sd, n, level=0.95):
    alpha = 1 - float(level)
    df = max(1, int(n) - 1)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    return float(mean - tcrit * se), float(mean + tcrit * se)

def welch_df(s1, n1, s2, n2):
    v1 = (s1 ** 2) / n1
    v2 = (s2 ** 2) / n2
    num = (v1 + v2) ** 2
    den = (v1 ** 2) / (n1 - 1) + (v2 ** 2) / (n2 - 1)
    return num / den if den > 0 else np.nan

def ci_diff_means(mean1, sd1, n1, mean2, sd2, n2, level=0.95, equal_var=False):
    alpha = 1 - float(level)
    if equal_var:
        # pooled
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / df
        se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
        tcrit = stats.t.ppf(1 - alpha / 2, df)
    else:
        # Welch
        df = welch_df(sd1, n1, sd2, n2)
        se = np.sqrt(sd1 ** 2 / n1 + sd2 ** 2 / n2)
        tcrit = stats.t.ppf(1 - alpha / 2, df)
    diff = mean1 - mean2
    return float(diff - tcrit * se), float(diff + tcrit * se), float(df)

def z_ci_mean(mean, sigma, n, level=0.95):
    alpha = 1 - float(level)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    se = sigma / np.sqrt(n)
    return float(mean - zcrit * se), float(mean + zcrit * se)

def z_ci_diff_means(mean1, sigma1, n1, mean2, sigma2, n2, level=0.95):
    alpha = 1 - float(level)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt((sigma1 ** 2) / n1 + (sigma2 ** 2) / n2)
    diff = mean1 - mean2
    return float(diff - zcrit * se), float(diff + zcrit * se)

def wald_ci_prop(p, n, level=0.95):
    alpha = 1 - float(level)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(p * (1 - p) / n)
    return float(p - zcrit * se), float(p + zcrit * se)

def wald_ci_diff_props(p1, n1, p2, n2, level=0.95):
    alpha = 1 - float(level)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    return float(diff - zcrit * se), float(diff + zcrit * se)

# -------------------------------- Main window --------------------------------
class ParametricTestsWindow(tk.Toplevel):
    def __init__(self, master=None, title='Parametric & Proportion Tests'):
        super().__init__(master=master)
        self.title(title)
        self.geometry('1200x800')
        self.minsize(1060, 680)
        self.configure(padx=8, pady=8)
        self._mono = tkfont.Font(
            family='Consolas' if 'Consolas' in tkfont.families() else 'Courier New',
            size=10
        )
        self.formatter = Formatter(DEFAULT_DECIMALS)

        container = ttk.Frame(self)
        container.pack(fill='both', expand=True)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        self.nb = ttk.Notebook(container)
        self.nb.grid(row=0, column=0, sticky='nsew')

        # Tabs (each wrapped in ScrollableFrame)
        # t-test: one-sample
        self._t1_wrap = ScrollableFrame(self.nb)
        self.tab_t1 = self._t1_wrap.scrollable_frame
        # t-test: two-sample
        self._t2_wrap = ScrollableFrame(self.nb)
        self.tab_t2 = self._t2_wrap.scrollable_frame
        # t-test: paired
        self._tp_wrap = ScrollableFrame(self.nb)
        self.tab_tp = self._tp_wrap.scrollable_frame
        # z-test: one-sample (mean, sigma known)
        self._z1_wrap = ScrollableFrame(self.nb)
        self.tab_z1 = self._z1_wrap.scrollable_frame
        # z-test: two-sample (means, sigmas known)
        self._z2_wrap = ScrollableFrame(self.nb)
        self.tab_z2 = self._z2_wrap.scrollable_frame
        # z-test: one proportion
        self._p1_wrap = ScrollableFrame(self.nb)
        self.tab_p1 = self._p1_wrap.scrollable_frame
        # z-test: two proportions
        self._p2_wrap = ScrollableFrame(self.nb)
        self.tab_p2 = self._p2_wrap.scrollable_frame

        self.nb.add(self._t1_wrap, text='t-test: One-sample')
        self.nb.add(self._t2_wrap, text='t-test: Two-sample')
        self.nb.add(self._tp_wrap, text='t-test: Paired')
        self.nb.add(self._z1_wrap, text='z-test: One-sample (mean, σ known)')
        self.nb.add(self._z2_wrap, text='z-test: Two-sample (means, σ known)')
        self.nb.add(self._p1_wrap, text='z-test: One proportion')
        self.nb.add(self._p2_wrap, text='z-test: Two proportions')

        # Build tabs
        self._build_t1_tab(self.tab_t1)
        self._build_t2_tab(self.tab_t2)
        self._build_tp_tab(self.tab_tp)
        self._build_z1_tab(self.tab_z1)
        self._build_z2_tab(self.tab_z2)
        self._build_p1_tab(self.tab_p1)
        self._build_p2_tab(self.tab_p2)

        self.status_var = tk.StringVar(value='Ready')
        ttk.Label(self, textvariable=self.status_var, anchor='w').pack(side='bottom', fill='x')
        self.after(100, self._center_on_screen)

    def _center_on_screen(self):
        self.update_idletasks()
        w = self.winfo_width()
        h = self.winfo_height()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = int((sw - w) / 2)
        y = int((sh - h) / 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _set_status(self, msg):
        self.status_var.set(msg)

    # ---------------------------- Common UI helpers ----------------------------
    def _add_summary_panel(self, parent, row_upper=3, row_summary=4):
        """Configure rows and add a scrollable summary text widget."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(row_upper, weight=1, minsize=120)   # upper area
        parent.rowconfigure(row_summary, weight=5, minsize=220) # summary bigger
        summ = ttk.Labelframe(parent, text='Summary')
        summ.grid(row=row_summary, column=0, sticky='nsew', pady=(8, 0))
        summ_container = ttk.Frame(summ)
        summ_container.pack(fill='both', expand=True)
        text = tk.Text(summ_container, wrap='word', font=self._mono, height=12)
        vscroll = ttk.Scrollbar(summ_container, orient='vertical', command=text.yview)
        text.configure(yscrollcommand=vscroll.set)
        text.pack(side='left', fill='both', expand=True)
        vscroll.pack(side='right', fill='y')
        return text

    def _alt_values(self):
        return ['two-sided', 'less', 'greater']

    def _parse_level(self, s, default=0.95):
        try:
            v = float(parse_number(s))
            if 0.0 < v < 1.0:
                return v
            return default
        except Exception:
            return default

    # ------------------------------ t-test: one-sample ------------------------------
    def _build_t1_tab(self, parent):
        opts = ttk.Frame(parent)
        opts.grid(row=0, column=0, sticky='ew', pady=(0, 6))
        ttk.Label(opts, text='Input:').pack(side='left')
        self.t1_mode = tk.StringVar(value='raw')
        ttk.Combobox(opts, textvariable=self.t1_mode, state='readonly', width=12,
                     values=['raw', 'summary']).pack(side='left', padx=(4, 12))
        ttk.Label(opts, text='μ₀:').pack(side='left')
        self.t1_mu0 = tk.StringVar(value='0')
        ttk.Entry(opts, textvariable=self.t1_mu0, width=10).pack(side='left', padx=(4, 12))
        ttk.Label(opts, text='Alternative:').pack(side='left')
        self.t1_alt = tk.StringVar(value='two-sided')
        ttk.Combobox(opts, textvariable=self.t1_alt, state='readonly', width=10,
                     values=self._alt_values()).pack(side='left', padx=(4, 12))
        ttk.Label(opts, text='CI level:').pack(side='left')
        self.t1_ci = tk.StringVar(value='0,95')
        ttk.Entry(opts, textvariable=self.t1_ci, width=8).pack(side='left', padx=(4, 12))
        ttk.Label(opts, text='Decimals:').pack(side='left')
        self.dec_t1 = tk.IntVar(value=DEFAULT_DECIMALS)
        ttk.Spinbox(opts, from_=0, to=12, width=4, textvariable=self.dec_t1,
                    command=lambda: None).pack(side='left', padx=(4, 12))

        # Raw data editor (one column)
        self.t1_sheet = SheetEditor(parent, rows=10, cols=1, lock_size=False, col_headers=['X'])
        self.t1_sheet.grid(row=1, column=0, sticky='ew')

        # Summary stats panel
        sumf = ttk.Labelframe(parent, text='Summary stats (use when Input="summary")')
        sumf.grid(row=2, column=0, sticky='ew', pady=(6, 0))
        for i in range(7):
            sumf.columnconfigure(i, weight=0)
        ttk.Label(sumf, text='n:').grid(row=0, column=0, sticky='w', padx=(8, 2), pady=4)
        ttk.Label(sumf, text='mean:').grid(row=0, column=2, sticky='w', padx=(8, 2))
        ttk.Label(sumf, text='sd:').grid(row=0, column=4, sticky='w', padx=(8, 2))
        self.t1_n = tk.StringVar(value='')
        self.t1_mean = tk.StringVar(value='')
        self.t1_sd = tk.StringVar(value='')
        ttk.Entry(sumf, width=10, textvariable=self.t1_n).grid(row=0, column=1, padx=(0, 8))
        ttk.Entry(sumf, width=10, textvariable=self.t1_mean).grid(row=0, column=3, padx=(0, 8))
        ttk.Entry(sumf, width=10, textvariable=self.t1_sd).grid(row=0, column=5, padx=(0, 8))

        actions = ttk.Frame(parent)
        actions.grid(row=3, column=0, sticky='ew', pady=(6, 0))
        ttk.Button(actions, text='Compute', command=self._compute_t1).pack(side='left')

        self.t1_text = self._add_summary_panel(parent, row_upper=3, row_summary=4)

    def _compute_t1(self):
        self.formatter.decimals = int(self.dec_t1.get())
        mu0 = parse_number(self.t1_mu0.get())
        level = self._parse_level(self.t1_ci.get(), 0.95)
        alt = self.t1_alt.get().strip()
        try:
            mode = self.t1_mode.get()
            if mode == 'raw':
                x = self.t1_sheet.read_column(0)
                n = len(x)
                mean = float(np.mean(x))
                sd = float(np.std(x, ddof=1)) if n > 1 else np.nan
                res = stats.ttest_1samp(x, popmean=mu0, alternative=alt)
                df = n - 1
                if n > 1 and np.isfinite(sd):
                    ci_lo, ci_hi = t_ci_mean(mean, sd, n, level)
                else:
                    ci_lo = ci_hi = np.nan
                cohen_d = (mean - mu0) / sd if sd > 0 else np.nan
                hedges_g = cohen_d * (1 - 3 / (4 * df - 1)) if df > 1 and np.isfinite(cohen_d) else np.nan
            else:
                n = int(parse_number(self.t1_n.get()))
                mean = parse_number(self.t1_mean.get())
                sd = parse_number(self.t1_sd.get())
                if n <= 1 or sd <= 0:
                    raise ValueError("Summary: need n>1 and sd>0.")
                df = n - 1
                se = sd / np.sqrt(n)
                tstat = (mean - mu0) / se
                # p-value with alternative
                if alt == 'two-sided':
                    p = 2 * (1 - stats.t.cdf(abs(tstat), df))
                elif alt == 'greater':
                    p = 1 - stats.t.cdf(tstat, df)
                else:
                    p = stats.t.cdf(tstat, df)
                res = type('obj', (object,), {'statistic': tstat, 'pvalue': p})
                ci_lo, ci_hi = t_ci_mean(mean, sd, n, level)
                cohen_d = (mean - mu0) / sd
                hedges_g = cohen_d * (1 - 3 / (4 * df - 1)) if df > 1 else np.nan

            lines = []
            lines.append("One-sample t-test")
            lines.append(f" n = {self.formatter.fmt(n,0)}, mean = {self.formatter.fmt(mean)}, sd = {self.formatter.fmt(sd)}")
            lines.append(f" H0: μ = {self.formatter.fmt(mu0)}; alternative = {alt}")
            lines.append(f" t = {self.formatter.fmt(res.statistic)}, df = {int(df)}; p = {self.formatter.fmt(res.pvalue)}")
            lines.append(f" {self.formatter.fmt(level*100,1)}% CI for μ: [{self.formatter.fmt(ci_lo)}, {self.formatter.fmt(ci_hi)}]")
            lines.append(f" Effect size: Cohen's d = {self.formatter.fmt(cohen_d)}, Hedges' g = {self.formatter.fmt(hedges_g)}")
            self.t1_text.delete('1.0', 'end')
            self.t1_text.insert('1.0', '\n'.join(lines))
            self._set_status('One-sample t-test computed')
        except Exception as e:
            messagebox.showerror('t-test (one-sample)', str(e))

    # --------------------------- t-test: two-sample (independent) ---------------------------
    def _build_t2_tab(self, parent):
        opts = ttk.Frame(parent)
        opts.grid(row=0, column=0, sticky='ew', pady=(0, 6))
        ttk.Label(opts, text='Input:').pack(side='left')
        self.t2_mode = tk.StringVar(value='raw')
        ttk.Combobox(opts, textvariable=self.t2_mode, state='readonly', width=12,
                     values=['raw', 'summary']).pack(side='left', padx=(4, 12))
        self.t2_equal = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text='Assume equal variances (Student)', variable=self.t2_equal).pack(side='left', padx=(0,12))
        ttk.Label(opts, text='Alternative:').pack(side='left')
        self.t2_alt = tk.StringVar(value='two-sided')
        ttk.Combobox(opts, textvariable=self.t2_alt, state='readonly', width=10,
                     values=self._alt_values()).pack(side='left', padx=(4, 12))
        ttk.Label(opts, text='CI level:').pack(side='left')
        self.t2_ci = tk.StringVar(value='0,95')
        ttk.Entry(opts, textvariable=self.t2_ci, width=8).pack(side='left', padx=(4, 12))
        ttk.Label(opts, text='Decimals:').pack(side='left')
        self.dec_t2 = tk.IntVar(value=DEFAULT_DECIMALS)
        ttk.Spinbox(opts, from_=0, to=12, width=4, textvariable=self.dec_t2).pack(side='left', padx=(4, 12))

        self.t2_sheet = SheetEditor(parent, rows=10, cols=2, lock_size=False, col_headers=['Group A','Group B'])
        self.t2_sheet.grid(row=1, column=0, sticky='ew')

        # Summary stats
        sumf = ttk.Labelframe(parent, text='Summary stats (use when Input="summary")')
        sumf.grid(row=2, column=0, sticky='ew', pady=(6,0))
        # group A
        ttk.Label(sumf, text='A: n').grid(row=0, column=0, padx=(8,2), pady=4, sticky='w')
        ttk.Label(sumf, text='mean').grid(row=0, column=2, padx=(8,2), sticky='w')
        ttk.Label(sumf, text='sd').grid(row=0, column=4, padx=(8,2), sticky='w')
        self.t2_n1 = tk.StringVar(); self.t2_m1 = tk.StringVar(); self.t2_s1 = tk.StringVar()
        ttk.Entry(sumf, width=10, textvariable=self.t2_n1).grid(row=0, column=1)
        ttk.Entry(sumf, width=10, textvariable=self.t2_m1).grid(row=0, column=3)
        ttk.Entry(sumf, width=10, textvariable=self.t2_s1).grid(row=0, column=5)
        # group B
        ttk.Label(sumf, text='B: n').grid(row=1, column=0, padx=(8,2), pady=4, sticky='w')
        ttk.Label(sumf, text='mean').grid(row=1, column=2, padx=(8,2), sticky='w')
        ttk.Label(sumf, text='sd').grid(row=1, column=4, padx=(8,2), sticky='w')
        self.t2_n2 = tk.StringVar(); self.t2_m2 = tk.StringVar(); self.t2_s2 = tk.StringVar()
        ttk.Entry(sumf, width=10, textvariable=self.t2_n2).grid(row=1, column=1)
        ttk.Entry(sumf, width=10, textvariable=self.t2_m2).grid(row=1, column=3)
        ttk.Entry(sumf, width=10, textvariable=self.t2_s2).grid(row=1, column=5)

        actions = ttk.Frame(parent); actions.grid(row=3, column=0, sticky='ew', pady=(6,0))
        ttk.Button(actions, text='Compute', command=self._compute_t2).pack(side='left')

        self.t2_text = self._add_summary_panel(parent, row_upper=3, row_summary=4)

    def _compute_t2(self):
        self.formatter.decimals = int(self.dec_t2.get())
        level = self._parse_level(self.t2_ci.get(), 0.95)
        alt = self.t2_alt.get().strip()
        equal = bool(self.t2_equal.get())
        try:
            mode = self.t2_mode.get()
            if mode == 'raw':
                a, b = self.t2_sheet.read_two_columns()
                res = stats.ttest_ind(a, b, alternative=alt, equal_var=equal)
                n1, n2 = len(a), len(b)
                m1, m2 = float(np.mean(a)), float(np.mean(b))
                s1, s2 = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
                # CI for difference and df
                ci_lo, ci_hi, df = ci_diff_means(m1, s1, n1, m2, s2, n2, level, equal_var=equal)
                # Effect sizes (Cohen's d pooled; Hedges' g)
                if equal:
                    df_p = n1 + n2 - 2
                    sp2 = ((n1-1)*s1**2 + (n2-1)*s2**2)/df_p
                    sp = np.sqrt(sp2)
                    d = (m1 - m2) / sp if sp > 0 else np.nan
                    g = d * (1 - 3/(4*df_p - 1)) if df_p > 1 and np.isfinite(d) else np.nan
                else:
                    # Use Glass's delta (A) as a fallback when variances unequal
                    d = (m1 - m2) / s2 if s2 > 0 else np.nan
                    g = np.nan
            else:
                n1 = int(parse_number(self.t2_n1.get()))
                m1 = parse_number(self.t2_m1.get())
                s1 = parse_number(self.t2_s1.get())
                n2 = int(parse_number(self.t2_n2.get()))
                m2 = parse_number(self.t2_m2.get())
                s2 = parse_number(self.t2_s2.get())
                if min(n1, n2) <= 1 or min(s1, s2) <= 0:
                    raise ValueError("Summary: need n1>1, n2>1 and sd>0.")
                # t statistic
                if equal:
                    df = n1 + n2 - 2
                    sp2 = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / df
                    se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
                else:
                    df = welch_df(s1, n1, s2, n2)
                    se = np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
                tstat = (m1 - m2) / se
                if alt == 'two-sided':
                    p = 2 * (1 - stats.t.cdf(abs(tstat), df))
                elif alt == 'greater':
                    p = 1 - stats.t.cdf(tstat, df)
                else:
                    p = stats.t.cdf(tstat, df)
                res = type('obj', (object,), {'statistic': tstat, 'pvalue': p})
                ci_lo, ci_hi, df = ci_diff_means(m1, s1, n1, m2, s2, n2, level, equal_var=equal)
                if equal:
                    df_p = n1 + n2 - 2
                    sp2 = ((n1-1)*s1**2 + (n2-1)*s2**2)/df_p
                    sp = np.sqrt(sp2)
                    d = (m1 - m2) / sp if sp > 0 else np.nan
                    g = d * (1 - 3/(4*df_p - 1)) if df_p > 1 and np.isfinite(d) else np.nan
                else:
                    d = (m1 - m2) / s2 if s2 > 0 else np.nan
                    g = np.nan

            lines = []
            lines.append(f"Two-sample t-test ({'equal var' if equal else 'Welch unequal var'})")
            lines.append(f" n1 = {self.formatter.fmt(n1,0)}, mean1 = {self.formatter.fmt(m1)}, sd1 = {self.formatter.fmt(s1)}")
            lines.append(f" n2 = {self.formatter.fmt(n2,0)}, mean2 = {self.formatter.fmt(m2)}, sd2 = {self.formatter.fmt(s2)}")
            lines.append(f" alternative = {alt}")
            lines.append(f" t = {self.formatter.fmt(res.statistic)}, df = {self.formatter.fmt(df,2)}; p = {self.formatter.fmt(res.pvalue)}")
            lines.append(f" {self.formatter.fmt(level*100,1)}% CI for (mean1 - mean2): [{self.formatter.fmt(ci_lo)}, {self.formatter.fmt(ci_hi)}]")
            lines.append(f" Effect size: Cohen's d = {self.formatter.fmt(d)}; Hedges' g = {self.formatter.fmt(g)}")
            self.t2_text.delete('1.0', 'end')
            self.t2_text.insert('1.0', '\n'.join(lines))
            self._set_status('Two-sample t-test computed')
        except Exception as e:
            messagebox.showerror('t-test (two-sample)', str(e))

    # -------------------------------- t-test: paired --------------------------------
    def _build_tp_tab(self, parent):
        opts = ttk.Frame(parent); opts.grid(row=0, column=0, sticky='ew', pady=(0, 6))
        ttk.Label(opts, text='Input:').pack(side='left')
        self.tp_mode = tk.StringVar(value='raw')
        ttk.Combobox(opts, textvariable=self.tp_mode, state='readonly', width=12,
                     values=['raw', 'summary']).pack(side='left', padx=(4, 12))
        ttk.Label(opts, text='Alternative:').pack(side='left')
        self.tp_alt = tk.StringVar(value='two-sided')
        ttk.Combobox(opts, textvariable=self.tp_alt, state='readonly', width=10,
                     values=self._alt_values()).pack(side='left', padx=(4, 12))
        ttk.Label(opts, text='CI level:').pack(side='left')
        self.tp_ci = tk.StringVar(value='0,95')
        ttk.Entry(opts, textvariable=self.tp_ci, width=8).pack(side='left', padx=(4, 12))
        ttk.Label(opts, text='Decimals:').pack(side='left')
        self.dec_tp = tk.IntVar(value=DEFAULT_DECIMALS)
        ttk.Spinbox(opts, from_=0, to=12, width=4, textvariable=self.dec_tp).pack(side='left', padx=(4, 12))

        self.tp_sheet = SheetEditor(parent, rows=10, cols=2, lock_size=False, col_headers=['Before','After'])
        self.tp_sheet.grid(row=1, column=0, sticky='ew')

        sumf = ttk.Labelframe(parent, text='Summary stats of differences D=(After-Before) (use when Input="summary")')
        sumf.grid(row=2, column=0, sticky='ew', pady=(6, 0))
        ttk.Label(sumf, text='n:').grid(row=0, column=0, padx=(8,2), pady=4, sticky='w')
        ttk.Label(sumf, text='mean(D):').grid(row=0, column=2, padx=(8,2), sticky='w')
        ttk.Label(sumf, text='sd(D):').grid(row=0, column=4, padx=(8,2), sticky='w')
        self.tp_n = tk.StringVar(); self.tp_md = tk.StringVar(); self.tp_sd = tk.StringVar()
        ttk.Entry(sumf, width=10, textvariable=self.tp_n).grid(row=0, column=1)
        ttk.Entry(sumf, width=10, textvariable=self.tp_md).grid(row=0, column=3)
        ttk.Entry(sumf, width=10, textvariable=self.tp_sd).grid(row=0, column=5)

        actions = ttk.Frame(parent); actions.grid(row=3, column=0, sticky='ew', pady=(6,0))
        ttk.Button(actions, text='Compute', command=self._compute_tp).pack(side='left')

        self.tp_text = self._add_summary_panel(parent, row_upper=3, row_summary=4)

    def _compute_tp(self):
        self.formatter.decimals = int(self.dec_tp.get())
        level = self._parse_level(self.tp_ci.get(), 0.95)
        alt = self.tp_alt.get().strip()
        try:
            mode = self.tp_mode.get()
            if mode == 'raw':
                before, after = self.tp_sheet.read_two_columns()
                # enforce pairing
                n = min(len(before), len(after))
                diff = after[:n] - before[:n]
                md = float(np.mean(diff))
                sd = float(np.std(diff, ddof=1)) if n > 1 else np.nan
                res = stats.ttest_rel(before[:n], after[:n], alternative=alt)
                df = n - 1
                if n > 1 and np.isfinite(sd):
                    ci_lo, ci_hi = t_ci_mean(md, sd, n, level)
                else:
                    ci_lo = ci_hi = np.nan
                dz = md / sd if sd > 0 else np.nan
            else:
                n = int(parse_number(self.tp_n.get()))
                md = parse_number(self.tp_md.get())
                sd = parse_number(self.tp_sd.get())
                if n <= 1 or sd <= 0:
                    raise ValueError("Summary: need n>1 and sd(D)>0.")
                df = n - 1
                se = sd / np.sqrt(n)
                tstat = md / se
                if alt == 'two-sided':
                    p = 2 * (1 - stats.t.cdf(abs(tstat), df))
                elif alt == 'greater':
                    p = 1 - stats.t.cdf(tstat, df)
                else:
                    p = stats.t.cdf(tstat, df)
                res = type('obj', (object,), {'statistic': tstat, 'pvalue': p})
                ci_lo, ci_hi = t_ci_mean(md, sd, n, level)
                dz = md / sd

            lines = []
            lines.append("Paired t-test")
            lines.append(f" n = {self.formatter.fmt(n,0)}, mean(D) = {self.formatter.fmt(md)}, sd(D) = {self.formatter.fmt(sd)}")
            lines.append(f" H0: mean(D)=0; alternative = {alt}")
            lines.append(f" t = {self.formatter.fmt(res.statistic)}, df = {int(df)}; p = {self.formatter.fmt(res.pvalue)}")
            lines.append(f" {self.formatter.fmt(level*100,1)}% CI for mean(D): [{self.formatter.fmt(ci_lo)}, {self.formatter.fmt(ci_hi)}]")
            lines.append(f" Effect size: dz = {self.formatter.fmt(dz)}")
            self.tp_text.delete('1.0', 'end')
            self.tp_text.insert('1.0', '\n'.join(lines))
            self._set_status('Paired t-test computed')
        except Exception as e:
            messagebox.showerror('t-test (paired)', str(e))

    # ------------------------- z-test: one-sample mean (σ known) -------------------------
    def _build_z1_tab(self, parent):
        opts = ttk.Frame(parent); opts.grid(row=0, column=0, sticky='ew', pady=(0,6))
        ttk.Label(opts, text='Input:').pack(side='left')
        self.z1_mode = tk.StringVar(value='summary')
        ttk.Combobox(opts, textvariable=self.z1_mode, state='readonly', width=12,
                     values=['summary', 'raw']).pack(side='left', padx=(4, 12))
        ttk.Label(opts, text='μ₀:').pack(side='left')
        self.z1_mu0 = tk.StringVar(value='0')
        ttk.Entry(opts, width=10, textvariable=self.z1_mu0).pack(side='left', padx=(4,12))
        ttk.Label(opts, text='σ (known):').pack(side='left')
        self.z1_sigma = tk.StringVar(value='')
        ttk.Entry(opts, width=10, textvariable=self.z1_sigma).pack(side='left', padx=(4,12))
        ttk.Label(opts, text='Alternative:').pack(side='left')
        self.z1_alt = tk.StringVar(value='two-sided')
        ttk.Combobox(opts, textvariable=self.z1_alt, state='readonly', width=10,
                     values=self._alt_values()).pack(side='left', padx=(4, 12))
        ttk.Label(opts, text='CI level:').pack(side='left')
        self.z1_ci = tk.StringVar(value='0,95')
        ttk.Entry(opts, width=8, textvariable=self.z1_ci).pack(side='left', padx=(4,12))
        ttk.Label(opts, text='Decimals:').pack(side='left')
        self.dec_z1 = tk.IntVar(value=DEFAULT_DECIMALS)
        ttk.Spinbox(opts, from_=0, to=12, width=4, textvariable=self.dec_z1).pack(side='left', padx=(4,12))

        self.z1_sheet = SheetEditor(parent, rows=10, cols=1, lock_size=False, col_headers=['X'])
        self.z1_sheet.grid(row=1, column=0, sticky='ew')

        sumf = ttk.Labelframe(parent, text='Summary stats (use when Input="summary")')
        sumf.grid(row=2, column=0, sticky='ew', pady=(6,0))
        ttk.Label(sumf, text='n:').grid(row=0, column=0, padx=(8,2), pady=4, sticky='w')
        ttk.Label(sumf, text='mean:').grid(row=0, column=2, padx=(8,2), sticky='w')
        self.z1_n = tk.StringVar(); self.z1_mean = tk.StringVar()
        ttk.Entry(sumf, width=10, textvariable=self.z1_n).grid(row=0, column=1)
        ttk.Entry(sumf, width=10, textvariable=self.z1_mean).grid(row=0, column=3)

        actions = ttk.Frame(parent); actions.grid(row=3, column=0, sticky='ew', pady=(6,0))
        ttk.Button(actions, text='Compute', command=self._compute_z1).pack(side='left')

        self.z1_text = self._add_summary_panel(parent, row_upper=3, row_summary=4)

    def _compute_z1(self):
        self.formatter.decimals = int(self.dec_z1.get())
        mu0 = parse_number(self.z1_mu0.get())
        sigma = parse_number(self.z1_sigma.get())
        level = self._parse_level(self.z1_ci.get(), 0.95)
        alt = self.z1_alt.get().strip()
        try:
            mode = self.z1_mode.get()
            if sigma <= 0:
                raise ValueError("Known σ must be > 0.")
            if mode == 'raw':
                x = self.z1_sheet.read_column(0)
                n = len(x)
                mean = float(np.mean(x))
            else:
                n = int(parse_number(self.z1_n.get()))
                mean = parse_number(self.z1_mean.get())
                if n <= 0:
                    raise ValueError("Summary: n must be > 0.")
            se = sigma / np.sqrt(n)
            z = (mean - mu0) / se
            if alt == 'two-sided':
                p = 2 * (1 - stats.norm.cdf(abs(z)))
            elif alt == 'greater':
                p = 1 - stats.norm.cdf(z)
            else:
                p = stats.norm.cdf(z)
            ci_lo, ci_hi = z_ci_mean(mean, sigma, n, level)
            lines = []
            lines.append("One-sample z-test (mean, σ known)")
            lines.append(f" n = {self.formatter.fmt(n,0)}, mean = {self.formatter.fmt(mean)}, σ = {self.formatter.fmt(sigma)}")
            lines.append(f" H0: μ = {self.formatter.fmt(mu0)}; alternative = {alt}")
            lines.append(f" z = {self.formatter.fmt(z)}, p = {self.formatter.fmt(p)}")
            lines.append(f" {self.formatter.fmt(level*100,1)}% CI for μ: [{self.formatter.fmt(ci_lo)}, {self.formatter.fmt(ci_hi)}]")
            self.z1_text.delete('1.0','end')
            self.z1_text.insert('1.0','\n'.join(lines))
            self._set_status('One-sample z-test computed')
        except Exception as e:
            messagebox.showerror('z-test (one-sample mean)', str(e))

    # --------------------------- z-test: two-sample mean (σ known) ---------------------------
    def _build_z2_tab(self, parent):
        opts = ttk.Frame(parent); opts.grid(row=0, column=0, sticky='ew', pady=(0,6))
        ttk.Label(opts, text='Input:').pack(side='left')
        self.z2_mode = tk.StringVar(value='summary')
        ttk.Combobox(opts, textvariable=self.z2_mode, state='readonly', width=12,
                     values=['summary', 'raw']).pack(side='left', padx=(4,12))
        ttk.Label(opts, text='Alternative:').pack(side='left')
        self.z2_alt = tk.StringVar(value='two-sided')
        ttk.Combobox(opts, textvariable=self.z2_alt, state='readonly', width=10,
                     values=self._alt_values()).pack(side='left', padx=(4, 12))
        ttk.Label(opts, text='CI level:').pack(side='left')
        self.z2_ci = tk.StringVar(value='0,95')
        ttk.Entry(opts, width=8, textvariable=self.z2_ci).pack(side='left', padx=(4,12))
        ttk.Label(opts, text='σ1, σ2 (known):').pack(side='left')
        self.z2_s1 = tk.StringVar(); self.z2_s2 = tk.StringVar()
        ttk.Entry(opts, width=8, textvariable=self.z2_s1).pack(side='left', padx=(4,6))
        ttk.Entry(opts, width=8, textvariable=self.z2_s2).pack(side='left', padx=(4,6))
        ttk.Label(opts, text='Decimals:').pack(side='left')
        self.dec_z2 = tk.IntVar(value=DEFAULT_DECIMALS)
        ttk.Spinbox(opts, from_=0, to=12, width=4, textvariable=self.dec_z2).pack(side='left', padx=(4,12))

        self.z2_sheet = SheetEditor(parent, rows=10, cols=2, lock_size=False, col_headers=['Group A','Group B'])
        self.z2_sheet.grid(row=1, column=0, sticky='ew')

        sumf = ttk.Labelframe(parent, text='Summary stats (use when Input="summary")')
        sumf.grid(row=2, column=0, sticky='ew', pady=(6,0))
        ttk.Label(sumf, text='A: n').grid(row=0, column=0, padx=(8,2), pady=4, sticky='w')
        ttk.Label(sumf, text='mean').grid(row=0, column=2, padx=(8,2), sticky='w')
        self.z2_n1 = tk.StringVar(); self.z2_m1 = tk.StringVar()
        ttk.Entry(sumf, width=10, textvariable=self.z2_n1).grid(row=0, column=1)
        ttk.Entry(sumf, width=10, textvariable=self.z2_m1).grid(row=0, column=3)
        ttk.Label(sumf, text='B: n').grid(row=1, column=0, padx=(8,2), pady=4, sticky='w')
        ttk.Label(sumf, text='mean').grid(row=1, column=2, padx=(8,2), sticky='w')
        self.z2_n2 = tk.StringVar(); self.z2_m2 = tk.StringVar()
        ttk.Entry(sumf, width=10, textvariable=self.z2_n2).grid(row=1, column=1)
        ttk.Entry(sumf, width=10, textvariable=self.z2_m2).grid(row=1, column=3)

        actions = ttk.Frame(parent); actions.grid(row=3, column=0, sticky='ew', pady=(6,0))
        ttk.Button(actions, text='Compute', command=self._compute_z2).pack(side='left')

        self.z2_text = self._add_summary_panel(parent, row_upper=3, row_summary=4)

    def _compute_z2(self):
        self.formatter.decimals = int(self.dec_z2.get())
        level = self._parse_level(self.z2_ci.get(), 0.95)
        alt = self.z2_alt.get().strip()
        sigma1 = parse_number(self.z2_s1.get())
        sigma2 = parse_number(self.z2_s2.get())
        try:
            if sigma1 <= 0 or sigma2 <= 0:
                raise ValueError("Known σ1 and σ2 must be > 0.")
            mode = self.z2_mode.get()
            if mode == 'raw':
                a, b = self.z2_sheet.read_two_columns()
                n1, n2 = len(a), len(b)
                m1, m2 = float(np.mean(a)), float(np.mean(b))
            else:
                n1 = int(parse_number(self.z2_n1.get()))
                m1 = parse_number(self.z2_m1.get())
                n2 = int(parse_number(self.z2_n2.get()))
                m2 = parse_number(self.z2_m2.get())
            if min(n1, n2) <= 0:
                raise ValueError("Summary: n1 and n2 must be > 0.")
            se = np.sqrt((sigma1**2)/n1 + (sigma2**2)/n2)
            z = (m1 - m2) / se
            if alt == 'two-sided':
                p = 2*(1 - stats.norm.cdf(abs(z)))
            elif alt == 'greater':
                p = 1 - stats.norm.cdf(z)
            else:
                p = stats.norm.cdf(z)
            ci_lo, ci_hi = z_ci_diff_means(m1, sigma1, n1, m2, sigma2, n2, level)
            lines = []
            lines.append("Two-sample z-test (means, σ known)")
            lines.append(f" n1={self.formatter.fmt(n1,0)}, mean1={self.formatter.fmt(m1)}, σ1={self.formatter.fmt(sigma1)}")
            lines.append(f" n2={self.formatter.fmt(n2,0)}, mean2={self.formatter.fmt(m2)}, σ2={self.formatter.fmt(sigma2)}")
            lines.append(f" alternative = {alt}")
            lines.append(f" z = {self.formatter.fmt(z)}, p = {self.formatter.fmt(p)}")
            lines.append(f" {self.formatter.fmt(level*100,1)}% CI for (mean1 - mean2): [{self.formatter.fmt(ci_lo)}, {self.formatter.fmt(ci_hi)}]")
            self.z2_text.delete('1.0','end')
            self.z2_text.insert('1.0','\n'.join(lines))
            self._set_status('Two-sample z-test computed')
        except Exception as e:
            messagebox.showerror('z-test (two-sample mean)', str(e))

    # -------------------------------- z-test: one proportion --------------------------------
    def _build_p1_tab(self, parent):
        opts = ttk.Frame(parent); opts.grid(row=0, column=0, sticky='ew', pady=(0,6))
        ttk.Label(opts, text='Successes x:').pack(side='left')
        self.p1_x = tk.StringVar(value='')
        ttk.Entry(opts, width=10, textvariable=self.p1_x).pack(side='left', padx=(4,12))
        ttk.Label(opts, text='Trials n:').pack(side='left')
        self.p1_n = tk.StringVar(value='')
        ttk.Entry(opts, width=10, textvariable=self.p1_n).pack(side='left', padx=(4,12))
        ttk.Label(opts, text='p₀:').pack(side='left')
        self.p1_p0 = tk.StringVar(value='0,5')
        ttk.Entry(opts, width=10, textvariable=self.p1_p0).pack(side='left', padx=(4,12))
        ttk.Label(opts, text='Alternative:').pack(side='left')
        self.p1_alt = tk.StringVar(value='two-sided')
        ttk.Combobox(opts, textvariable=self.p1_alt, state='readonly', width=10,
                     values=self._alt_values()).pack(side='left', padx=(4, 12))
        ttk.Label(opts, text='CI level:').pack(side='left')
        self.p1_ci = tk.StringVar(value='0,95')
        ttk.Entry(opts, width=8, textvariable=self.p1_ci).pack(side='left', padx=(4,12))
        ttk.Label(opts, text='Decimals:').pack(side='left')
        self.dec_p1 = tk.IntVar(value=DEFAULT_DECIMALS)
        ttk.Spinbox(opts, from_=0, to=12, width=4, textvariable=self.dec_p1).pack(side='left', padx=(4,12))

        upper = ttk.Frame(parent); upper.grid(row=3, column=0, sticky='ew')
        ttk.Button(upper, text='Compute', command=self._compute_p1).pack(side='left')

        self.p1_text = self._add_summary_panel(parent, row_upper=3, row_summary=4)

    def _compute_p1(self):
        self.formatter.decimals = int(self.dec_p1.get())
        try:
            x = int(parse_number(self.p1_x.get()))
            n = int(parse_number(self.p1_n.get()))
            p0 = float(parse_number(self.p1_p0.get()))
            if n <= 0 or not (0 <= x <= n) or not (0 < p0 < 1):
                raise ValueError("Invalid inputs: ensure 0 < n, 0 <= x <= n, and 0 < p₀ < 1.")
            alt = self.p1_alt.get().strip()
            level = self._parse_level(self.p1_ci.get(), 0.95)
            phat = x / n
            se0 = np.sqrt(p0 * (1 - p0) / n)
            z = (phat - p0) / se0
            if alt == 'two-sided':
                p = 2 * (1 - stats.norm.cdf(abs(z)))
            elif alt == 'greater':
                p = 1 - stats.norm.cdf(z)
            else:
                p = stats.norm.cdf(z)
            # Wald CI (for display; for small n consider Wilson/Clopper-Pearson)
            ci_lo, ci_hi = wald_ci_prop(phat, n, level)
            lines = []
            lines.append("One-proportion z-test")
            lines.append(f" x = {self.formatter.fmt(x,0)}, n = {self.formatter.fmt(n,0)}, p̂ = {self.formatter.fmt(phat)}")
            lines.append(f" H0: p = {self.formatter.fmt(p0)}; alternative = {alt}")
            lines.append(f" z = {self.formatter.fmt(z)}, p = {self.formatter.fmt(p)}")
            lines.append(f" {self.formatter.fmt(level*100,1)}% CI for p (Wald): [{self.formatter.fmt(ci_lo)}, {self.formatter.fmt(ci_hi)}]")
            self.p1_text.delete('1.0','end')
            self.p1_text.insert('1.0','\n'.join(lines))
            self._set_status('One-proportion z-test computed')
        except Exception as e:
            messagebox.showerror('z-test (one proportion)', str(e))

    # ------------------------------- z-test: two proportions -------------------------------
    def _build_p2_tab(self, parent):
        opts = ttk.Frame(parent); opts.grid(row=0, column=0, sticky='ew', pady=(0,6))
        ttk.Label(opts, text='A: x1').pack(side='left')
        self.p2_x1 = tk.StringVar(); ttk.Entry(opts, width=8, textvariable=self.p2_x1).pack(side='left', padx=(4,6))
        ttk.Label(opts, text='n1').pack(side='left'); self.p2_n1 = tk.StringVar()
        ttk.Entry(opts, width=8, textvariable=self.p2_n1).pack(side='left', padx=(4,12))
        ttk.Label(opts, text='B: x2').pack(side='left')
        self.p2_x2 = tk.StringVar(); ttk.Entry(opts, width=8, textvariable=self.p2_x2).pack(side='left', padx=(4,6))
        ttk.Label(opts, text='n2').pack(side='left'); self.p2_n2 = tk.StringVar()
        ttk.Entry(opts, width=8, textvariable=self.p2_n2).pack(side='left', padx=(4,12))
        ttk.Label(opts, text='Alternative:').pack(side='left')
        self.p2_alt = tk.StringVar(value='two-sided')
        ttk.Combobox(opts, textvariable=self.p2_alt, state='readonly', width=10,
                     values=self._alt_values()).pack(side='left', padx=(4, 12))
        ttk.Label(opts, text='CI level:').pack(side='left')
        self.p2_ci = tk.StringVar(value='0,95')
        ttk.Entry(opts, width=8, textvariable=self.p2_ci).pack(side='left', padx=(4,12))
        ttk.Label(opts, text='Decimals:').pack(side='left')
        self.dec_p2 = tk.IntVar(value=DEFAULT_DECIMALS)
        ttk.Spinbox(opts, from_=0, to=12, width=4, textvariable=self.dec_p2).pack(side='left', padx=(4,12))

        upper = ttk.Frame(parent); upper.grid(row=3, column=0, sticky='ew')
        ttk.Button(upper, text='Compute', command=self._compute_p2).pack(side='left')

        self.p2_text = self._add_summary_panel(parent, row_upper=3, row_summary=4)

    def _compute_p2(self):
        self.formatter.decimals = int(self.dec_p2.get())
        try:
            x1 = int(parse_number(self.p2_x1.get()))
            n1 = int(parse_number(self.p2_n1.get()))
            x2 = int(parse_number(self.p2_x2.get()))
            n2 = int(parse_number(self.p2_n2.get()))
            if min(n1, n2) <= 0 or not (0 <= x1 <= n1) or not (0 <= x2 <= n2):
                raise ValueError("Invalid inputs: ensure n1,n2>0 and 0<=x1<=n1, 0<=x2<=n2.")
            alt = self.p2_alt.get().strip()
            level = self._parse_level(self.p2_ci.get(), 0.95)
            p1, p2 = x1/n1, x2/n2
            # Pooled se under H0: p1=p2
            p_pool = (x1 + x2) / (n1 + n2)
            se_pool = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
            z = (p1 - p2) / se_pool if se_pool > 0 else np.nan
            if alt == 'two-sided':
                p = 2*(1 - stats.norm.cdf(abs(z)))
            elif alt == 'greater':
                p = 1 - stats.norm.cdf(z)
            else:
                p = stats.norm.cdf(z)
            # CI (unpooled/Wald)
            ci_lo, ci_hi = wald_ci_diff_props(p1, n1, p2, n2, level)
            lines = []
            lines.append("Two-proportion z-test")
            lines.append(f" A: x1={self.formatter.fmt(x1,0)}, n1={self.formatter.fmt(n1,0)}, p̂1={self.formatter.fmt(p1)}")
            lines.append(f" B: x2={self.formatter.fmt(x2,0)}, n2={self.formatter.fmt(n2,0)}, p̂2={self.formatter.fmt(p2)}")
            lines.append(f" alternative = {alt}")
            lines.append(f" z = {self.formatter.fmt(z)}, p = {self.formatter.fmt(p)}")
            lines.append(f" {self.formatter.fmt(level*100,1)}% CI for (p1 - p2) (Wald): [{self.formatter.fmt(ci_lo)}, {self.formatter.fmt(ci_hi)}]")
            self.p2_text.delete('1.0','end')
            self.p2_text.insert('1.0','\n'.join(lines))
            self._set_status('Two-proportion z-test computed')
        except Exception as e:
            messagebox.showerror('z-test (two proportions)', str(e))

# ------------------------------- Standalone demo -------------------------------
if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()  # show only the tool window
    ParametricTestsWindow(master=root)
    root.mainloop()
