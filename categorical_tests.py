
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont
import numpy as np

DEFAULT_DECIMALS = 6


# -------------------- Formatting helpers (decimal comma) --------------------
def parse_number(s: str) -> float:
    """Parse a number string robustly, accepting either comma or dot decimals."""
    if s is None:
        return 0.0
    s = str(s).strip()
    if s == '' or s.lower() in {'nan', 'none'}:
        return 0.0
    # Remove non‑breaking space and regular spaces (as thousands separators)
    s = s.replace('\u00A0', ' ').replace(' ', '')
    has_comma = ',' in s
    has_dot = '.' in s
    if has_comma and has_dot:
        # Decide which is decimal by last occurrence rule
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


# -------------------- Excel‑like grid editor --------------------
class SheetEditor(ttk.Frame):
    """Lightweight grid editor for small numeric tables (e.g., contingency tables)."""

    def __init__(self, parent, rows=3, cols=3, lock_size=False):
        super().__init__(parent)
        self.lock_size = bool(lock_size)
        self._rows = int(rows)
        self._cols = int(cols)
        self._cells = []  # 2D list of Entry
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
            bar, from_=1, to=100, width=4, textvariable=self.rows_var,
            command=self._apply_size, state=('disabled' if self.lock_size else 'normal')
        )
        self.spin_r.pack(side='left', padx=(4, 10))

        ttk.Label(bar, text='Cols:').pack(side='left')
        self.spin_c = ttk.Spinbox(
            bar, from_=1, to=100, width=4, textvariable=self.cols_var,
            command=self._apply_size, state=('disabled' if self.lock_size else 'normal')
        )
        self.spin_c.pack(side='left', padx=(4, 10))

        # Primary actions only (tidy toolbar)
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
        self._make_headers()
        self._build_cells()

        # Selection state
        self._sel = (0, 0)  # r, c
        self._select_cell(0, 0)

        # Keyboard shortcuts scoped to this frame
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

    # ---- build helpers ----
    def _make_headers(self):
        # Clear any old headers
        for child in list(self._grid.children.values()):
            child.destroy()

        hdr_tl = ttk.Label(self._grid, text='', background=self._hdr_bg)
        hdr_tl.grid(row=0, column=0, sticky='nsew')
        for c in range(self._cols):
            l = ttk.Label(self._grid, text=self._col_name(c), background=self._hdr_bg, anchor='center')
            l.grid(row=0, column=c + 1, sticky='nsew', padx=1, pady=1)
        for r in range(self._rows):
            l = ttk.Label(self._grid, text=str(r + 1), background=self._hdr_bg, anchor='e')
            l.grid(row=r + 1, column=0, sticky='nsew', padx=1, pady=1)

    def _build_cells(self):
        # Remove old cell widgets only (keep headers)
        for child in list(self._grid.children.values()):
            info = child.grid_info()
            if info and int(info.get('row', 0)) > 0 and int(info.get('column', 0)) > 0:
                child.destroy()
        self._cells = []
        for r in range(self._rows):
            row = []
            for c in range(self._cols):
                e = ttk.Entry(self._grid, width=10, justify='center', font=self._mono)
                e.grid(row=r + 1, column=c + 1, padx=1, pady=1, sticky='nsew')
                e.insert(0, '0')
                e.bind('<Button-1>', lambda ev, rr=r, cc=c: self._select_cell(rr, cc))
                e.bind('<Shift-Button-1>', lambda ev, rr=r, cc=c: self._select_cell(rr, cc))
                e.bind('<Return>', lambda ev, rr=r, cc=c: self._move_focus(rr + 1, cc))
                e.bind('<KP_Enter>', lambda ev, rr=r, cc=c: self._move_focus(rr + 1, cc))
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
            # Rebuild headers and cells
            self._make_headers()
            self._build_cells()
            # Keep selection sane
            self._select_cell(0, 0)

    # ---- selection and movement ----
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

    # ---- data API ----
    def set_array(self, arr: np.ndarray):
        r, c = arr.shape
        if not self.lock_size:
            self.rows_var.set(int(r))
            self.cols_var.set(int(c))
            self._apply_size()
        r = min(self._rows, int(r))
        c = min(self._cols, int(c))
        for i in range(r):
            for j in range(c):
                e = self._cells[i][j]
                e.delete(0, 'end')
                e.insert(0, str(arr[i, j]))

    def read_array(self) -> np.ndarray:
        r = self._rows
        c = self._cols
        arr = np.zeros((r, c), dtype=float)
        for i in range(r):
            for j in range(c):
                s = self._cells[i][j].get().strip()
                if s == '':
                    s = '0'
                try:
                    v = parse_number(s)
                except Exception:
                    raise ValueError(f'Invalid number at row {i+1}, col {j+1}: {s}')
                if v < 0:
                    raise ValueError(f'Negative count at row {i+1}, col {j+1}')
                arr[i, j] = v
        if np.all(arr == 0):
            raise ValueError('All counts are zero.')
        return arr

    def clear(self):
        for i in range(self._rows):
            for j in range(self._cols):
                e = self._cells[i][j]
                e.delete(0, 'end')
                e.insert(0, '0')

    # ---- clipboard and files ----
    def paste_block(self):
        try:
            raw = self.clipboard_get()
        except Exception:
            messagebox.showwarning('Paste', 'Nothing on clipboard.')
            return
        lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
        data = []
        for ln in lines:
            if '\t' in ln:
                parts = ln.split('\t')
            elif ',' in ln:
                parts = ln.split(',')
            else:
                parts = ln.split()
            row = []
            for p in parts:
                p = p.strip()
                if p == '':
                    continue
                try:
                    row.append(parse_number(p))
                except Exception:
                    row.append(np.nan)
            if row:
                data.append(row)
        if not data:
            messagebox.showwarning('Paste', 'Clipboard does not look like a table of numbers.')
            return
        r0, c0 = self._sel
        # Fill into grid starting at current cell
        for i, row in enumerate(data):
            if r0 + i >= self._rows:
                break
            for j, val in enumerate(row):
                if c0 + j >= self._cols:
                    break
                e = self._cells[r0 + i][c0 + j]
                e.delete(0, 'end')
                e.insert(0, str(val))

    def copy_block(self):
        # Copy whole grid for simplicity
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
            arr = np.zeros(values.shape, dtype=float)
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    arr[i, j] = parse_number(values[i, j])
            self.set_array(arr)
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
            arr = np.zeros(values.shape, dtype=float)
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    arr[i, j] = parse_number(values[i, j])
            self.set_array(arr)
        except Exception as e:
            messagebox.showerror('Excel', f'Could not load Excel: {e}')

    # ---- helpers ----
    def _col_name(self, idx):
        # Convert 0->A, 25->Z, 26->AA, etc.
        name = ''
        n = idx
        while True:
            n, r = divmod(n, 26)
            name = chr(ord('A') + r) + name
            if n == 0:
                break
            n -= 1
        return name


# -------------------- Main window --------------------
class CategoricalTestsWindow(tk.Toplevel):
    def __init__(self, master=None, title='Categorical tests'):
        super().__init__(master=master)
        self.title(title)
        self.geometry('1200x780')
        self.minsize(1024, 640)
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

        self.tab_chi = ttk.Frame(self.nb)
        self.tab_fisher = ttk.Frame(self.nb)
        self.tab_or = ttk.Frame(self.nb)
        self.tab_mcnemar = ttk.Frame(self.nb)

        self.nb.add(self.tab_chi, text='Chi-square (R x C)')
        self.nb.add(self.tab_fisher, text="Fisher's exact (2x2)")
        self.nb.add(self.tab_or, text='Odds ratio (2x2)')
        self.nb.add(self.tab_mcnemar, text='McNemar (paired 2x2)')

        self._build_chi_tab(self.tab_chi)
        self._build_fisher_tab(self.tab_fisher)
        self._build_or_tab(self.tab_or)
        self._build_mcnemar_tab(self.tab_mcnemar)

        self.status_var = tk.StringVar(value='Ready')
        ttk.Label(self, textvariable=self.status_var, anchor='w').pack(side='bottom', fill='x')

        # Center the window when first shown
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

    # ---------- Chi-square tab ----------
    def _build_chi_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        # Make results (row 3) lighter and summary (row 4) heavier & taller
        parent.rowconfigure(3, weight=1, minsize=120)   # Observed/Expected/Residuals
        parent.rowconfigure(4, weight=5, minsize=220)   # Summary larger + grows more

        opts = ttk.Frame(parent)
        opts.grid(row=0, column=0, sticky='ew', pady=(0, 6))

        self.chi_yates = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Yates' correction (apply if 2x2)",
                        variable=self.chi_yates).pack(side='left', padx=(0, 12))

        ttk.Label(opts, text='Decimals:').pack(side='left')
        self.decimals_var = tk.IntVar(value=self.formatter.decimals)
        ttk.Spinbox(opts, from_=0, to=12, width=4, textvariable=self.decimals_var,
                    command=self._update_decimals).pack(side='left', padx=(4, 12))

        self.strip_totals_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text='Strip totals (last row/col if sums match)',
                        variable=self.strip_totals_var).pack(side='left')

        # Editor: keep resizable here; set lock_size=True if you prefer fixed size
        self.chi_sheet = SheetEditor(parent, rows=3, cols=3, lock_size=False)
        self.chi_sheet.grid(row=1, column=0, sticky='ew')

        actions = ttk.Frame(parent)
        actions.grid(row=2, column=0, sticky='ew', pady=(6, 0))
        ttk.Button(actions, text='Compute', command=self._compute_chi).pack(side='left')

        # Results area
        paned = ttk.Panedwindow(parent, orient='horizontal')
        paned.grid(row=3, column=0, sticky='nsew', pady=(8, 0))

        left = ttk.Labelframe(paned, text='Observed')
        mid = ttk.Labelframe(paned, text='Expected')
        right = ttk.Labelframe(paned, text='Residuals ((O-E)/sqrt(E))')
        paned.add(left, weight=1)
        paned.add(mid, weight=1)
        paned.add(right, weight=1)

        self.chi_obs = tk.Text(left, wrap='none', font=self._mono)
        self.chi_exp = tk.Text(mid, wrap='none', font=self._mono)
        self.chi_res = tk.Text(right, wrap='none', font=self._mono)
        self.chi_obs.pack(fill='both', expand=True)
        self.chi_exp.pack(fill='both', expand=True)
        self.chi_res.pack(fill='both', expand=True)

        # Summary (bigger AND scrollable)
        summ = ttk.Labelframe(parent, text='Summary')
        summ.grid(row=4, column=0, sticky='nsew', pady=(8, 0))

        # a small container to host the Text + vertical scrollbar
        summ_container = ttk.Frame(summ)
        summ_container.pack(fill='both', expand=True)

        self.chi_sum = tk.Text(summ_container, wrap='word', font=self._mono, height=10)
        vscroll = ttk.Scrollbar(summ_container, orient='vertical', command=self.chi_sum.yview)
        self.chi_sum.configure(yscrollcommand=vscroll.set)

        # Layout: text expands; scrollbar pinned to the right
        self.chi_sum.pack(side='left', fill='both', expand=True)
        vscroll.pack(side='right', fill='y')

    def _update_decimals(self):
        try:
            self.formatter.decimals = max(0, min(12, int(self.decimals_var.get())))
        except Exception:
            self.formatter.decimals = DEFAULT_DECIMALS
        self._set_status(f'Decimals set to {self.formatter.decimals}')

    def _maybe_strip_totals(self, arr: np.ndarray) -> np.ndarray:
        a = np.array(arr, dtype=float)
        changed = False
        if a.shape[0] > 1:
            row_sums = np.sum(a[:-1, :], axis=0)
            if np.allclose(row_sums, a[-1, :], rtol=0, atol=1e-6):
                a = a[:-1, :]
                changed = True
        if a.shape[1] > 1:
            col_sums = np.sum(a[:, :-1], axis=1)
            if np.allclose(col_sums, a[:, -1], rtol=0, atol=1e-6):
                a = a[:, :-1]
                changed = True
        if changed:
            self._set_status('Totals detected and stripped from data')
        return a

    def _compute_chi(self):
        for t in (self.chi_obs, self.chi_exp, self.chi_res, self.chi_sum):
            t.delete('1.0', 'end')
        try:
            table = self.chi_sheet.read_array()
            if self.strip_totals_var.get():
                table = self._maybe_strip_totals(table)
        except Exception as e:
            messagebox.showerror('Input', str(e))
            return

        R, C = table.shape
        try:
            from scipy import stats
            correction = bool(self.chi_yates.get() and (R == 2 and C == 2))
            chi2, p, dof, expected = stats.chi2_contingency(table, correction=correction)
            with np.errstate(divide='ignore', invalid='ignore'):
                resid = (table - expected) / np.sqrt(expected)
            N = np.sum(table)
            phi = np.sqrt(chi2 / N) if R == 2 and C == 2 and N > 0 else np.nan
            cramer_v = np.sqrt(chi2 / (N * max(1, min(R - 1, C - 1)))) if N > 0 else np.nan
        except Exception as e:
            messagebox.showerror('Chi-square', f'Failed to compute chi-square: {e}')
            return

        self.chi_obs.insert('1.0', self.formatter.array_to_str(table))
        self.chi_exp.insert('1.0', self.formatter.array_to_str(expected))
        self.chi_res.insert('1.0', self.formatter.array_to_str(resid))

        lines = []
        lines.append(f"N = {self.formatter.fmt(N, 0)}")
        lines.append('')
        lines.append('Chi-square test of independence:')
        lines.append(f" chi2 = {self.formatter.fmt(chi2)}; df = {int(dof)}; p = {self.formatter.fmt(p)}")
        lines.append(f" Yates' correction: {'ON' if correction else 'OFF'}")
        if R == 2 and C == 2:
            lines.append(f" Phi coefficient = {self.formatter.fmt(phi)}")
            lines.append(f" Cramer's V = {self.formatter.fmt(cramer_v)}")
        self.chi_sum.insert('1.0', '\n'.join(lines))
        self._set_status('Chi-square computed')

    # ---------- Fisher tab ----------
    def _build_fisher_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)

        opts = ttk.Frame(parent)
        opts.grid(row=0, column=0, sticky='ew', pady=(0, 6))
        ttk.Label(opts, text='Alternative:').pack(side='left')
        self.fisher_alt = tk.StringVar(value='two-sided')
        ttk.Combobox(opts, textvariable=self.fisher_alt, state='readonly', width=11,
                     values=['two-sided', 'less', 'greater']).pack(side='left', padx=(4, 12))

        self.fisher_sheet = SheetEditor(parent, rows=2, cols=2, lock_size=True)
        self.fisher_sheet.grid(row=1, column=0, sticky='ew')

        actions = ttk.Frame(parent)
        actions.grid(row=2, column=0, sticky='ew', pady=(6, 0))
        ttk.Button(actions, text='Compute', command=self._compute_fisher).pack(side='left')

        res = ttk.Labelframe(parent, text='Results')
        res.grid(row=3, column=0, sticky='nsew', pady=(8, 0))
        self.fisher_text = tk.Text(res, wrap='word', font=self._mono)
        self.fisher_text.pack(fill='both', expand=True)

    def _compute_fisher(self):
        self.fisher_text.delete('1.0', 'end')
        try:
            table = self.fisher_sheet.read_array()
        except Exception as e:
            messagebox.showerror('Input', str(e))
            return
        if table.shape != (2, 2):
            messagebox.showerror('Fisher', 'Table must be 2x2 for Fisher\'s exact test.')
            return
        try:
            from scipy import stats
            or_est, p = stats.fisher_exact(table, alternative=self.fisher_alt.get().strip())
        except Exception as e:
            messagebox.showerror('Fisher', f'Failed to compute Fisher\'s exact test: {e}')
            return

        lines = []
        lines.append('Observed 2x2 table:')
        lines.append(self.formatter.array_to_str(table))
        lines.append('')
        lines.append("Fisher's exact test (2x2):")
        lines.append(f" alternative = {self.fisher_alt.get().strip()}")
        lines.append(f" OR (Fisher) = {self.formatter.fmt(or_est)}")
        lines.append(f" p = {self.formatter.fmt(p)}")
        self.fisher_text.insert('1.0', '\n'.join(lines))
        self._set_status('Fisher computed')

    # ---------- Odds ratio tab ----------
    def _build_or_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)

        opts = ttk.Frame(parent)
        opts.grid(row=0, column=0, sticky='ew', pady=(0, 6))
        ttk.Label(opts, text='CI level:').pack(side='left')
        self.or_ci = tk.StringVar(value='0,95')
        ttk.Entry(opts, textvariable=self.or_ci, width=8).pack(side='left', padx=(4, 12))
        self.or_haldane = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text='Haldane-Anscombe 0.5 correction (if any zero cell)',
                        variable=self.or_haldane).pack(side='left')

        self.or_sheet = SheetEditor(parent, rows=2, cols=2, lock_size=True)
        self.or_sheet.grid(row=1, column=0, sticky='ew')

        actions = ttk.Frame(parent)
        actions.grid(row=2, column=0, sticky='ew', pady=(6, 0))
        ttk.Button(actions, text='Compute', command=self._compute_or).pack(side='left')

        res = ttk.Labelframe(parent, text='Results')
        res.grid(row=3, column=0, sticky='nsew', pady=(8, 0))
        self.or_text = tk.Text(res, wrap='word', font=self._mono)
        self.or_text.pack(fill='both', expand=True)

    def _compute_or(self):
        self.or_text.delete('1.0', 'end')
        try:
            table = self.or_sheet.read_array()
        except Exception as e:
            messagebox.showerror('Input', str(e))
            return
        if table.shape != (2, 2):
            messagebox.showerror('Odds ratio', 'Table must be 2x2 for odds ratio.')
            return
        try:
            a, b, c, d = [float(table[0, 0]), float(table[0, 1]),
                          float(table[1, 0]), float(table[1, 1])]
            haldane = bool(self.or_haldane.get())
            if haldane and (a == 0 or b == 0 or c == 0 or d == 0):
                a += 0.5; b += 0.5; c += 0.5; d += 0.5
            if min(a, b, c, d) == 0:
                raise ValueError('Odds ratio undefined when any cell is zero (and correction is off).')
            or_est = (a * d) / (b * c)
            se = np.sqrt(1/a + 1/b + 1/c + 1/d)
            try:
                level = float(parse_number(self.or_ci.get()))
                if not (0.0 < level < 1.0):
                    level = 0.95
            except Exception:
                level = 0.95
            from scipy.stats import norm
            z = norm.ppf(1 - (1 - level) / 2.0)
            lo = np.exp(np.log(or_est) - z * se)
            hi = np.exp(np.log(or_est) + z * se)
        except Exception as e:
            messagebox.showerror('Odds ratio', f'Failed to compute odds ratio: {e}')
            return

        lines = []
        lines.append('Observed 2x2 table:')
        lines.append(self.formatter.array_to_str(table))
        lines.append('')
        lines.append('Odds ratio (2x2):')
        lines.append(f" OR = {self.formatter.fmt(or_est)}")
        lines.append(f" {self.formatter.fmt(level * 100, 1)}% CI = "
                     f"[{self.formatter.fmt(lo)}, {self.formatter.fmt(hi)}] "
                     f"(Woolf log-OR, {'Haldane 0.5' if haldane else 'no'} correction)")
        self.or_text.insert('1.0', '\n'.join(lines))
        self._set_status('OR computed')

    # ---------- McNemar tab ----------
    def _build_mcnemar_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)

        opts = ttk.Frame(parent)
        opts.grid(row=0, column=0, sticky='ew', pady=(0, 6))
        ttk.Label(opts, text='Report:').pack(side='left')
        self.mc_report = tk.StringVar(value='both')
        ttk.Combobox(opts, textvariable=self.mc_report, state='readonly', width=12,
                     values=['both', 'exact', 'chi2']).pack(side='left', padx=(4, 12))
        self.mc_yates = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text='Continuity correction (chi2)',
                        variable=self.mc_yates).pack(side='left')

        self.mc_sheet = SheetEditor(parent, rows=2, cols=2, lock_size=True)
        self.mc_sheet.grid(row=1, column=0, sticky='ew')

        actions = ttk.Frame(parent)
        actions.grid(row=2, column=0, sticky='ew', pady=(6, 0))
        ttk.Button(actions, text='Compute', command=self._compute_mcnemar).pack(side='left')

        res = ttk.Labelframe(parent, text='Results')
        res.grid(row=3, column=0, sticky='nsew', pady=(8, 0))
        self.mc_text = tk.Text(res, wrap='word', font=self._mono)
        self.mc_text.pack(fill='both', expand=True)

    def _compute_mcnemar(self):
        self.mc_text.delete('1.0', 'end')
        try:
            tab = self.mc_sheet.read_array()
        except Exception as e:
            messagebox.showerror('Input', str(e))
            return
        if tab.shape != (2, 2):
            messagebox.showerror('McNemar', 'Table must be 2x2.')
            return

        b = float(tab[0, 1])
        c = float(tab[1, 0])
        n = b + c

        lines = []
        lines.append('Paired 2x2 table (Before/After):')
        lines.append(self.formatter.array_to_str(tab))
        lines.append('')

        if n == 0:
            lines.append('No discordant pairs (b + c = 0). McNemar test not applicable.')
            self.mc_text.insert('1.0', '\n'.join(lines))
            self._set_status('McNemar computed (no discordant pairs)')
            return

        report = self.mc_report.get().strip()
        p_exact = None
        p_chi = None
        try:
            # Exact requires integer discordant counts
            b_int = abs(b - round(b)) < 1e-9
            c_int = abs(c - round(c)) < 1e-9
            if report in ('both', 'exact') and b_int and c_int:
                from scipy.stats import binomtest
                p_exact = binomtest(int(min(round(b), round(c))),
                                    int(round(n)),
                                    0.5, alternative='two-sided').pvalue
            if report in ('both', 'chi2'):
                from scipy.stats import chi2
                if self.mc_yates.get():
                    stat = (abs(b - c) - 1) ** 2 / n
                else:
                    stat = (b - c) ** 2 / n
                p_chi = 1 - chi2.cdf(stat, df=1)
        except Exception as e:
            messagebox.showerror('McNemar', f'Failed to compute: {e}')
            return

        if p_exact is not None:
            lines.append('McNemar exact (binomial):')
            lines.append(f" b = {self.formatter.fmt(b, 0)}, c = {self.formatter.fmt(c, 0)}, n = {self.formatter.fmt(n, 0)}")
            lines.append(f" p = {self.formatter.fmt(p_exact)}")
            lines.append('')
        elif report in ('both', 'exact'):
            lines.append('McNemar exact (binomial): not shown (discordant counts are not integers).')
            lines.append('')

        if p_chi is not None:
            lines.append('McNemar chi-square (df=1):')
            lines.append(f" continuity correction: {'ON' if self.mc_yates.get() else 'OFF'}")
            if self.mc_yates.get():
                stat = (abs(b - c) - 1) ** 2 / n
            else:
                stat = (b - c) ** 2 / n
            lines.append(f" chi2 = {self.formatter.fmt(stat)}; p = {self.formatter.fmt(p_chi)}")

        self.mc_text.insert('1.0', '\n'.join(lines))
        self._set_status('McNemar computed')


# -------------------- Standalone demo --------------------
if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()  # Keep only the tool window visible
    CategoricalTestsWindow(master=root)
    root.mainloop()
