
# scientific_calculator.py
# RStudio-like console layout for a scientific calculator (Tkinter + NumPy)
# - Left: Console area split by a vertical Panedwindow (Transcript top, Input bottom; draggable sash)
# - Bottom (pane): Console input (multi-line; Enter/Ctrl+Enter = run, Shift+Enter = newline)
# - Right: Tabs: History (expr + result), Environment (variables, ans), Formulas (named snippets)
# - Safe evaluation in a restricted namespace (NumPy-based)
# - Variables, ans, degree/radian toggle, implicit multiplication support
# - Persistent history + formulas; robust window close on Windows

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import re
from datetime import datetime
from typing import Optional

import numpy as np

HISTORY_FILE = "calc_history.json"
FORMULAS_FILE = "calc_formulas.json"
MAX_HISTORY = 500  # ring-buffer cap


class ScientificCalculator(tk.Toplevel):
    def __init__(self, master: Optional[tk.Misc] = None, title: str = "Scientific Calculator"):
        # Create as Toplevel if master is given, else as main Tk window with hidden root
        if master is None:
            self._is_root = True
            root = tk.Tk()
            super().__init__(root)
            root.withdraw()
        else:
            self._is_root = False
            super().__init__(master)

        # guard against re-entrant close on Windows
        self._closing = False
        self.title(title)
        try:
            self.state("zoomed")
        except Exception:
            self.geometry("1100x800+60+40")  # Tall default window

        # ---------- state ----------
        self.vars: dict[str, float | np.ndarray] = {}
        self.ans = 0.0
        self.use_degrees = tk.BooleanVar(value=False)

        self._history: list[dict] = self._load_history()
        self._cmd_history: list[str] = [h.get("expr", "") for h in self._history if h.get("expr")]
        self._hist_index: int = len(self._cmd_history)

        # Formulas store (list of {"name": str, "expr": str})
        self._formulas: list[dict] = self._load_formulas()

        # ---------- styles ----------
        self.style = ttk.Style(self)
        self.style.configure("Console.TFrame", background="#1E1E1E")
        self.style.configure("Console.TLabel", foreground="#D4D4D4", background="#1E1E1E")
        self.style.configure("Cmd.TText", font=("Consolas", 14))
        self.style.configure("Transcript.TText", font=("Consolas", 12))

        # ---------- layout ----------
        outer = ttk.Frame(self, padding=(8, 8))
        outer.pack(fill="both", expand=True)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=0)

        # LEFT: Console column with resizable transcript/input via Panedwindow
        console_frame = ttk.Frame(outer, style="Console.TFrame")
        console_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)

        # Panedwindow: vertical split (top transcript, bottom input container)
        console_pane = ttk.Panedwindow(console_frame, orient="vertical")
        console_pane.grid(row=0, column=0, sticky="nsew")

        # --- Top pane: Transcript (RESULT) ---
        transcript_frame = ttk.Frame(console_pane, style="Console.TFrame")
        console_pane.add(transcript_frame, weight=3)  # top result area gets higher weight
        self.transcript = tk.Text(
            transcript_frame,
            wrap="word",
            height=20,  # initial height; pane is resizable anyway
            bg="#1E1E1E",
            fg="#D4D4D4",
            insertbackground="#D4D4D4",
            bd=0,
            relief="flat",
            font=("Consolas", 12),
        )
        t_y = ttk.Scrollbar(transcript_frame, orient="vertical", command=self.transcript.yview)
        self.transcript.configure(yscrollcommand=t_y.set)
        self.transcript.pack(side="left", fill="both", expand=True)
        t_y.pack(side="right", fill="y")
        self.transcript.config(state="disabled")
        self._init_transcript_tags()

        # --- Bottom pane: Input area (toolbar + label + input box) ---
        input_area = ttk.Frame(console_pane)
        console_pane.add(input_area, weight=2)  # bottom input area (~40%)
        # Top row: small toolbar
        toolbar = ttk.Frame(input_area)
        toolbar.pack(fill="x", pady=(0, 4))
        ttk.Button(toolbar, text="Run (Enter)", command=self.evaluate).pack(side="left")
        ttk.Button(toolbar, text="Clear Input", command=lambda: self._set_input_text("")).pack(side="left", padx=6)
        ttk.Button(toolbar, text="Clear Console", command=self._clear_transcript).pack(side="left", padx=6)
        ttk.Checkbutton(toolbar, text="Degrees", variable=self.use_degrees).pack(side="left", padx=12)
        ttk.Button(toolbar, text="Save Transcript…", command=self._save_transcript).pack(side="right")
        ttk.Button(toolbar, text="Help", command=self._show_calculations_help).pack(side="right", padx=(0, 8))

        # Label for input
        ttk.Label(input_area, text="Console >", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(2, 2))

        # Input Text (fills remaining height of bottom pane)
        input_frame = ttk.Frame(input_area)
        input_frame.pack(fill="both", expand=True)
        self.input = tk.Text(
            input_frame,
            height=10,  # initial height; pane allows resizing
            wrap="word",
            font=("Consolas", 14),
            undo=True,
            bg="#111318",
            fg="#E6E6E6",
            insertbackground="#E6E6E6",
            bd=1,
            relief="solid",
        )
        i_y = ttk.Scrollbar(input_frame, orient="vertical", command=self.input.yview)
        self.input.configure(yscrollcommand=i_y.set)
        self.input.pack(side="left", fill="both", expand=True)
        i_y.pack(side="right", fill="y")

        # Configure min sizes so panes can’t fully collapse
        try:
            console_pane.paneconfigure(transcript_frame, minsize=80)
            console_pane.paneconfigure(input_area, minsize=80)
        except Exception:
            pass

        # --- NEW: Robust initial sash placement using the PANEDWINDOW height + bounded retries ---
        def _set_sash_initial_retry(tries_left=10):
            try:
                self.update_idletasks()
                h = console_pane.winfo_height()  # measure the panedwindow itself
                if h and h > 40:
                    # Top (transcript/result) ~ 60%, Bottom (input) ~ 40%
                    console_pane.sashpos(0, int(h * 0.60))
                    return  # success; stop retrying
            except Exception:
                pass
            if tries_left > 0:
                self.after(60, lambda: _set_sash_initial_retry(tries_left - 1))

        # Schedule the first attempt once the window is idle
        self.after_idle(_set_sash_initial_retry)

        # RIGHT: tabs (History, Environment, Formulas)
        right = ttk.Notebook(outer)
        right.grid(row=0, column=1, sticky="ns")

        # History tab
        hist_tab = ttk.Frame(right, padding=(6, 6))
        right.add(hist_tab, text="History")
        self.tree = ttk.Treeview(hist_tab, columns=("expr", "result"), show="headings", height=22)
        self.tree.heading("expr", text="Expression")
        self.tree.heading("result", text="Result")
        self.tree.column("expr", width=320, anchor="w")
        self.tree.column("result", width=220, anchor="w")
        hist_v = ttk.Scrollbar(hist_tab, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=hist_v.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        hist_v.grid(row=0, column=1, sticky="ns")
        hist_tab.rowconfigure(0, weight=1)
        hist_tab.columnconfigure(0, weight=1)

        # History actions
        btns_hist = ttk.Frame(hist_tab)
        btns_hist.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(btns_hist, text="Reuse → Input", command=self._reuse_selected).pack(side="left")
        ttk.Button(btns_hist, text="Delete", command=self._delete_selected).pack(side="left", padx=(6, 0))

        # Environment tab
        env_tab = ttk.Frame(right, padding=(6, 6))
        right.add(env_tab, text="Environment")
        self.env = ttk.Treeview(env_tab, columns=("name", "value"), show="headings", height=22)
        self.env.heading("name", text="Name")
        self.env.heading("value", text="Value")
        self.env.column("name", width=120, anchor="w")
        self.env.column("value", width=380, anchor="w")
        env_v = ttk.Scrollbar(env_tab, orient="vertical", command=self.env.yview)
        self.env.configure(yscrollcommand=env_v.set)
        self.env.grid(row=0, column=0, sticky="nsew")
        env_v.grid(row=0, column=1, sticky="ns")
        env_tab.rowconfigure(0, weight=1)
        env_tab.columnconfigure(0, weight=1)

        # Clear Environment button under the Environment tree
        env_btns = ttk.Frame(env_tab)
        env_btns.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(env_btns, text="Clear Environment", command=self._clear_environment).pack(side="left")

        # Formulas tab
        form_tab = ttk.Frame(right, padding=(6, 6))
        right.add(form_tab, text="Formulas")
        self.formulas_tv = ttk.Treeview(form_tab, columns=("name", "expr"), show="headings", height=22)
        self.formulas_tv.heading("name", text="Name")
        self.formulas_tv.heading("expr", text="Expression")
        self.formulas_tv.column("name", width=160, anchor="w")
        self.formulas_tv.column("expr", width=440, anchor="w")
        form_v = ttk.Scrollbar(form_tab, orient="vertical", command=self.formulas_tv.yview)
        self.formulas_tv.configure(yscrollcommand=form_v.set)
        self.formulas_tv.grid(row=0, column=0, sticky="nsew")
        form_v.grid(row=0, column=1, sticky="ns")
        form_tab.rowconfigure(0, weight=1)
        form_tab.columnconfigure(0, weight=1)

        # Formulas actions
        btns_form = ttk.Frame(form_tab)
        btns_form.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(btns_form, text="Add…", command=self._add_formula).pack(side="left")
        ttk.Button(btns_form, text="Edit…", command=self._edit_formula).pack(side="left", padx=(6, 0))
        ttk.Button(btns_form, text="Run", command=self._run_selected_formula).pack(side="left", padx=(12, 0))
        ttk.Button(btns_form, text="→ Input", command=self._send_selected_formula_to_input).pack(side="left", padx=(6, 0))
        ttk.Button(btns_form, text="Delete", command=self._delete_selected_formula).pack(side="left", padx=(6, 0))
        self.formulas_tv.bind("<Double-1>", lambda e: self._send_selected_formula_to_input())
        self.formulas_tv.bind("<Return>", lambda e: (self._run_selected_formula(), "break"))
        self._reload_formulas_tv()

        # History context menu
        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="Reuse → Input", command=self._reuse_selected)
        self.menu.add_command(label="Copy expression", command=lambda: self._copy_field("expr"))
        self.menu.add_command(label="Copy result", command=lambda: self._copy_field("result"))
        self.tree.bind("<Button-3>", self._show_context_menu)
        self.tree.bind("<Double-1>", lambda e: self._reuse_selected())

        # Populate history/env views
        self._reload_tree()
        self._refresh_environment_view()

        # Menu bar
        self._build_menubar()

        # Bindings: Console behavior
        self._bind_console_keys()

        # Mouse wheel scrolling on hover (works without focusing the widget)
        self._enable_hover_scroll(self.transcript)
        self._enable_hover_scroll(self.input)

        # Close handling
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Put caret in the console input
        self.after(0, lambda: (self.input.focus_set(), self.input.see("end")))

        # Initial banner
        self._append_transcript(
            "Scientific console ready. Enter to run. Shift+Enter = new line. "
            "Type ';' to chain commands. 'ans' holds last result.\n",
            tag="banner",
        )

    # ---------- UI construction helpers ----------
    def _build_menubar(self):
        menubar = tk.Menu(self)

        # File
        m_file = tk.Menu(menubar, tearoff=0)
        m_file.add_command(label="Save Transcript…", command=self._save_transcript)
        m_file.add_separator()
        m_file.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=m_file)

        # Edit
        m_edit = tk.Menu(menubar, tearoff=0)
        m_edit.add_command(label="Clear Input", command=lambda: self._set_input_text(""))
        m_edit.add_command(label="Clear Console", command=self._clear_transcript)
        menubar.add_cascade(label="Edit", menu=m_edit)

        # Options
        m_opt = tk.Menu(menubar, tearoff=0)
        m_opt.add_checkbutton(label="Trig in Degrees", onvalue=True, offvalue=False, variable=self.use_degrees)
        menubar.add_cascade(label="Options", menu=m_opt)

        # Help
        m_help = tk.Menu(menubar, tearoff=0)
        m_help.add_command(label="Calculations reference", command=self._show_calculations_help)
        menubar.add_cascade(label="Help", menu=m_help)

        self.config(menu=menubar)

    def _init_transcript_tags(self):
        self.transcript.tag_configure("banner", foreground="#9CDCFE")
        self.transcript.tag_configure("prompt", foreground="#C586C0")
        self.transcript.tag_configure("expr", foreground="#D4D4D4")
        self.transcript.tag_configure("result", foreground="#B5CEA8")
        self.transcript.tag_configure("error", foreground="#F44747")

    def _append_transcript(self, text: str, tag: str | None = None):
        self.transcript.config(state="normal")
        if tag:
            self.transcript.insert("end", text, (tag,))
        else:
            self.transcript.insert("end", text)
        self.transcript.see("end")
        self.transcript.config(state="disabled")

    def _clear_transcript(self):
        self.transcript.config(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.config(state="disabled")

    def _save_transcript(self):
        path = filedialog.asksaveasfilename(
            title="Save Transcript", defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            data = self.transcript.get("1.0", "end-1c")
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
            messagebox.showinfo("Transcript", f"Saved to {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Transcript", f"Could not save: {e}")

    # ---------- SAFE Help window (non-modal Toplevel) ----------
    def _show_calculations_help(self):
        """
        Show a small, non-modal Toplevel with a calculations cheat sheet.
        Owned by the main window (transient); closing it won't touch the app.
        """
        if hasattr(self, "_help_win") and self._help_win and self._help_win.winfo_exists():
            try:
                self._help_win.deiconify()
                self._help_win.lift()
                self._help_win.focus_force()
            except Exception:
                pass
            return

        self._help_win = tk.Toplevel(self)
        self._help_win.title("Calculations reference")
        self._help_win.geometry("560x520+100+80")
        self._help_win.transient(self)
        self._help_win.resizable(True, True)

        def _close_help():
            try:
                if self._help_win and self._help_win.winfo_exists():
                    self._help_win.destroy()
            except tk.TclError:
                pass

        self._help_win.protocol("WM_DELETE_WINDOW", _close_help)

        frm = ttk.Frame(self._help_win, padding=(10, 10))
        frm.pack(fill="both", expand=True)

        txt = tk.Text(
            frm,
            wrap="word",
            bg="#1E1E1E",
            fg="#D4D4D4",
            insertbackground="#D4D4D4",
            font=("Consolas", 12),
            bd=1,
            relief="solid",
        )
        y = ttk.Scrollbar(frm, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=y.set)
        txt.pack(side="left", fill="both", expand=True)
        y.pack(side="right", fill="y")

        cheat = (
            "CALCULATIONS YOU CAN USE\n"
            "=========================\n\n"
            "• Basic arithmetic: + - * / // % ^ (use ^ or ** for power)\n"
            "  Examples: 2+3*4, (2+3)^2, 7//3, 7%3, 2**10\n\n"
            "• Parentheses & chaining with ';'\n"
            "  Example: a=5; b=3; a^2 + b^2\n\n"
            "• Implicit multiplication (auto-inserts '*'):\n"
            "  2x → 2*x | 2(x+1) → 2*(x+1) | (x+1)(x-1) → (x+1)*(x-1)\n"
            "  5sin(x) → 5*sin(x) | x2 → x*2 | )2 → )*2\n\n"
            "• Constants: pi, e, tau\n"
            "• Last result: ans (updated after every evaluation)\n\n"
            "• Functions (radians by default; toggle 'Degrees'):\n"
            "  sin, cos, tan, asin, acos, atan\n"
            "  exp, log (natural log), log10\n"
            "  sqrt, abs, round, floor, ceil, min, max, pow\n\n"
            "  Examples:\n"
            "  sin(pi/6), cos(0.5), tan(30)  # if Degrees checked, 30 is degrees\n"
            "  ln(10) or log(10), log10(1000), sqrt(2)\n"
            "  a=5; b=sin(pi/6); a+b\n"
            "  2x + 3, 2(x+1), (x+1)(x-1), 5sin(x)\n\n"
            "• Arrays (NumPy): You can assign arrays to variables (limited display).\n"
            "• Tips:\n"
            "  - Use ';' to run multiple expressions.\n"
            "  - 'ans' can be reused in the next command, e.g.: ans^2 + 1\n"
        )
        txt.insert("1.0", cheat)
        txt.config(state="disabled")

        btns = ttk.Frame(frm, padding=(0, 8))
        btns.pack(fill="x", side="bottom")
        ttk.Button(btns, text="Close", command=_close_help).pack(side="right")

    # ---------- Bindings for console behavior ----------
    def _bind_console_keys(self):
        # Run on Enter / Ctrl+Enter
        self.input.bind("<Return>", self._on_return_run)
        self.input.bind("<Control-Return>", self._on_return_run)
        self.input.bind("<KP_Enter>", self._on_return_run)
        # Newline on Shift+Enter
        self.input.bind("<Shift-Return>", self._on_shift_return_newline)
        self.input.bind("<Shift-KP_Enter>", self._on_shift_return_newline)
        # History navigation
        self.input.bind("<Up>", self._on_up_history)
        self.input.bind("<Down>", self._on_down_history)
        # Right panel: allow Enter to run from the history tree as well
        self.tree.bind("<Return>", lambda e: (self.evaluate(), "break"))
        self.tree.bind("<KP_Enter>", lambda e: (self.evaluate(), "break"))

    def _on_return_run(self, event=None):
        self.evaluate()
        return "break"

    def _on_shift_return_newline(self, event=None):
        self.input.insert("insert", "\n")
        return "break"

    def _on_up_history(self, event=None):
        # Only intercept when caret is at very start
        if self.input.index("insert linestart") != self.input.index("1.0"):
            return
        if self._hist_index > 0:
            self._hist_index -= 1
            self._set_input_text(self._cmd_history[self._hist_index])
            return "break"

    def _on_down_history(self, event=None):
        # Only intercept when caret is at very end
        if self.input.index("insert lineend") != self.input.index("end-1c"):
            return
        if self._hist_index < len(self._cmd_history) - 1:
            self._hist_index += 1
            self._set_input_text(self._cmd_history[self._hist_index])
        else:
            # Move past last item = empty input
            self._hist_index = len(self._cmd_history)
            self._set_input_text("")
        return "break"

    def _set_input_text(self, text: str):
        self.input.delete("1.0", "end")
        self.input.insert("1.0", text)
        self.input.see("end")
        self.input.focus_set()

    # ---------- Hover mouse-wheel scrolling ----------
    def _enable_hover_scroll(self, widget: tk.Widget):
        """
        Let the mouse wheel scroll a widget while hovering, without requiring focus.
        Works across Windows, macOS, and Linux (X11).
        """
        def _on_enter(_):
            # Windows / X11
            widget.bind_all("<MouseWheel>", _on_wheel, add="+")
            widget.bind_all("<Shift-MouseWheel>", _on_wheel, add="+")
            # Linux (Button-4/5)
            widget.bind_all("<Button-4>", _on_wheel_btn, add="+")
            widget.bind_all("<Button-5>", _on_wheel_btn, add="+")
            # macOS (delta is different but <MouseWheel> still fires)
            widget.bind_all("<Shift-Button-4>", _on_wheel_btn, add="+")
            widget.bind_all("<Shift-Button-5>", _on_wheel_btn, add="+")

        def _on_leave(_):
            # Remove only our handlers
            try:
                widget.unbind_all("<MouseWheel>")
                widget.unbind_all("<Shift-MouseWheel>")
                widget.unbind_all("<Button-4>")
                widget.unbind_all("<Button-5>")
                widget.unbind_all("<Shift-Button-4>")
                widget.unbind_all("<Shift-Button-5>")
            except Exception:
                pass

        def _on_wheel(event):
            # event.delta: Windows multiples of 120; macOS small deltas; X11 often 0
            delta = event.delta
            lines = 0
            if delta > 0:
                lines = -1  # scroll up
            elif delta < 0:
                lines = 1   # scroll down
            if lines:
                try:
                    widget.yview_scroll(lines, "units")
                except Exception:
                    pass
            return "break"

        def _on_wheel_btn(event):
            # X11 button events: 4 = up, 5 = down
            if event.num == 4:
                step = -1
            elif event.num == 5:
                step = 1
            else:
                step = 0
            if step:
                try:
                    widget.yview_scroll(step, "units")
                except Exception:
                    pass
            return "break"

        widget.bind("<Enter>", _on_enter, add="+")
        widget.bind("<Leave>", _on_leave, add="+")

    # ---------- History + Environment panels ----------
    def _show_context_menu(self, event):
        try:
            self.tree.selection_set(self.tree.identify_row(event.y))
        except Exception:
            pass
        self.menu.tk_popup(event.x_root, event.y_root)

    def _get_selected_item(self):
        sel = self.tree.selection()
        if not sel:
            return None
        try:
            i = int(sel[0])
        except Exception:
            return None
        base = max(0, len(self._history) - len(self.tree.get_children()))
        idx = base + i
        if 0 <= idx < len(self._history):
            return self._history[idx]
        return None

    def _reuse_selected(self):
        item = self._get_selected_item()
        if not item:
            return
        expr = item.get("expr", "")
        self._set_input_text(expr)

    def _copy_field(self, which: str):
        item = self._get_selected_item()
        if not item:
            return
        text = item.get(which, "")
        self.clipboard_clear()
        self.clipboard_append(str(text))

    def _delete_selected(self):
        sel = self.tree.selection()
        if not sel:
            return
        idxs = sorted([int(i) for i in sel], reverse=True)
        base = max(0, len(self._history) - len(self.tree.get_children()))
        for i in idxs:
            j = base + i
            if 0 <= j < len(self._history):
                del self._history[j]
        self._reload_tree()
        self._persist_history()

    def _reload_tree(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for i, item in enumerate(self._history[-MAX_HISTORY:]):
            self.tree.insert("", "end", iid=str(i), values=(item.get("expr", ""), item.get("result", "")))

    def _refresh_environment_view(self):
        for row in self.env.get_children():
            self.env.delete(row)
        items = []
        for k, v in sorted(self.vars.items(), key=lambda kv: kv[0]):
            items.append((k, self._val_to_str(v)))
        items.append(("ans", self._val_to_str(self.ans)))
        for name, val in items:
            self.env.insert("", "end", values=(name, val))

    # ---------- Environment control ----------
    def _clear_environment(self):
        """Clear all user-defined variables (keep ans)."""
        self.vars.clear()
        self._refresh_environment_view()
        self._append_transcript("[Environment cleared]\n", tag="banner")

    # ---------- Formulas panel handlers ----------
    def _reload_formulas_tv(self):
        for row in self.formulas_tv.get_children():
            self.formulas_tv.delete(row)
        for idx, item in enumerate(self._formulas):
            self.formulas_tv.insert("", "end", iid=str(idx), values=(item.get("name", ""), item.get("expr", "")))

    def _selected_formula_index(self) -> int | None:
        sel = self.formulas_tv.selection()
        if not sel:
            return None
        try:
            return int(sel[0])
        except Exception:
            return None

    def _add_formula(self):
        self._open_formula_editor(index=None)

    def _edit_formula(self):
        idx = self._selected_formula_index()
        if idx is None:
            messagebox.showinfo("Formulas", "Select a formula to edit.")
            return
        self._open_formula_editor(index=idx)

    def _open_formula_editor(self, index: int | None):
        """Open a small dialog to add/edit a formula (name + expression)."""
        win = tk.Toplevel(self)
        win.title("Add formula" if index is None else "Edit formula")
        win.geometry("520x320+120+100")
        win.transient(self)
        win.resizable(True, True)

        name_var = tk.StringVar(value=(self._formulas[index]["name"] if index is not None else ""))

        ttk.Label(win, text="Name:").pack(anchor="w", padx=10, pady=(10, 2))
        name_entry = ttk.Entry(win, textvariable=name_var)
        name_entry.pack(fill="x", padx=10)

        ttk.Label(win, text="Expression:").pack(anchor="w", padx=10, pady=(10, 2))
        expr_text = tk.Text(win, height=6, wrap="word", font=("Consolas", 12))
        expr_text.pack(fill="both", expand=True, padx=10)
        if index is not None:
            expr_text.insert("1.0", self._formulas[index].get("expr", ""))

        btns = ttk.Frame(win)
        btns.pack(fill="x", padx=10, pady=10)

        def _save():
            name = (name_var.get() or "").strip()
            expr = expr_text.get("1.0", "end-1c").strip()
            if not name:
                messagebox.showwarning("Formulas", "Please enter a name.")
                return
            if not expr:
                messagebox.showwarning("Formulas", "Please enter an expression.")
                return
            item = {"name": name, "expr": expr}
            if index is None:
                self._formulas.append(item)
            else:
                self._formulas[index] = item
            self._persist_formulas()
            self._reload_formulas_tv()
            try:
                win.destroy()
            except tk.TclError:
                pass

        ttk.Button(btns, text="Save", command=_save).pack(side="right")
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="right", padx=(0, 6))
        self.after(0, name_entry.focus_set)

    def _run_selected_formula(self):
        idx = self._selected_formula_index()
        if idx is None:
            messagebox.showinfo("Formulas", "Select a formula to run.")
            return
        expr = self._formulas[idx].get("expr", "").strip()
        if not expr:
            return
        self._set_input_text(expr)
        self.evaluate()

    def _send_selected_formula_to_input(self):
        idx = self._selected_formula_index()
        if idx is None:
            messagebox.showinfo("Formulas", "Select a formula to send to input.")
            return
        expr = self._formulas[idx].get("expr", "").strip()
        self._set_input_text(expr)

    def _delete_selected_formula(self):
        idx = self._selected_formula_index()
        if idx is None:
            messagebox.showinfo("Formulas", "Select a formula to delete.")
            return
        name = self._formulas[idx].get("name", "")
        if messagebox.askyesno("Formulas", f"Delete formula '{name}'?"):
            del self._formulas[idx]
            self._persist_formulas()
            self._reload_formulas_tv()

    # ---------- Evaluation ----------
    @staticmethod
    def _insert_implicit_multiplication(s: str) -> str:
        """Insert '*' for common cases without breaking f(x)."""
        s = s.strip()
        if not s:
            return s

        def fix_token(tok: str) -> str:
            t = tok
            # 1) 2x -> 2*x
            t = re.sub(r"(\d)\s*([A-Za-z])", r"\1*\2", t)
            # 2) x2 -> x*2  (single-letter variable followed by number)
            t = re.sub(r"([A-Za-z])\s*(\d)", r"\1*\2", t)
            # 3) 2(x+1) -> 2*(x+1)
            t = re.sub(r"(\d)\s*(\()", r"\1*\2", t)
            # 4) x(y) -> x*(y)   (single-letter var to avoid sin(x) -> sin*(x))
            t = re.sub(r"\b([A-Za-z])\s*(\()", r"\1*\2", t)
            # 5) )2 -> )*2
            t = re.sub(r"(\))\s*(\d)", r"\1*\2", t)
            # 6) )x -> )*x
            t = re.sub(r"(\))\s*([A-Za-z])", r"\1*\2", t)
            # 7) ')(' -> ')*('
            t = re.sub(r"(\))\s*(\()", r"\1*\2", t)
            # 8) 5sin(x) -> 5*sin(x)
            t = re.sub(
                r"(\d)\s*(sin|cos|tan|exp|ln|log|sinh|cosh|tanh|asin|acos|atan)\s*(\()",
                r"\1*\2\3",
                t,
                flags=re.IGNORECASE,
            )
            return t

        parts = [fix_token(tok) for tok in s.split(";")]
        return ";".join(parts)

    def _namespace(self):
        """Build allowed names for eval (no builtins)."""
        ns = dict(
            pi=np.pi,
            e=np.e,
            tau=2 * np.pi,
            abs=np.abs,
            sqrt=np.sqrt,
            pow=np.power,
            floor=np.floor,
            ceil=np.ceil,
            round=np.round,
            min=np.minimum,
            max=np.maximum,
            exp=np.exp,
            log=np.log,
            ln=np.log,
            log10=np.log10,
            sin=np.sin,
            cos=np.cos,
            tan=np.tan,
            asin=np.arcsin,
            acos=np.arccos,
            atan=np.arctan,
            ans=self.ans,
        )
        if self.use_degrees.get():
            ns.update(
                dict(
                    sin=lambda v: np.sin(np.deg2rad(v)),
                    cos=lambda v: np.cos(np.deg2rad(v)),
                    tan=lambda v: np.tan(np.deg2rad(v)),
                    asin=lambda v: np.rad2deg(np.arcsin(v)),
                    acos=lambda v: np.rad2deg(np.arccos(v)),
                    atan=lambda v: np.rad2deg(np.arctan(v)),
                )
            )
        ns.update(self.vars)
        return ns

    def _safe_eval(self, expr: str):
        """Evaluate semicolon-chained expressions w/ assignment in a restricted namespace."""
        if not expr.strip():
            return None
        s = self._insert_implicit_multiplication(expr)
        s = s.replace("^", "**").replace("ln(", "log(")
        tokens = [t.strip() for t in s.split(";") if t.strip()]
        value = None
        for tok in tokens:
            if "=" in tok:
                lhs, rhs = tok.rsplit("=", 1)
                var = lhs.strip()
                if not var.isidentifier():
                    raise ValueError(f"Invalid variable name: {var}")
                rhs_val = eval(rhs, {"__builtins__": {}}, self._namespace())
                self.vars[var] = rhs_val
                value = rhs_val
            else:
                value = eval(tok, {"__builtins__": {}}, self._namespace())
            # keep ans after each token
            try:
                self.ans = float(np.asarray(value).astype(float))
            except Exception:
                self.ans = value
        return value

    def evaluate(self):
        # Grab input as a single string
        expr = self.input.get("1.0", "end-1c").strip()
        if not expr:
            # prompt only
            self._append_transcript(">> \n", tag="prompt")
            return

        # show prompt + expr
        self._append_transcript(">> ", tag="prompt")
        self._append_transcript(expr + "\n", tag="expr")

        # Try to evaluate
        try:
            value = self._safe_eval(expr)
        except Exception as ex:
            self._append_transcript(f"{type(ex).__name__}: {ex}\n", tag="error")
            return

        # Render result to string
        if isinstance(value, np.ndarray):
            s = np.array2string(value, precision=10, suppress_small=False, max_line_width=120)
        else:
            try:
                v = float(np.asarray(value).astype(float))
                s = f"{v:.12g}"
            except Exception:
                s = str(value)

        self._append_transcript(s + "\n", tag="result")

        # Update history + environment
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._history.append({"time": ts, "expr": expr, "result": s})
        if len(self._history) > MAX_HISTORY:
            self._history = self._history[-MAX_HISTORY:]
        self._reload_tree()
        self._persist_history()
        self._refresh_environment_view()

        # Update command history list and index
        if not self._cmd_history or self._cmd_history[-1] != expr:
            self._cmd_history.append(expr)
        self._hist_index = len(self._cmd_history)

        # Clear input
        self._set_input_text("")

    # ---------- Persistence ----------
    def _load_history(self):
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data[-MAX_HISTORY:]
        except Exception:
            pass
        return []

    def _persist_history(self):
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self._history[-MAX_HISTORY:], f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_formulas(self):
        try:
            if os.path.exists(FORMULAS_FILE):
                with open(FORMULAS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    out = []
                    for it in data:
                        name = str(it.get("name", "")).strip()
                        expr = str(it.get("expr", "")).strip()
                        if name and expr:
                            out.append({"name": name, "expr": expr})
                    return out
        except Exception:
            pass
        return []

    def _persist_formulas(self):
        try:
            with open(FORMULAS_FILE, "w", encoding="utf-8") as f:
                json.dump(self._formulas, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ---------- Window close (robust on Windows) ----------
    def _on_close(self):
        """Persist then destroy safely after event returns (avoid TclError)."""
        if self._closing:
            return
        self._closing = True
        try:
            self._persist_history()
            self._persist_formulas()
        except Exception:
            pass
        try:
            self.protocol("WM_DELETE_WINDOW", None)  # prevent re-entry
        except Exception:
            pass
        self.after_idle(self._really_close)

    def _really_close(self):
        try:
            if getattr(self, "_is_root", False):
                try:
                    if self.master and hasattr(self.master, "winfo_exists") and self.master.winfo_exists():
                        self.master.destroy()
                except tk.TclError:
                    pass
            if hasattr(self, "winfo_exists") and self.winfo_exists():
                try:
                    self.destroy()
                except tk.TclError:
                    pass
        except Exception:
            pass

    # ---------- Utilities ----------
    @staticmethod
    def _val_to_str(v) -> str:
        if isinstance(v, np.ndarray):
            return np.array2string(v, precision=6, suppress_small=True, max_line_width=60)
        try:
            fv = float(np.asarray(v).astype(float))
            return f"{fv:.6g}"
        except Exception:
            return str(v)


# ---- Convenience launcher ----
def launch_calculator(master: Optional[tk.Misc] = None, title: str = "Scientific Calculator"):
    """
    Launch the calculator as a Toplevel if master is provided, otherwise standalone window.
    Returns the created ScientificCalculator instance.
    """
    win = ScientificCalculator(master=master, title=title)
    win.focus_set
    return win


if __name__ == "__main__":
    # Standalone run
    calc = launch_calculator(None)
    calc.mainloop()
