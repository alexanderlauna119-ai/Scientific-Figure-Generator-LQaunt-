
import tkinter as tk
from tkinter import ttk, messagebox


def launch_chemical_calculator():
    win = tk.Toplevel()
    win.title("Chemical Calculator")
    win.geometry("420x420")

    notebook = ttk.Notebook(win)
    notebook.pack(fill="both", expand=True)

    # ---------------------------
    # TAB 1: Basic molarity
    # ---------------------------
    frame1 = ttk.Frame(notebook)
    notebook.add(frame1, text="Molarity (m = V·MW·c)")

    m_var = tk.StringVar()
    mw_var = tk.StringVar()
    c_var = tk.StringVar()
    v_var = tk.StringVar()
    unit_var = tk.StringVar(value="L")

    def add_row(parent, label, var, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(parent, textvariable=var, width=18).grid(row=row, column=1, padx=6, pady=6)

    add_row(frame1, "Mass (g)", m_var, 0)
    add_row(frame1, "Molecular Weight (g/mol)", mw_var, 1)
    add_row(frame1, "Concentration (mol/L)", c_var, 2)

    # Volume row with unit
    ttk.Label(frame1, text="Volume").grid(row=3, column=0, sticky="w", padx=6, pady=6)
    ttk.Entry(frame1, textvariable=v_var, width=10).grid(row=3, column=1, sticky="w", padx=6)
    ttk.Combobox(frame1, textvariable=unit_var, values=["L", "mL"], width=6, state="readonly").grid(row=3, column=1, sticky="e", padx=6)

    def get_float(var):
        try:
            return float(var.get())
        except:
            return None

    def volume_to_L(v):
        if v is None:
            return None
        return v / 1000 if unit_var.get() == "mL" else v

    def volume_from_L(v):
        if v is None:
            return None
        return v * 1000 if unit_var.get() == "mL" else v

    def solve():
        m = get_float(m_var)
        mw = get_float(mw_var)
        c = get_float(c_var)
        v = get_float(v_var)

        v_L = volume_to_L(v)

        known = [m is not None, mw is not None, c is not None, v_L is not None]

        try:
            # Solve whichever is missing
            if m is None and mw and c and v_L:
                m = mw * c * v_L
                m_var.set(f"{m:.6g}")

            elif mw is None and m and c and v_L:
                mw = m / (c * v_L)
                mw_var.set(f"{mw:.6g}")

            elif c is None and m and mw and v_L:
                c = m / (mw * v_L)
                c_var.set(f"{c:.6g}")

            elif v_L is None and m and mw and c:
                v_L = m / (mw * c)
                v_var.set(f"{volume_from_L(v_L):.6g}")

            else:
                messagebox.showinfo("Input needed", "Fill 3 fields to solve the 4th.")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    ttk.Button(frame1, text="Solve", command=solve).grid(row=5, column=0, columnspan=2, pady=12)

    # ---------------------------
    # TAB 2: Dilution calculator
    # ---------------------------
    frame2 = ttk.Frame(notebook)
    notebook.add(frame2, text="Dilution (C1V1=C2V2)")

    c1_var = tk.StringVar()
    v1_var = tk.StringVar()
    c2_var = tk.StringVar()
    v2_var = tk.StringVar()

    add_row(frame2, "C1 (stock)", c1_var, 0)
    add_row(frame2, "V1 (used)", v1_var, 1)
    add_row(frame2, "C2 (final)", c2_var, 2)
    add_row(frame2, "V2 (final)", v2_var, 3)

    def solve_dilution():
        c1 = get_float(c1_var)
        v1 = get_float(v1_var)
        c2 = get_float(c2_var)
        v2 = get_float(v2_var)

        try:
            if c1 is None and v1 and c2 and v2:
                c1 = (c2 * v2) / v1
                c1_var.set(f"{c1:.6g}")

            elif v1 is None and c1 and c2 and v2:
                v1 = (c2 * v2) / c1
                v1_var.set(f"{v1:.6g}")

            elif c2 is None and c1 and v1 and v2:
                c2 = (c1 * v1) / v2
                c2_var.set(f"{c2:.6g}")

            elif v2 is None and c1 and v1 and c2:
                v2 = (c1 * v1) / c2
                v2_var.set(f"{v2:.6g}")

            else:
                messagebox.showinfo("Input needed", "Fill 3 fields to solve the 4th.")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    ttk.Button(frame2, text="Solve Dilution", command=solve_dilution).grid(row=5, column=0, columnspan=2, pady=12)

    # ---------------------------
    # TAB 3: Unit converter
    # ---------------------------
    frame3 = ttk.Frame(notebook)
    notebook.add(frame3, text="Units")

    value_var = tk.StringVar()
    from_unit = tk.StringVar(value="mL")
    to_unit = tk.StringVar(value="L")
    result_var = tk.StringVar()

    add_row(frame3, "Value", value_var, 0)

    ttk.Label(frame3, text="From").grid(row=1, column=0)
    ttk.Combobox(frame3, textvariable=from_unit,
                 values=["mL", "L", "mg", "g"],
                 state="readonly").grid(row=1, column=1)

    ttk.Label(frame3, text="To").grid(row=2, column=0)
    ttk.Combobox(frame3, textvariable=to_unit,
                 values=["mL", "L", "mg", "g"],
                 state="readonly").grid(row=2, column=1)

    ttk.Label(frame3, text="Result").grid(row=3, column=0)
    ttk.Entry(frame3, textvariable=result_var).grid(row=3, column=1)

    def convert():
        try:
            val = float(value_var.get())
            f = from_unit.get()
            t = to_unit.get()

            # Volume
            if f == "mL" and t == "L":
                val = val / 1000
            elif f == "L" and t == "mL":
                val = val * 1000

            # Mass
            elif f == "mg" and t == "g":
                val = val / 1000
            elif f == "g" and t == "mg":
                val = val * 1000

            result_var.set(f"{val:.6g}")

        except:
            messagebox.showerror("Error", "Invalid value.")

    ttk.Button(frame3, text="Convert", command=convert).grid(row=4, column=0, columnspan=2, pady=10)


# If run standalone (testing)
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    launch_chemical_calculator()
    root.mainloop()
