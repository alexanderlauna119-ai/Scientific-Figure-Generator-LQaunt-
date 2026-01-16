# Scientific Figure Generator: Lightweight Tool for Fast, Customizable Data Visualization and Analysis
A statistical program with good quality for exported pictures. The plots are costumizable and easy to use.  

# LQ — Data Analysis Program

LQ is a Python-based data analysis tool with a GUI for statistical tests and visualization.

# Features
- Q–Q plots for normality checks
- Games–Howell post-hoc tests (optional dependency)
- Two-way ANOVA (optional dependency)
- Friedman test (nonparametric repeated measures)
- Mann–Whitney U and Wilcoxon paired tests
- Descriptive statistics with confidence intervals
- Multiple plot types: Bar + scatter, Strip, Mean ± CI, Line ± CI, Line (means), Area (quartiles stacked), Lines (series), Areas (series), Regression (series), Regression (global), Pie chart

# Windows
1. Install Python 3.10+.
3. Run the program: python main.py
-  download geometry_canvas.py for acces to the canvas function, analysis_extensions for exponential functions, and place it in the same folder ad LQuant.py


### macOS
- Install via [python.org](https://www.python.org) or Homebrew:
- If the GUI fails to start on older macOS, install XQuartz.


# Notes
- `tkinter` is required for the GUI and is typically bundled with Python on macOS/Windows; on Linux, install `python3-tk`.
- Excel support uses `openpyxl` (for .xlsx) and `xlrd` (legacy .xls).




