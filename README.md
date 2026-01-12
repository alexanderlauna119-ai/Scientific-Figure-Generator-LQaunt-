# Statistics-program-for-medicine
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
- Multiple plot types: Bar + scatter, Box, Violin, Strip, Mean ± CI, Line ± CI, Line (means), Area (quartiles stacked), Lines (series), Areas (series), Regression (series), Regression (global), Pie chart

# Installation
1. Install Python 3.10+.
2. Install dependencies: pip install -r requirements.txt

Optional dependencies for certain analyses: pip install scikit-posthocs statsmodels openpyxl xlrd

# Usage
Run the program: python main.py

# Cross-Platform Usage (Windows, macOS, Linux)

### macOS
- Install via [python.org](https://www.python.org) or Homebrew:
```bash
brew install python
```
- If the GUI fails to start on older macOS, install XQuartz.

### Linux
- Install Python and Tkinter:
```bash
sudo apt-get install python3 python3-pip python3-tk
```

# Notes
- `tkinter` is required for the GUI and is typically bundled with Python on macOS/Windows; on Linux, install `python3-tk`.
- Excel support uses `openpyxl` (for .xlsx) and `xlrd` (legacy .xls).

# Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature-name`)
5. Open a Pull Request


