# Project Goal
Describe you project here.

## Data Structure
Describe your data structure here.


# Coding Style
- use `uv run` to run Python scripts
- ALWAYS use `uv ruff check` to lint the code and `uvx ty check` to type check it 
- ALWAYS use `uv ruff format` to format the code according to our guidelines after you 
  are done editing a file
- all functions must have type hints for their arguments and return values 
- all functions should have docstings (numpy style), the docstring should be short and 
  simple. You MUST avoid notes sections and exemples sections in docstrings unless strictly necessary, 
  you should check with me if you think such a necessity arises.
- code should focus on simplicity and clarity, avoid scope creep, avoid implementing 
  non-necessary options, ALWAYS think about the DRY, KISS and YAGNI principles when implementing 
  new functionalities.
- polars should be prefered over pandas where possible, the only exceptions being 
  geo operations where geopandas should be used
- project configurations and data paths are defined in `config.yaml` and `src/python_base/config.py`, 
  data folder paths should not be hardcoded in scripts, rather they should be loaded from configurations
