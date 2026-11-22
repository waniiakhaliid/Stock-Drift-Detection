"""Central place for every folder the pipeline reads or writes (so scripts work from any directory)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"                 # input CSVs (indices, event calendars)
RESULTS = ROOT / "results"           # stored pickles (detector outputs, tables)
OUT = ROOT / "outputs"               # everything generated for the report
FIGS = OUT / "figs"
TABLES = OUT / "tables"
STUDY = ROOT / "user_study"          # E4 kit: stimuli, response sheet, analysis

for _p in (DATA, RESULTS, FIGS, TABLES, STUDY / "stimuli"):
    _p.mkdir(parents=True, exist_ok=True)


def data_file(name):
    """Resolve a CSV given on the command line: as typed if it exists, otherwise inside data/."""
    p = Path(name)
    return p if p.exists() else DATA / p.name
