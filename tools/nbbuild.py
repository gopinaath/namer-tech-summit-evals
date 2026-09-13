#!/usr/bin/env python3
"""Build the workshop notebooks from percent-format Python sources.

Three notebooks, two sources:

  Building_an_Eval.ipynb              the participant workbook, from workbook_src.py
  demo/Building_an_Eval_DEMO.ipynb    the same, with the ✏️ YOUR TURN cells replaced by
                                      the worked solutions in demo_cells/
  Bigger_Model_or_Better_Agent.ipynb  the model-vs-tuning lab, from lab_src.py

Keeping one source per notebook means a fix to the harness, the setup cell or the prose
can't drift between variants — only the cells that are genuinely different are duplicated.

    python tools/nbbuild.py            # rebuild every notebook
    python tools/nbbuild.py --check    # exit 1 if a notebook is out of date

Percent format (the convention jupytext uses):

    # %% [markdown] id=title
    # Comment lines after the marker are the markdown body.

    # %% id=tasks
    print("a code cell")

    # %% include=setup
    # Splices in every cell of shared/setup.py at this point.

Cells are separated by `# %%` markers. `id=` is optional and only needed for cells the
demo overrides. Markdown bodies are comment lines; the leading "# " is stripped so
markdown indentation survives.

`include=<name>` pulls in `shared/<name>.py`, so notebooks that need the same credential
and install handling share one copy of it rather than two that drift. Includes are not
recursive.

Overrides live in `demo_cells/<id>.py` (code) or `demo_cells/<id>.md` (markdown).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import nbformat

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / "workbook_src.py"
LAB_SRC = HERE / "lab_src.py"
DEMO_CELLS = HERE / "demo_cells"
SHARED = HERE / "shared"

WORKBOOK = ROOT / "Building_an_Eval.ipynb"
DEMO = ROOT / "demo" / "Building_an_Eval_DEMO.ipynb"
LAB = ROOT / "Bigger_Model_or_Better_Agent.ipynb"


def parse_percent(text: str) -> list[tuple[str, str | None, str]]:
    """Split percent-format text into [(cell_type, cell_id, source), ...].

    `# %% include=<name>` becomes an ("include", name, name) entry for expand_includes()
    to replace. It has to survive this far because the empty-cell filter below would
    otherwise drop a marker with no body of its own."""
    cells: list[tuple[str, str | None, list[str]]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %%"):
            if "include=" in stripped:
                name = stripped.split("include=", 1)[1].split()[0]
                cells.append(("include", name, [name]))
                continue
            kind = "markdown" if "[markdown]" in stripped else "code"
            cell_id = None
            if "id=" in stripped:
                cell_id = stripped.split("id=", 1)[1].split()[0]
            cells.append((kind, cell_id, []))
            continue
        if not cells:
            continue  # header comments above the first marker
        if cells[-1][0] == "include":
            continue  # commentary under an include marker is documentation, not content
        cells[-1][2].append(line)

    out = []
    for kind, cell_id, lines in cells:
        if kind == "markdown":
            body = []
            for line in lines:
                if line.startswith("# "):
                    body.append(line[2:])
                elif line.strip() == "#":
                    body.append("")
                else:
                    body.append(line)
            source = "\n".join(body).strip("\n")
        else:
            source = "\n".join(lines).strip("\n")
        if source.strip():
            out.append((kind, cell_id, source))
    return out


def expand_includes(cells) -> list[tuple[str, str | None, str]]:
    """Replace ("include", name, name) entries with the cells of shared/<name>.py."""
    out = []
    for kind, cell_id, source in cells:
        if kind != "include":
            out.append((kind, cell_id, source))
            continue
        path = SHARED / f"{cell_id}.py"
        if not path.is_file():
            raise SystemExit(f"include=({cell_id}) has no file at {path}")
        for inner in parse_percent(path.read_text()):
            if inner[0] == "include":
                raise SystemExit(f"{path.name} includes {inner[1]}; includes aren't recursive")
            out.append(inner)
    return out


def load_source(path: pathlib.Path):
    return expand_includes(parse_percent(path.read_text()))


def load_overrides() -> dict[str, tuple[str, str]]:
    """{cell_id: (cell_type, source)} from demo_cells/*.py and *.md."""
    overrides: dict[str, tuple[str, str]] = {}
    if not DEMO_CELLS.is_dir():
        return overrides
    for path in sorted(DEMO_CELLS.iterdir()):
        if path.suffix == ".py":
            overrides[path.stem] = ("code", path.read_text().strip("\n"))
        elif path.suffix == ".md":
            overrides[path.stem] = ("markdown", path.read_text().strip("\n"))
    return overrides


def build(cells, overrides=None) -> nbformat.NotebookNode:
    overrides = overrides or {}
    unused = set(overrides)
    nb = nbformat.v4.new_notebook()
    for index, (kind, cell_id, source) in enumerate(cells):
        if cell_id and cell_id in overrides:
            kind, source = overrides[cell_id]
            unused.discard(cell_id)
        new = nbformat.v4.new_markdown_cell if kind == "markdown" else nbformat.v4.new_code_cell
        # Deterministic cell ids. nbformat assigns a random one otherwise, which makes every
        # build differ textually from the last — `--check` would always report "out of date",
        # and every rebuild would produce a noisy diff.
        nb.cells.append(new(source, id=f"cell-{index:03d}"))
    if unused:
        raise SystemExit(
            f"demo_cells/ has overrides with no matching `id=` in {SRC.name}: "
            + ", ".join(sorted(unused))
        )
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    return nb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if a notebook differs from its source")
    args = ap.parse_args()

    cells = load_source(SRC)
    variants = [
        (WORKBOOK, build(cells)),
        (DEMO, build(cells, load_overrides())),
    ]
    if LAB_SRC.is_file():
        variants.append((LAB, build(load_source(LAB_SRC))))

    stale = []
    for dest, nb in variants:
        rendered = nbformat.writes(nb) + "\n"
        if args.check:
            if (dest.read_text() if dest.exists() else None) != rendered:
                stale.append(dest)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rendered)
        print(f"wrote {dest.relative_to(ROOT)}  ({len(nb.cells)} cells)")

    if stale:
        for dest in stale:
            print(f"out of date: {dest.relative_to(ROOT)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
