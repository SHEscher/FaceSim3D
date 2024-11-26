"""Generate the code reference pages and navigation."""  # noqa: INP001

from pathlib import Path
import shutil

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()
mod_symbol = '<code class="doc-symbol doc-symbol-nav doc-symbol-module"></code>'

# First copy the notebook to the docs/ folder
# We do the following manually, for now
# notebook_src = Path(__file__).parent.parent / Path("code", "notebooks", "facesim_results.ipynb")
# notebook_dst = Path(__file__).parent.parent / Path("docs", "notebooks", "facesim_results.ipynb")
# notebook_dst.parent.mkdir(parents=True, exist_ok=True)
#
# # Remove the "old" notebook in docs/ first
# if notebook_dst.exists():
#     notebook_dst.unlink()
#
# # Copy the notebook to the docs/ folder
# if notebook_src.exists():
#     shutil.copy(notebook_src, notebook_dst)
# else:
#     print(f"Notebook not found: '{notebook_src}'")

# Create the reference pages
for path in sorted(Path("code").rglob("*.py")):
    if "tests" in path.parts or "SPoSE" in path.parts or "VICE" in path.parts:
        continue
    module_path = path.relative_to("code").with_suffix("")
    doc_path = path.relative_to("code/facesim3d").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1].startswith("_"):
        continue

    nav_parts = [f"{mod_symbol} {part}" for part in parts]
    nav[tuple(nav_parts)] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, ".." / path)

with mkdocs_gen_files.open("reference/SUMMARY.txt", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
