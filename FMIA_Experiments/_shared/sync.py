"""Copy the shared code (fmia/, run.py, the Kaggle notebook) into every experiment folder.

Each E* folder must run on its own Kaggle account with no reference to any other
folder, so the code is duplicated on purpose.  Edit the master copy here, then:

    python FMIA_Experiments/_shared/sync.py          # copy
    python FMIA_Experiments/_shared/sync.py --check  # exit 1 if any copy is stale
"""
import filecmp
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FOLDERS = sorted(d for d in os.listdir(ROOT) if d.startswith("E") and os.path.isdir(os.path.join(ROOT, d)))
NOTEBOOK_TEMPLATE = os.path.join(HERE, "kaggle_notebook_template.ipynb")


def notebook_for(folder):
    with open(NOTEBOOK_TEMPLATE) as f:
        txt = f.read()
    return txt.replace("__FOLDER__", folder)


def files_for(folder):
    yield os.path.join(HERE, "run.py"), os.path.join(ROOT, folder, "run.py")
    src = os.path.join(HERE, "fmia")
    for name in sorted(os.listdir(src)):
        if name.endswith(".py"):
            yield os.path.join(src, name), os.path.join(ROOT, folder, "fmia", name)


def main():
    check = "--check" in sys.argv
    stale = []
    for folder in FOLDERS:
        for s, d in files_for(folder):
            if not os.path.exists(d) or not filecmp.cmp(s, d, shallow=False):
                stale.append(d)
                if not check:
                    os.makedirs(os.path.dirname(d), exist_ok=True)
                    shutil.copy2(s, d)
        nb = os.path.join(ROOT, folder, "kaggle_notebook.ipynb")
        want = notebook_for(folder)
        json.loads(want)
        have = open(nb).read() if os.path.exists(nb) else None
        if have != want:
            stale.append(nb)
            if not check:
                with open(nb, "w") as f:
                    f.write(want)
    if check:
        print("\n".join(stale) if stale else "all copies up to date")
        sys.exit(1 if stale else 0)
    print(f"synced {len(FOLDERS)} folders: {', '.join(FOLDERS)} ({len(stale)} files updated)")


if __name__ == "__main__":
    main()
