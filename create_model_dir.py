from pathlib import Path
import shutil

src = Path("template")
dst = Path("models/does_this_even_work")

# make sure parent dirs exist
dst.parent.mkdir(parents=True, exist_ok=True)

# copy everything; if destination exists, merge/overwrite files
# (Python 3.8+: dirs_exist_ok)
shutil.copytree(src, dst, dirs_exist_ok=True, copy_function=shutil.copy2)

print(f"Copied {src} -> {dst}")