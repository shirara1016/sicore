"""Making the documentation for the project."""

import shutil
import subprocess
from pathlib import Path

subprocess.run(["make", "html"], check=False)

path = Path("../docs")
if path.exists():
    shutil.rmtree(path.resolve(), ignore_errors=True)

path = Path("_build/html")
path.rename(Path("../docs"))

Path("../docs/.nojekyll").touch()

path = Path("_build")
shutil.rmtree(path.resolve(), ignore_errors=True)
