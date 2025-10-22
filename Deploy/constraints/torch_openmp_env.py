"""Runtime hook to prevent Intel OpenMP initialization conflicts in frozen builds."""

import os

# Allow duplicate Intel OpenMP runtimes that ship with torch/numpy/scipy
# to coexist inside the bundled application.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
