import os
import sys

# The code imports its packages as top-level modules (``from llm import ...``),
# the same way main.py is run from inside code/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # shared test helpers

import matplotlib  # noqa: E402

matplotlib.use('Agg')
