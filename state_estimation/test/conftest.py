import os
import sys

# Let the tests import the package straight from the source tree (no colcon build needed).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
