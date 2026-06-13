import os
import sys

# Make the src/ layout importable without installation.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
