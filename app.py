"""Root entry point for Hugging Face Spaces and Streamlit deployment."""

import os
import sys

# Ensure current directory is on Python module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Execute the main Streamlit application
from lunar_correspondence.ui.app import *

