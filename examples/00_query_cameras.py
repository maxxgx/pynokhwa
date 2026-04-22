"""
Query a list of available cameras
"""

import pynokhwa

print(*pynokhwa.query(), sep="\n")
