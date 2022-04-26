"""Utility functions"""


import os
from pathlib import Path


DATAPATH = os.path.join(Path(__file__).parent.absolute(), 'data')
FIGSPATH = os.path.join(Path(__file__).parent.absolute().parent.absolute(), 'figures')
