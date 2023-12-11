import pstats
import time
import subprocess
import sys
import os
import tuna
import pandas as pd
from io import StringIO
import cProfile
from line_profiler import LineProfiler


class Profiling:

    """
        # profiler is a type of profiling method like (CProfiler, line_profiler)
        # functions is a list of function or single function reference
        # this @Profiling class needs to instantiate with these parameters
    """
    """
        * pass a single function / list of function
        * function list can be used only in line profiler
        * method/profiler = line_profiler or CProfiler
        * Generate CSV reports
    """
    def __init__(self, profiler, functions):
        self.profiler = profiler
        self.functions = functions



