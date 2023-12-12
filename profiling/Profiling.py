import pstats
import time
import subprocess
import sys
import os
# import line_profiler
# import tuna
import pandas as pd
from io import StringIO
import cProfile
# from line_profiler import LineProfiler


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

    def __init__(self, func=None, funcs=None):
        self._func = func
        self._funcs = funcs
        self.c_profiler = cProfile.Profile()
        # self.line_profiler = LineProfiler()

    # cp = cProfiler
    # lp = line_profiler

    @property
    def func(self):
        return self.func

    def setUniFunctionLP(self, func):
        self._func = func

    @property
    def funcs(self):
        return self._funcs

    def setMultiFunctionLP(self, funcs: list):
        self._funcs = funcs

    @classmethod
    def cProfilerEnable(cls, self):
        if self.profiler == 'cp':
            self.c_profiler.enable()

    @classmethod
    def cProfilerDisable(cls, self):
        if self.profiler == 'cp':
            self.c_profiler.disable()

    def __cprofiler_df(self):
        stream = StringIO()
        stats = pstats.Stats(self.c_profiler, stream=stream)
        stats.print_stats()
        output = stream.getvalue()
        lines = output.strip().split('\n')

        data = []
        cols = ['ncalls', 'totime', 'percall', 'cumtime', 'percall_cum', 'filename:lineno(func)']

        for line in lines[5:]:
            parts = line.split()
            if (len(parts)) >= 6:
                row = {
                    'ncalls': parts[0],
                    'totime': parts[1],
                    'percall': parts[2],
                    'cumtime': parts[3],
                    'percall_cum': parts[4],
                    'filename:lineno(func)': ' '.join(parts[5:])
                }
                data.append(row)

        df = pd.DataFrame(data, columns=cols)
        return df


    def singlelineprofiler_df(self):
        stream = StringIO()
        lp_wrapper = None
        if self._func is not None and self._func is not isinstance(self._func, list):
            lp_wrapper = self.line_profiler(self._func)

        lp_wrapper()
        self.line_profiler.print_stats(stream=stream)
        output = stream.getvalue()
        lines = output.strip().split('\n')

        data = []
        cols = ['Line', 'Hits', 'Time', 'Per Hit', '%Time', 'Line Contents']

        newlines = [line.split() for line in lines[6:]]

        for line in newlines[2:]:
            if len(line) >= 6:
                rows = {
                    'Line': line[0],
                    'Hits': line[1],
                    'Time': line[2],
                    'Per hit': line[3],
                    '%Time': line[4],
                    'Line contents': ' '.join(line[5:])
                }
                data.append(rows)
        df = pd.DataFrame(data)

        return df


    def getCrofilerStat(self):
        df = self.cprofiler_df()
        # convert to csv
        print(f'\n---------------------------------------- {df} \n')

    def getSingleLineProfilerStat(self):
        df = self.lineprofiler_df()
        print(f'line-profiler: {df}')

    @_func.setter
    def _func(self, value):
        self.__func = value
