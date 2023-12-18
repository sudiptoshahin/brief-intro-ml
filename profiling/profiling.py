import pstats
import time
import subprocess
import sys
import os
import pandas as pd
from io import StringIO
import cProfile
from line_profiler import LineProfiler
import re


class Profiling:
    """
        # profiler is a type of time_profiling method like (CProfiler, line_profiler)
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
        self.line_profiler = LineProfiler()

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


    def cProfilerEnable(self):
        self.c_profiler.enable()


    def cProfilerDisable(self):
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
        df = self.__cprofiler_df()
        df.to_csv('cp.csv', header=True, index=True, index_label='Index', encoding='utf-8')

    def get_single_line_profiler_stat(self):
        df = self.get_lp_obj(self._func)
        filename = f"single_lp.csv"
        df.to_csv(filename, header=True, index=True, index_label='Index', encoding='utf-8')

    def get_multi_line_profiler_stat(self):
        self.multi_df_csv(self._funcs)
        # df.to_csv('multiple_lp.csv', index=True, index_label='Index', encoding='utf-8')

    def do_line_profile(self, follow=[]):
        def innrer(func):
            def profiled_func(*args, **kwargs):
                try:
                    # profiler = self.line_profiler()
                    self.line_profiler.add_function(func)
                    for f in follow:
                        self.line_profiler.add_function(f)
                    self.line_profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    self.line_profiler.print_stats()
                    return innrer

    #### multi function line_profile

    def multi_df_csv(self, functions: list):
        new_df_list = []
        for idx, fn in enumerate(functions):
            result_obj = self.get_lp_obj(fn)  # dict length = 6
            new_df_list.append(result_obj)

        new_df = pd.concat(new_df_list)
        new_df.to_csv('multi_line_profiler1.csv')



    def get_lp_obj(self, func):
        stream = StringIO()

        line_wrapper = self.line_profiler(func)
        line_wrapper()
        self.line_profiler.print_stats(stream=stream)

        output = stream.getvalue()
        output = re.split(r', \n', output)
        output = ' '.join(output).strip().split('\n')
        # stream.close()

        lists = []
        for line in output[8:]:
            parts = line.split()
            # print(len(parts))
            if (len(parts)) >= 6:
                row = {
                    'Line': parts[0],
                    'Hits': parts[1],
                    'Time': parts[2],
                    'Per Hit': parts[3],
                    '% Time': parts[4],
                    'Line Contents': ' '.join(parts[5:])
                }
                lists.append(row)
        df = pd.DataFrame(lists)
        return df

        # return rows
