import time


def py_loop(n):
    result = 0
    for i in range(n):
        temp_res = (i + 1) ** 3
        result += temp_res
    return result


def func_1():
    time.sleep(5.23)
    print(py_loop(20))


def func_2():
    result = py_loop(1020120)
    return result


def func_3():
    result = py_loop(1006500)
    return result

def func_4():
    result = py_loop(100)
    return result

def func_5():
    time.sleep(3)
    result = py_loop(30)
    return result


################profiling################

import pstats
import time
import subprocess
import sys
import os
import line_profiler
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

    def __init__(self, func=None, funcs=None):
        self._func = func
        self._funcs = funcs
        self.c_profiler = cProfile.Profile()
        self.line_profiler = LineProfiler()

    # cp = cProfiler
    # lp = line_profiler
    @property
    def func(self):
        return self._func

    # @func.setter
    def setUniFunctionLP(self, func):
        self._func = func

    @property
    def funcs(self):
        return self._funcs

    # @funcs.setter
    def setMultiFunctionLP(self, funcs):
        self._funcs = funcs


    def cProfilerEnable(self):
        self.c_profiler.enable()

    def cProfilerDisable(self):
        self.c_profiler.disable()

    def cprofiler_df(self):
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


    def test(self, func):
        stream = StringIO()
        output = ''
        lp_wrapper = self.line_profiler(func)
        lp_wrapper()
        self.line_profiler.print_stats(stream=stream)
        output = stream.getvalue()
        lines = output.strip().split('\n')
        newlines = [line.split() for line in lines]
        output = ''
        stream.close()
        return newlines

    def multiLineProfilerDataList(self):
        dataList = []
        for func in self._funcs:
            fn = self.test(func)
            dataList.append(fn)
        return dataList[-1]


    def multilineprofiler_df(self):
        stream = StringIO()
        lp_wrapper = None

        data = []
        if self._funcs is not None or self._funcs is isinstance(self._funcs, list):
            rawList = self.multiLineProfilerDataList()
            empty_list_indices = [index for index, sublist in enumerate(rawList) if not sublist]
            print(empty_list_indices)
            # for data in rawList:
            #     print(data)
            data = []
            for idx in range(len(empty_list_indices)):
                if idx != 0 and idx != len(empty_list_indices)-2:
                    obj = rawList[empty_list_indices[idx]: empty_list_indices[idx+1]]
                    data.append(obj)
                elif idx != 0 and idx != len(empty_list_indices)-1:
                    obj = rawList[empty_list_indices[idx]: empty_list_indices[idx+1]]

            print(data)




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
        df = self.singlelineprofiler_df()
        print(f'line-profiler: \n{df}')


###########################################

if __name__ == '__main__':
    profiling = Profiling()
    # profiling.cProfilerEnable()
    print(func_1())
    print(func_2())
    print(func_3())
    print(func_4())
    print(func_5())
    # profiling.cProfilerDisable()
    # profiling.getCrofilerStat()

    profiling.setUniFunctionLP(func_1)
    profiling.setMultiFunctionLP([func_1, func_2, func_3, func_4, func_5])
    # print(profiling.funcs)
    # profiling.getSingleLineProfilerStat()
    # print(profiling.multilineprofiler_df())
    profiling.multilineprofiler_df()
