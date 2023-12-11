import pstats
import time
import cProfile
import subprocess
import sys
import os
import tuna
import pandas as pd
from line_profiler import LineProfiler


def py_loop(n):
    result = 0
    for i in range(n):
        temp_res = (i+1) ** 3
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


def print_func():
    print('--------done----------')


if __name__ == '__main__':
    print(func_1())
    print(func_2())
    print(func_3())

    # linepro = LineProfiler()
    # lp_wrapper = linepro(func_2)
    # lp_wrapper()
    # linepro.print_stats(output_unit=1e-03)

    # print('function: ', type(func_2))
    # print(f'lineprofiler object: {type(linepro)}')
    # print(f'linewrapper: {type(lp_wrapper)}')

    lineprofiler = LineProfiler()
    funclist = [func_1, func_2, func_3]
    for func in funclist:
        lp_wrapper = lineprofiler(func)
        lp_wrapper()
        print('\n\n')
        lineprofiler.print_stats(output_unit=1e-03)





