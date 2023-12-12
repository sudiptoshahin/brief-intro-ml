import pstats
import time
import cProfile
import subprocess
import sys
import os
import tuna
import pandas as pd
from line_profiler import LineProfiler
from io import StringIO


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


def lineprofiler_df(lineprofiler):
    stream = StringIO()
    stats = pstats.Stats(lineprofiler, stream=stream)
    stats.print_stats()
    output = stream.getvalue()
    lines = output.strip().split('\n')

    data = []
    cols = ['Line', 'Hits', 'Time', 'Per Hit', '% Time', 'Line Contents']

    for line in lines[5:]:
        parts = line.split()
        if (len(parts)) >= 6:
            row = {
                'Line': parts[0],
                'Hits': parts[1],
                'Time': parts[2],
                'Per Hit': parts[3],
                '% Time': parts[4],
                'Line Contents': ' '.join(parts[5:])
            }
            data.append(row)

    df = pd.DataFrame(data, columns=cols)
    return df


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
    # for func in funclist:
    #     lp_wrapper = lineprofiler(func)
    #     lp_wrapper()
    #     print('\n\n')
    #     lineprofiler.print_stats(output_unit=1e-03)

    stream = StringIO()

    lp_wrapper = lineprofiler(func_1)
    lp_wrapper()
    lineprofiler.print_stats(stream=stream)
    output = stream.getvalue()
    lines = output.strip().split('\n')
    # lines = [line.strip() for line in lines]
    # print(lines)

    data = []
    cols = ['Line', 'Hits', 'Time', 'Per Hit', '% Time', 'Line Contents']

    print('------------------')
    # lines.pop(0)
    # lines.pop(2)
    # print(len(lines[5:]))
    newlines = [ line.split() for line in lines[6:]]
    # for line in lines[6:]:
    #     parts = line.split()
    #     newlines.append(parts)

    for line in newlines[2:]:
        # parts = [word.split() for word in newlines]
        # print(line)

        if len(line) >= 6:
            rows = {
                'Line': line[0],
                'Hits': line[1],
                'Time': line[2],
                'Per Hit': line[3],
                '% Time': line[4],
                'Line Contents': ' '.join(line[5:])
            }
            data.append(rows)

    df = pd.DataFrame(data)

    # print(f'hellooo: {lines[5:]}')
    print(df)

