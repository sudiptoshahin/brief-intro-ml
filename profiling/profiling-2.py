import pstats
import time
import cProfile
import subprocess
import sys
import os
import tuna
import pandas as pd
from io import StringIO


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


def profile_df(profile_stats):
    stream = StringIO()

    stats = pstats.Stats(profile_stats, stream=stream)
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


if __name__ == '__main__':
    profile = cProfile.Profile()
    profile.enable()
    print(func_1())
    print(func_2())
    print(func_3())
    print_func()
    profile.disable()
    profile.print_stats()
    # df = pd.DataFrame(profile.getstats(), columns=['ncalls', ''])
    # df.to_csv('/home/sudiptoshahin/Projects/JupyterNotebookFiles/brief-intro-ml/profiling/test.csv')
    df = profile_df(profile)
    print(df)