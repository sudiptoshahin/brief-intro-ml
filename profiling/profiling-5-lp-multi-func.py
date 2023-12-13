import json
import re
from line_profiler import LineProfiler
import time
from io import StringIO

def func_1():
    time.sleep(1.5)
    for i in range(1000):
        res = 0
        res = res + (i ** 2)
        return res

def func_2():
    time.sleep(2.0)
    for i in range(500):
        res = 0
        res = res + (i ** 2)
        return res

def func_3():
    time.sleep(3.5)
    for i in range(100):
        res = 0
        res = res + (i ** 2)
        return res



if __name__ == '__main__':
    # print(func_1())
    # print(func_2())
    # print(func_3())

    stream = StringIO()
    lineprofiler = LineProfiler()
    lp_wrapper = lineprofiler(func_3)
    lp_wrapper()
    lineprofiler.print_stats(stream=stream)

    output = stream.getvalue()
    print(output)
    # output = output.strip().split('\n')
    output = re.split(r', \n,', output)
    output = ' '.join(output).strip().split('\n')

    # tempstr = output.split('\n')
    # rows = dict()
    # for idx, str in enumerate(output[8:]):
    #     # lists.append(str.strip())
    #     rows[idx] = str
    #
    # print(rows)















