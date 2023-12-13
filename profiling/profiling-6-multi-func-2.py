import time
from line_profiler import LineProfiler
import re
from io import StringIO
import pandas as pd
import csv


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


# {
#     0: '    20                                           def func_3():',
#     1: '    21         1 3501153821.0    4e+09    100.0      time.sleep(3.5)',
#     2: '    22         1       8669.0   8669.0      0.0      for i in range(100):',
#     3: '    23         1        792.0    792.0      0.0          res = 0',
#     4: '    24         1       2540.0   2540.0      0.0          res = res + (i ** 2)',
#     5: '    25         1       1341.0   1341.0      0.0          return res'
# }

def multi_df(functions: list):
    new_df_list = []
    for idx, func in enumerate(functions):
        result_obj = getLpObj(func)  # dict length = 6
        new_df_list.append(result_obj)

    new_df = pd.concat(new_df_list)
    new_df.to_csv('lp_test_1.csv')

def getLpObj(func):
    stream = StringIO()

    lineprofiler = LineProfiler()
    line_wrapper = lineprofiler(func)
    line_wrapper()
    lineprofiler.print_stats(stream=stream)

    output = stream.getvalue()
    output = re.split(r', \n', output)
    output = ' '.join(output).strip().split('\n')

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


if __name__ == '__main__':
    multi_df([func_1, func_2, func_3])
