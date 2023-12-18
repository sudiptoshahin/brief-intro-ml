import time
import subprocess
from profiling import Profiling

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


################time_profiling################

if __name__ == '__main__':
    profiling = Profiling()
    # time_profiling.cProfilerEnable()
    print(func_1())
    print(func_2())
    print(func_3())
    print(func_4())
    print(func_5())

    @profiling.do_line_profile(follow=[py_loop])
    def test():
        func_1()
        func_3()
        func_2()
    # time_profiling.cProfilerDisable()
    # time_profiling.getCrofilerStat()

    # time_profiling.setUniFunctionLP(func_1)
    # time_profiling.get_single_line_profiler_stat()

    profiling.setMultiFunctionLP([func_1, func_2, func_3])
    profiling.multi_df_csv([func_1, func_2, func_3])
    # print(time_profiling.funcs)
    # time_profiling.getSingleLineProfilerStat()
    # print(time_profiling.multilineprofiler_df())
    # time_profiling.multilineprofiler_df()
