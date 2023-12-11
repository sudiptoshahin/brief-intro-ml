import pstats
import time
import cProfile
import subprocess
import sys
import os
import tuna


def add(x, y):
    result = 0
    result += x
    result += y
    return result


def fact(n):
    result = 1
    for i in range(1, n+1):
        result *= 1
    return result


def do_stuff():
    result = []
    for x in range(100000):
        result.append(x ** 2)
    return result


def waste_time():
    time.sleep(5)
    print('Hello-world')


if __name__ == '__main__':
    r1 = add(100, 50000)
    print(r1)

    r2 = fact(85)
    print(r2)

    with cProfile.Profile() as profile:
        r3 = do_stuff()
        print(r3)
        # waste_time()
    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.CUMULATIVE)
    output = results.print_stats()
    # results.dump_stats('results.prof')

    # test-1
    # output = subprocess.check_output('python -m snakeviz results.prof --server', shell=True)
    # output = output.decode('utf-8')

    # test-2
    # output = subprocess.run(['python3', '-m', 'tuna', 'results.prof', '--server'], stdout=subprocess.PIPE)
    # output = subprocess.run(['ls', '-l'], stdout=subprocess.PIPE)
    # output = subprocess.getoutput('ls -l')

    try:
        output = subprocess.getoutput('python3 -m tuna results.prof')
    except KeyboardInterrupt:
        print('---close---')
    #
    # print(output.stdout)
    # user_paths = os.environ.get('PYTHONPATH', '').split(os.pathsep)
    # print(f'\n\n{output}')
