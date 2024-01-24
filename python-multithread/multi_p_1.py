
from multiprocessing import Process
import os
import time

# https://www.youtube.com/watch?v=cQP8WApzIQQ&list=PLrw6a1wE39_tb2fErI4-WkMbsvGQk9_UB

def square_numbers():
    for i in range(100):
        p = i * i
        time.sleep(0.1)


processes = []
num_process = os.cpu_count()

# create process
for i in range(num_process):
    p = Process(target=square_numbers)
    processes.append(p)


# start
for p in processes:
    p.start()

# join
for p in processes:
    p.join()

print('___end_main___')