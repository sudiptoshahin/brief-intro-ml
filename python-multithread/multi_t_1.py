
from threading import Thread
import os
import time

# https://www.youtube.com/watch?v=cQP8WApzIQQ&list=PLrw6a1wE39_tb2fErI4-WkMbsvGQk9_UB

def square_numbers():
    for i in range(100):
        p = i * i
        time.sleep(0.1)


threads = []
num_threads = 10

# create process
for i in range(num_threads):
    t = Thread(target=square_numbers)
    threads.append(t)


# start
for t in threads:
    t.start()

# join
for t in threads:
    t.join()

print('___end_main___')