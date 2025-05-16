# class based
from contextlib import contextmanager
import os


class Open_File:

    def __init__(self, filename: str, mode: str):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, traceback):
        self.file.close()


# @contextmanager
# def get_current_directory():

@contextmanager
def open_file(file, mode):
    try:
        f = open(file, mode)
        yield f
    finally:
        f.close()


@contextmanager
def change_dir(destination: str):
    try:
        cwd = os.getcwd()
        os.chdir(destination)
        yield
    finally:
        os.chdir(cwd)


with change_dir('python-basics'):
    print(os.listdir())

# with Open_File(f"{os.getcwd()}/python-basics"+'/sample.txt', 'w') as file:
#     file.write('testing')
# print(file.closed)
