
# simple decorators
def greet_decorator(func):
    def wrapper():
        print("Hello")
        func()
        print('World')
    return wrapper


@greet_decorator
def say_name():
    print('name is pythons')


def log_args(func):
    def wrapper(*args, **kwargs):
        print(f"Arguments: {args}, {kwargs}")
        return func(*args, **kwargs)
    return wrapper


if __name__ == '__main__':
    say_name()

    @log_args
    def multiply(a, b):
        return a*b

    print(multiply(3, 4))
