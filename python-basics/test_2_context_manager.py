

def test_var_args(f_arg, *args):
    print(f'normal arguments: {f_arg}')

    for idx, arg in enumerate(args):
        print(f'Arg with *args[{idx}]: {arg}')


test_var_args('test1', 'arg1', 'arg2', 'arg3')


# ______**kwargs___________
"""
    **kwargs allows you to pass keyworded variable length of arguments to a
    function.
    You should use **kwargs if you want to handle named arguments
    in a function.
"""


def using_kwargs(**kwargs):
    # for key, value in kwargs.items():
    #     print(f"{key} = {value}")
    print(kwargs)


using_kwargs(first_name="Sudipto", last_name="Shahin", id="#112233")
