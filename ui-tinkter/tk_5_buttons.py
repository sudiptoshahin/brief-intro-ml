import tkinter as tk
from tkinter import ttk


def button_func():
    pass

# window
window = tk.Tk()
window.title('buttons')
window.geometry('500x300')


# buttons
button_string = tk.StringVar(value='Button with string var')
button = ttk.Button(master=window, text='Simple button', command=lambda: print('a basic button'), textvariable=button_string)
button.pack()

# checked button
# check_var = tk.StringVar()
check_var = tk.IntVar()
check_button = ttk.Checkbutton(
    master=window,
    text='Checkbox 1',
    command=lambda: print(check_var.get()),
    variable=check_var
)
check_button.pack()
# run
window.mainloop()