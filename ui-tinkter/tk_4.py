import tkinter as tk
from tkinter import ttk


def button_func():
    print(string_var.get())
    string_var.set('button pressed')


# window
window = tk.Tk()
window.title('Tkinter vars')
window.geometry('500x200')

## tkinkter variables
string_var = tk.StringVar(value='Start value')


# widgets
label = ttk.Label(master=window, text='label', textvariable=string_var)
label.pack()

entry = ttk.Entry(master=window, textvariable=string_var)
entry.pack()

entry2 = ttk.Entry(master=window, textvariable=string_var)
entry2.pack()

button = ttk.Button(master=window, text='Button', command=button_func)
button.pack()



# run
window.mainloop()