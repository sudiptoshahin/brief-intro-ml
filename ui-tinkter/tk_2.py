import tkinter as tk
import ttkbootstrap as ttk


def button_function():
    print('pressed')

def hello_button():
    print('hello')

# create window
window = tk.Tk()
window.title('Window and widgets')
window.geometry('800x500')

# ttk labels
label = ttk.Label(master=window, text='This is a test')
label.pack()

# create widgets
# multiline text input
text = tk.Text(master=window)
text.pack()

# button/entry
entry = ttk.Entry(master=window)
entry.pack(pady=20)

# ttk/button
button1 = ttk.Button(master=window, text='Button', command=button_function)
button1.pack(side='left', padx=10)

# ttk/button
button2 = ttk.Button(master=window, text='Hello', command=hello_button)
button2.pack(side='left', padx=10)


# run
# 1. updates the GUI
# 2. check the events
window.mainloop()
