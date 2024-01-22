import tkinter as tk
from tkinter import ttk

def button_func():
    # get the content of the entry
    # print(entry.get())
    entry_text = entry.get()
    # update the label
    # label.configure(text='Some other text')
    label['text'] = entry_text
    entry['state'] = 'disabled'
    # get all the options
    # print(label.configure())
    button2.pack()

def changes_func():

    entry['state'] = 'enabled'
    entry_text = entry.get()
    label.configure(text=entry_text)
    label.configure(foreground='#f54260')


# window
window = tk.Tk()
window.title('Getting and setting widgets')
window.geometry('800x500')


# widgets
label = ttk.Label(master=window, text='Some text')
label.pack()

entry = ttk.Entry(master=window)
entry.pack()

button = ttk.Button(master=window, text='The button', command=button_func)
button.pack()

button2 = ttk.Button(master=window, text='Changes', command=changes_func)



# run
window.mainloop()