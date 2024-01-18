import tkinter as tk
# from tkinter import ttk
import ttkbootstrap as ttk
def convert():
    mile_input = entry_int.get()
    km_output = mile_input * 1.61

    output_string.set(km_output)
# window
window = ttk.Window(themename='darkly')
window.title('Demo')
window.geometry('500x250')

# title
title_label = ttk.Label(master=window, text="Miles to Kilometer", font='Roboto 20 bold')
title_label.pack()

# input field
input_frame = ttk.Frame(master=window)

entry_int = tk.IntVar()

entry = ttk.Entry(master=input_frame, textvariable=entry_int)
button = ttk.Button(master=input_frame, text='Convert', command=convert)
entry.pack(side='left', padx=10, pady=20)
button.pack(side='left')

input_frame.pack(pady=20)

# output
output_string = tk.StringVar()
output_label = ttk.Label(
    master=window,
    text='Output',
    font='Calbri 20',
    textvariable=output_string
)
output_label.pack(pady=5)

# run
window.mainloop()


# 21:29