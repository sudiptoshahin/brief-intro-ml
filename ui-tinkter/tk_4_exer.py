import tkinter as tk
from tkinter import ttk
# import ttkbootstrap as ttk


def show_res():
    user_name = name_text.get()
    user_age = age_text.get()

    info_label.configure(text=f'Name: {user_name} \nAge: {user_age}')


# window
window = tk.Tk()
window.title('Info')
window.geometry('500x300')

# variables
name_text = tk.StringVar(value='Please enter your name')
age_text = tk.IntVar(value='Enter age')

# widgets
name_entry = ttk.Entry(master=window, textvariable=name_text)
age_entry = ttk.Entry(master=window, textvariable=age_text)

label = ttk.Label(master=window, text='Please enter your information')
label.pack()

submit_button = ttk.Button(master=window, text='Submit', command=show_res)

info_label = ttk.Label(master=window, text='Info will show here...')

name_entry.pack(padx=5, pady=5)
age_entry.pack(padx=5, pady=5)
submit_button.pack()
info_label.pack(pady=20)
# run
window.mainloop()