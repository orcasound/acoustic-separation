from tkinter import *

root= Tk()
root.title("Calculator")

e = Entry(root, width=35, borderwidth=5)
e.grid(row=0,column=0, columnspan=3, padx=10, pady=10)

def button_click(number):
    #e.delete(0, END) #delete text in the box
    current = e.get() #the input is a string
    e.delete(0, END)  # delete text in the box
    e.insert(0, str(current)+str(number))

def button_clear():
    e.delete(0, END)

def button_add():
    first=e.get()
    global f_num #so f_num is available outside the function to button_equal
    global math
    math = "addition"
    f_num = int(first)
    e.delete(0, END)

def button_equal():
    second = e.get()
    e.delete(0, END)
    if math == "addition":
        e.insert(0, f_num+int(second))
    if math == "subtraction":
        e.insert(0, f_num-int(second))
    if math == "multiplication":
        e.insert(0, f_num*int(second))
    if math == "division":
        e.insert(0, f_num/int(second))

def button_subtract():
    first = e.get()
    global f_num  # so f_num is available outside the function to button_equal
    global math
    math = "subtraction"
    f_num = int(first)
    e.delete(0, END)

def button_multiply():
    first = e.get()
    global f_num  # so f_num is available outside the function to button_equal
    global math
    math = "multiplication"
    f_num = int(first)
    e.delete(0, END)

def button_divide():
    first = e.get()
    global f_num  # so f_num is available outside the function to button_equal
    global math
    math = "division"
    f_num = int(first)
    e.delete(0, END)

button1 = Button(root, text="1", padx=40, pady=20, command=lambda: button_click(1))
button2 = Button(root, text="2", padx=40, pady=20, command=lambda: button_click(2))
button3 = Button(root, text="3", padx=40, pady=20, command=lambda: button_click(3))
button4 = Button(root, text="4", padx=40, pady=20, command=lambda: button_click(4))
button5 = Button(root, text="5", padx=40, pady=20, command=lambda: button_click(5))
button6 = Button(root, text="6", padx=40, pady=20, command=lambda: button_click(6))
button7 = Button(root, text="7", padx=40, pady=20, command=lambda: button_click(7))
button8 = Button(root, text="8", padx=40, pady=20, command=lambda: button_click(8))
button9 = Button(root, text="9", padx=40, pady=20, command=lambda: button_click(9))
button0 = Button(root, text="0", padx=40, pady=20, command=lambda: button_click(0))
button_plus = Button(root, text="+", padx=39, pady=20, command=button_add)
button_equals = Button(root, text="=", padx=91, pady=20, command=button_equal)
button_clear = Button(root, text="clear", padx=79, pady=20, command=button_clear)
button_minus = Button(root, text="-", padx=41, pady=20, command=button_subtract)
button_per = Button(root, text="*", padx=40, pady=20, command=button_multiply)
button_divide = Button(root, text="/", padx=41, pady=20, command=button_divide)


#Put buttons on screen
button1.grid(row=3, column=0)
button2.grid(row=3, column=1)
button3.grid(row=3, column=2)
button4.grid(row=2, column=0)
button5.grid(row=2, column=1)
button6.grid(row=2, column=2)
button7.grid(row=1, column=0)# column 0 for input
button8.grid(row=1, column=1)
button9.grid(row=1, column=2)
button0.grid(row=4, column=0)
button_clear.grid(row=4, column=1, columnspan=2)
button_plus.grid(row=5, column=0)
button_equals.grid(row=5, column=1, columnspan=2)
button_minus.grid(row=6, column=0)
button_per.grid(row=6, column=1)
button_divide.grid(row=6, column=2)

root.mainloop()
