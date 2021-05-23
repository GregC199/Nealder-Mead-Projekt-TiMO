'''from PythonApplication3 import PythonApplication3
import copy
import math
import random 
import sympy as sympy
import numpy as np
'''
'''
import PySimpleGUI as sg

# Define the window's contents
layout = [[sg.Text("What's your name?")],
          [sg.Input(key='-INPUT-')],
          [sg.Text(size=(40,1), key='-OUTPUT-')],
          [sg.Button('Ok'), sg.Button('Quit')]]
# Create the window
window = sg.Window('Algorytm Neldera-Meada', layout)

# Display and interact with the Window using an Event Loop
while True:
    event, values = window.read()
    # See if user wants to quit or window was closed
    if event == sg.WINDOW_CLOSED or event == 'Quit':
        break
    # Output a message to the window
    window['-OUTPUT-'].update('Hello ' + values['-INPUT-'] + "! Thanks for trying PySimpleGUI")
'''

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
import pandas as pd

f_str='x**2+y*x+0.5*y**2-x-y'

def f(x, y):
    return eval(f_str)#np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

x = np.linspace(-8, 8, 50)
y = np.linspace(-8, 8, 40)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

arrPunktowX = [1, 3]
arrPunktowY = [1, 3]

df = pd.DataFrame()
df['x'] = arrPunktowX
df['y'] = arrPunktowY
df.head()

plt.plot(df['x'], df['y'])
plt.plot(df['x'], df['y'],'o')
plt.contourf(X, Y, Z, 20, cmap='jet')
plt.colorbar()
plt.show()