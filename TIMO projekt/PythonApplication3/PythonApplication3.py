import copy
import math
import random 
from matplotlib import pyplot
import sympy as sympy
import numpy as np

'''
Pure Python/Numpy implementation of the Nelder-Mead algorithm.
Reference: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
'''
f_str = ''
l_x = 0
argm = 0
simplexes = []
best_points = []
epsilon = 1e-3

def f(x):
    x1,x2,x3,x4,x5 = sympy.symbols('x1 x2 x3 x4 x5')
    l_x = len(x)
    if l_x  == 2:
        x1 = x[0]
        x2 = x[1]
    if l_x == 3:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
    if l_x  == 4:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
    if l_x  == 5:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
    print(f_str)
    return eval(f_str)


def nelder_mead(f, x_start,
                step=0.1, no_improve_thr=epsilon,
                no_improv_break=10, max_iter=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):

    # init
    global best_points
    global simplexes
    dim = len(x_start)
    prev_best = f(x_start)
    no_improv = 0
    res = [[x_start, prev_best]]
    bestlist = []

    for i in range(dim):
        x = copy.copy(x_start) #tworzenie punktów początkowych 1. kopia startowego
        x[i] = x[i] + step # 2. stworzenie nowego punktu po przesunieciu kopii w jednym wymiarze o step
        score = f(x) #wyliczenie wartosci funkcji w nowym punkcie
        res.append([x, score]) #dodanie nowego zestawu argumentow i wartosci funkcji do listy

    # simplex iter
    iters = 0
    while 1:
        print(iters)
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]
        bestlist.append(res[0][1]) 

        simplexes.append(res[0])
        simplexes.append(res[1])
        simplexes.append(res[2])

        print('test')
        print(res[0][0][0],res[0][0][1],res[0][1])
        print(*simplexes,sep='\n')
        # break after max_iter
        if max_iter and iters >= max_iter:
            print('best val',res[0])
            best_points = bestlist
            return bestlist
            #return res[0]

        iters += 1

        # break after no_improv_break iterations with no improvement
        #print '...best so far:', best

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            print('best val',res[0])
            best_points = bestlist
            return bestlist

        # centroid
        x0 = [0.] * (dim)
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        # reflection
        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = f(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            escore = f(xe)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + rho*(x0 - res[-1][0])
        cscore = f(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            score = f(redx)
            nres.append([redx, score])
        res = nres


def algorytm(start):

    result = nelder_mead(f, start)
    #print(result)
    print(*result, sep='\n')
    print('dupa')
    print(*simplexes,sep='\n')

def start_rand(arguments): #przyjmuje ilosc argumentów i tworzy wektor startowy
    while itr < arguments:
        print('x',itr+1)
        print('a')
        rnd_a = int(input()) #zakresy do random.uniform musi zassysać skądś po wciśnięciu przycisku na GUI
        print('b')
        rnd_b = int(input())
        start[itr] = random.uniform(-rnd_a, rnd_b) #nie wiem co z tym whilem calym
        itr = itr +1
    return start

def start_eval(temp_str):

    #print('Input your expression:')
    #f_str = input()
    str_arg = 'x0'
    itr = 0
    arguments = 0
    global f_str
    f_str = temp_str
    print('taki chuj',f_str)
    f_str = f_str.replace('^','**')
    f_str = f_str.replace('pi','math.pi')
    f_str = f_str.replace('sin','math.sin')
    print('taki chuj po poprawkach',f_str)
    while itr < 5:
        old_str = str(itr)
        new_str = str(itr+1)
        str_arg = str_arg.replace(old_str,new_str)
        print('argument to check',str_arg)
        if str_arg in f_str:
            arguments = arguments + 1
            print('exists')
        else:
            print("doesn't exists")
            break
        itr = itr + 1

    #start = np.empty(arguments)
    #itr = 0
    print('Function has',arguments,'arguments')
    return arguments
    
    #start = start_rand(arguments)

    #algorytm(f_str,start) #wywolanie czesci z algorytmem - podanie stringa funkcji i argumentow startowych
 
