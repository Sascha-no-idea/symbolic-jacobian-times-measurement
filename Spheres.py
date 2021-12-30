# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:51:56 2020

@author: Estevez

"""
from functools import lru_cache
import numpy as np

from scipy.optimize import fsolve as fsolve
from scipy.integrate import solve_ivp
from sympy import symbols, Matrix

from timeit import default_timer as timer
import matplotlib.pyplot as plt
from sympy.simplify.fu import fu

from src_newton import CalculationCase

#############################
#   Beispiel fuer fsolve
#############################

# time measurements for symbolic calculation using subs/evalf (naturally slow)
def funcIntersectingSpheres(x, *data):
    
    n = x.shape[0]
    f, J, r, _, _ = data
    
    for k in range(n):
        f[k] = -r[k] ** 2
        for j in range(n):
            if j == k:
                f[k] += (x[j] - r[k]) ** 2
            else:
                f[k] += x[j] ** 2
    return f

def JIntersectingSpheres(x, *data):
    
    n = x.shape[0]
    f, J, r, _, _ = data
    
    for k in range(n):
        J[k,:] = 2 * x
        J[k,k] += 2 * (-r[k])
    return J

def create_symbols(n):
    jacobian_symbols = []
    for i in range(n):
        jacobian_symbols.append(symbols(f'x_{i}'))
    return jacobian_symbols

def create_jacobian(x, *data):
    _, _, _, jacobian_symbols, _ = data
    f = funcIntersectingSpheres(x, *data)
    sympy_matrix = Matrix(f)
    sympy_params = Matrix([jacobian_symbols])
    empty_jacobian = sympy_matrix.jacobian(sympy_params)
    return empty_jacobian

def evaluate_jacobian(x, *args):
    _, _, _, jacobian_symbols, empty_jacobian = args
    assigned_params = dict(zip(jacobian_symbols, x))
    filled_jacobian = empty_jacobian.subs(assigned_params)  # highest impact on performance
    a = timer()
    filled_jacobian = filled_jacobian.evalf()
    b = timer()
    print(f'time = {b - a}')
    return np.array(filled_jacobian.tolist(), dtype=float)

#Initialisierung
n=20 # Dimension
f=np.zeros(n).tolist()
J=np.zeros((n,n))

# Hinweis: Sind alle r gleich, so gibt es die Loesungen
# [0,....,0] und [2r/n, ..., 2r/n]


r=np.ones(n).tolist()
x0=np.ones(n)*4  # Scheint damit gegen [2/n,...,2/n] konvergieren
#x0=-r # sollte gegen 0 konvergieren, es konvergieren beide aber leider nicht fuer z.B. n=50



# initialize symbolic calculation of jacobian
jacobian_symbols = create_symbols(n)
data=(f,J,r, jacobian_symbols, None)
empty_jacobian = create_jacobian(x0, *data)
data=(f,J,r, jacobian_symbols, empty_jacobian)

#################################################

# solve with approximated jacobian
print(f'approximated jacobian')
start = timer()
sol=fsolve(funcIntersectingSpheres,x0,full_output=True,args=data) # Loese nichtlineares Gleichungssystem
end = timer()
print("Time sol=",end - start)
print("sol=",sol) # Achtung: Konvergiert nicht immer (?), z.B. n=50

# solve with manual jacobian
print(f'manual jacobian') 
start = timer()
solJ=fsolve(funcIntersectingSpheres,x0,fprime=JIntersectingSpheres,full_output=True,args=data) # Loese nichtlineares Gleichungssystem
end = timer()
print("Time solJ=",end - start)
print("solJ=",solJ)

# solve with manual jacobian via symbolic math
print(f'manual jacobian via symbolic math')
start = timer()
sol_symbolic = fsolve(funcIntersectingSpheres, x0 , fprime=evaluate_jacobian, full_output=True,args=data)
end = timer()
print("Time sol_symbolic=",end - start)
print("sol_symbolic=",sol_symbolic)

#start = timer()
#my_sol = CalculationCase(funcIntersectingSpheres, x0, 1000, 1e-8, args=data).approximate()[0]
#end = timer()
#print("Time my_sol=",end - start)
#
#print("my_sol=", my_sol)

################################################
# Beispiel mit solve_ivp
################################################
# Laeuft auf die Stationaere Loesung, die fsolve geliefert hat
""" 
def funcIntersectingSpheresODE(t,x,*data):
    f=-funcIntersectingSpheres(x,*data) # Negatives Vorzeichen, aus Stabilitätsgründen
    return f # 


def JIntersectingSpheresODE(t,x,*data):
    J=-JIntersectingSpheres(x,*data) # Negatives Vorzeichen, s. oben
    return J # 

ode_method='Radau'
#ode_method='BDF'
tspan=[0.0,4.0]

start = timer()
sol = solve_ivp(funcIntersectingSpheresODE, tspan, x0,args=data,method=ode_method)
plt.plot(sol.t,sol.y[0,:])
end = timer()
print("Time ODE sol=",end - start)
print('  nfev:',sol.nfev)
print('  njev:',sol.njev)
#print('  nlu: ',sol.nlu)
print('  len_t:',len(sol.t))

start = timer()
solJ = solve_ivp(funcIntersectingSpheresODE, tspan, x0,jac=JIntersectingSpheresODE,args=data,method=ode_method)
plt.plot(solJ.t,solJ.y[0,:])
end = timer()
print("Time ODE solJ=",end - start)
print('  nfev:',sol.nfev)
print('  njev:',sol.njev)
#print('  nlu: ',sol.nlu)
print('  len_t:',len(sol.t))
 """