import numpy as np
from sympy import symbols, Matrix, Array

def funcIntersectingSpheres(x,*data):
    """
    TODO
    - create lists instead of numpy arrays
    - use yield to fill list
    - compare pure numpy results with mixed results and pure sympy results
    - measure run time for all procedures and scale n 
    """
    n=x.shape[0]
    f,J,r=data
    
    for k in range(n):
        f[k]=-r[k]**2  # subtract constant
        for j in range(n):
            if j==k:
                f[k]+=(x[j]-r[k])**2  # on main diagonal subtract constant and square
            else:
                f[k]+=x[j]**2  # square other values
    return f

def JIntersectingSpheres(x,*data):
    
    n=x.shape[0]
    f,J,r=data
    
    for k in range(n):
        J[k,:]=2*x
        J[k,k]+=2*(-r[k])
    return J

def create_jacobian(x, *data):
    n = x.shape[0]

    jacobian_symbols = np.array(list(create_symbols(n)))
    f = funcIntersectingSpheres(jacobian_symbols, *data)
    sympy_matrix = Matrix(f)
    sympy_params = Matrix([jacobian_symbols])
    empty_jacobian = sympy_matrix.jacobian(sympy_params)
    return empty_jacobian

def eqs(x0, *data):
    
            x_1, x_2 = x0
            return [
                x_1 ** 2 + x_2 ** 2 - 9,
                x_1 ** 2 + (x_2 - 4) ** 2 - 9
            ]
def create_symbols(n):
    for i in range(n):
        yield symbols(f'x{i}')

n=2 # Dimension
f=np.array(np.zeros(n))
J=np.array(np.zeros((n,n)))

r=np.array(np.ones(n))
x0=np.array(np.ones(n)*4)  # Scheint damit gegen [2/n,...,2/n] konvergieren

data=(f,J,r)

a = funcIntersectingSpheres(x0, *data)
print(a)
#b = eqs(x0, *data)
jacobian_a = create_jacobian(x0, *data)
print(jacobian_a)