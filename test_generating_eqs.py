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

def create_jacobian(x, jacobian_symbols, *data):
    n = x.shape[0]
    f = funcIntersectingSpheres(jacobian_symbols, *data)
    sympy_matrix = Matrix(f)
    sympy_params = Matrix([jacobian_symbols])
    empty_jacobian = sympy_matrix.jacobian(sympy_params)
    return empty_jacobian

def create_symbols(n):
    for i in range(n):
        yield symbols(f'x{i}')

def filled_jacobian(x, jacobian_symbols, empty_jacobian):
    n = x.shape[0]
    assigned_params = dict(zip(jacobian_symbols, x))
    filled_jacobian = empty_jacobian.subs(assigned_params).evalf()
    return np.array(filled_jacobian.tolist())

n=2 # Dimension
f=np.array(np.zeros(n)).tolist()
J=np.array(np.zeros((n,n)))

r=np.array(np.ones(n)).tolist()
x0=np.array(np.ones(n)*4)  # Scheint damit gegen [2/n,...,2/n] konvergieren

data=(f,J,r)

a = funcIntersectingSpheres(x0, *data)
print(a)
#b = eqs(x0, *data)
jacobian_symbols = np.array(list(create_symbols(n)))
print(jacobian_symbols)
empty_jacobian = create_jacobian(x0, jacobian_symbols, *data)
print(empty_jacobian)
filled_jacobian = filled_jacobian(x0, jacobian_symbols, empty_jacobian)
print(filled_jacobian)

evaluated_jacobian = JIntersectingSpheres(x0, *data)
print(evaluated_jacobian)
