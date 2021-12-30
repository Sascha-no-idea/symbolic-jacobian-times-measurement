#from _typeshed import NoneType
import string
import numpy as np
from sympy import symbols, Matrix, lambdify

class CalculationCase:
    def __init__(
        self,
        equation_system,
        start_value,
        max_iterations,
        tolerance,
        name=None,
        manual_jacobian=None,
        args=[],
        sympy_method='sympy',
        ):
        self.equation_system = equation_system
        self.start_value = np.array([float(element) for element in start_value], dtype=complex)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.name = name
        self.args = args
        self.sympy_method = sympy_method
        if not manual_jacobian:
            self.jacobian_symbols = np.array(list(self.create_symbols()))
            self.jacobian = self.create_jacobian()
        else:
            self.jacobian_symbols = ''
            self.jacobian = manual_jacobian

    @property
    def number_of_params(self):
        return len(self.start_value)
    
    def create_symbols(self):
        for i in range(self.number_of_params):
            yield symbols(f'x_{i}')

    def create_jacobian(self):
        f = self.equation_system(self.jacobian_symbols, *self.args)
        sympy_matrix = Matrix(f)
        sympy_params = Matrix([self.jacobian_symbols])
        empty_jacobian = sympy_matrix.jacobian(sympy_params)
        if self.sympy_method == 'sympy':
            return empty_jacobian
        elif self.sympy_method == 'math':
            empty_jacobian_lambdifed = lambdify(self.jacobian_symbols, empty_jacobian)
            return empty_jacobian_lambdifed
        elif self.sympy_method == 'numpy':
            empty_jacobian_lambdifed = lambdify(self.jacobian_symbols, empty_jacobian, self.sympy_method)
            return empty_jacobian_lambdifed

    def filled_jacobian(self, params):
        if type(self.jacobian_symbols) == str:  # in case of manual jacobian
            return np.array(self.jacobian(params, *self.args), dtype=complex)
        if self.sympy_method == 'sympy':
            assigned_params = dict(zip(self.jacobian_symbols, params))
            filled_jacobian = self.jacobian.subs(assigned_params).evalf()
            return np.array(filled_jacobian.tolist(), dtype=complex)
        else:
            return self.jacobian(*params)

    def complex_to_float(self, vector):
        for element in vector:
            yield element if element.imag else element.real

    @property
    def approximate(self):
        x_vector = self.start_value
        current_iteration = 0
        delta_vector = np.inf
        while current_iteration <= self.max_iterations and np.linalg.norm(delta_vector) > self.tolerance:
            y_vector = self.equation_system(x_vector, *self.args)
            y_vector = np.array(y_vector, dtype=complex)
            assert len(x_vector) == len(y_vector)

            filled_jacobi_matrix = self.filled_jacobian(x_vector)
            if np.linalg.matrix_rank(filled_jacobi_matrix) < len(x_vector):
                raise ValueError(f'Calculation failed due to singular jacobian matrix!')
            
            delta_vector = np.linalg.solve(filled_jacobi_matrix, -y_vector)
            
            # check if still convergent
            # NOTE delta vector progress is only stable after second iteration
            if current_iteration > 1 and not all(np.less_equal(np.absolute(delta_vector), np.absolute(old_delta_vector))):
                raise ValueError('Newton\'s method is not converging!')
            old_delta_vector = delta_vector
            x_vector += delta_vector
    
            current_iteration += 1

        # beautify non-complex results
        x_vector = list(self.complex_to_float(x_vector))
        if current_iteration <= self.max_iterations:
            # approximation successful
            return x_vector, current_iteration + 1
        raise ValueError(
            f'Calculation exceeded the limit of {self.max_iterations} iterations! The latest result is {x_vector}\n'
            f'residuum: {np.linalg.norm(y_vector)}\n'
            f'delta: {np.linalg.norm(delta_vector)}')
