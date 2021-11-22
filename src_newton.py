import numpy as np
from sympy import symbols, Matrix

class CalculationCase:
    def __init__(self, equation_system, start_value, max_iterations, tolerance, name=None, manual_jacobian=None, args=[]):
        self.equation_system = equation_system
        self.start_value = [float(element) for element in start_value]
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.name = name
        self.args = args
        if not manual_jacobian:
            self.jacobian_symbols = np.array(list(self.create_symbols()))
            self.jacobian = self.create_jacobian()
        else:
            self.jacobian_symbols = None
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
        return empty_jacobian

    def filled_jacobian(self, params):
        if not self.jacobian_symbols:
            return np.array(self.jacobian(params), dtype=complex)
        assigned_params = dict(zip(self.jacobian_symbols, params))
        filled_jacobian = self.jacobian.subs(assigned_params).evalf()
        return np.array(filled_jacobian.tolist(), dtype=complex)

    def complex_to_float(self, vector):
        for element in vector:
            yield element if element.imag else element.real

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
