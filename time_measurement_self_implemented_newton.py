import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from src_newton import CalculationCase

# define equation system
def funcIntersectingSpheres(x, *data):
    n = x.shape[0]
    f, J, r = data
    
    for k in range(n):
        f[k] = -r[k] ** 2
        for j in range(n):
            if j == k:
                f[k] += (x[j] - r[k]) ** 2
            else:
                f[k] += x[j] ** 2
    return f

# define manual Jacobian
def JIntersectingSpheres(x, *data):
    n = x.shape[0]
    f, J, r = data
    
    for k in range(n):
        J[k] = 2 * x
        J[k][k] += 2 * (-r[k])
    return J

# initialze calculation case
repeats_for_avg = 5
max_iterations = 1000
init_value = 4
tolerance = 1e-8
raw_n_list = np.arange(2, 10, 2, dtype=int)  # contains x values
n_list = np.tile(raw_n_list, repeats_for_avg)  # repeats x values

start_value_list = [np.ones(element, dtype=float) * init_value for element in n_list]
f_list = [np.zeros(element, dtype=float).tolist() for element in n_list]
J_list = [np.zeros((element, element), dtype=float).tolist() for element in n_list]
r_list = [np.ones(element, dtype=float).tolist() for element in n_list]

# prepare for measurement
time_symbolic_list = []
time_symbolic_list_math = []
time_symbolic_list_numpy = []
time_manual_list = []
for i, n in enumerate(tqdm(n_list)):
    data = (f_list[i], J_list[i], r_list[i])

    calc_case_symbolic_jacobian = CalculationCase(
        funcIntersectingSpheres,
        start_value_list[i],
        max_iterations,
        tolerance,
        args=data
        )
    start_symbolic_jacobian = time.time()
    calc_case_symbolic_jacobian.approximate
    end_symbolic_jacobian = time.time()
    time_symbolic_list.append(end_symbolic_jacobian - start_symbolic_jacobian)

    calc_case_symbolic_jacobian_math = CalculationCase(
        funcIntersectingSpheres,
        start_value_list[i],
        max_iterations,
        tolerance,
        args=data,
        sympy_method='math'
        )
    start_symbolic_jacobian_numpy = time.time()
    calc_case_symbolic_jacobian_math.approximate
    end_symbolic_jacobian_numpy = time.time()
    time_symbolic_list_math.append(end_symbolic_jacobian_numpy - start_symbolic_jacobian_numpy)

    calc_case_symbolic_jacobian_math = CalculationCase(
        funcIntersectingSpheres,
        start_value_list[i],
        max_iterations,
        tolerance,
        args=data,
        sympy_method='numpy'
        )
    start_symbolic_jacobian_cupy = time.time()
    calc_case_symbolic_jacobian_math.approximate
    end_symbolic_jacobian_cupy = time.time()
    time_symbolic_list_numpy.append(end_symbolic_jacobian_cupy - start_symbolic_jacobian_cupy)

    calc_case_manual_jacobian = CalculationCase(
        funcIntersectingSpheres,
        start_value_list[i],
        max_iterations,
        tolerance,
        manual_jacobian=JIntersectingSpheres,
        args=data
        )
    start_manual_jacobian = time.time()
    calc_case_manual_jacobian.approximate
    end_manual_jacobian = time.time()
    time_manual_list.append(end_manual_jacobian - start_manual_jacobian)

# combine results
measurement_list = [time_symbolic_list, time_symbolic_list_math, time_symbolic_list_numpy, time_manual_list]

# reshape lists to 2D array and calculate average values and standard errors
mean_list = []
std_error_list = []
for i, measurement_list_element in enumerate(measurement_list):
    measurement_list[i] = np.array(measurement_list_element).reshape(len(raw_n_list), repeats_for_avg)
    inner_mean_list = []
    inner_std_error_list = []
    for j in range(len(raw_n_list)):
        inner_mean_list.append(np.mean(measurement_list[i][j]))
        inner_std_error_list.append(np.std(measurement_list[i][j])/np.sqrt(repeats_for_avg))
    mean_list.append(inner_mean_list)
    std_error_list.append(inner_std_error_list)

# define labels
label_list = ['symbolic', 'symbolic math', 'symbolic numpy', 'manual']
        
# plot time comparison
plt.figure()
plt.style.use('seaborn-whitegrid')
for i, label in enumerate(label_list):
    plt.errorbar(
        raw_n_list,
        mean_list[i],
        yerr=std_error_list[i],
        fmt='.',
        label=label,
        capsize=3,
    )
plt.xlabel('number of equations')
plt.ylabel('time [s]')
plt.legend()
plt.yscale('log')
plt.show()