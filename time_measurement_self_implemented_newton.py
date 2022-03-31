import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import pickle
from scipy.optimize import curve_fit

from src_newton import CalculationCase

def calculate():
    # ask user for parameters
    repeats_for_avg = str(input('How many repeats for averaging? '))
    if repeats_for_avg == '':
        repeats_for_avg = 5
    else:
        repeats_for_avg = int(repeats_for_avg)
    interval_start = str(input('Interval start? '))
    if interval_start == '':
        interval_start = 2
    else:
        interval_start = int(interval_start)
    interval_end = str(input('Interval end? '))
    if interval_end == '':
        interval_end = 10
    else:
        interval_end = int(interval_end)
    interval_step = str(input('Interval step? '))
    if interval_step == '':
        interval_step = 2
    else:
        interval_step = int(interval_step)

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
    max_iterations = 1000
    init_value = 4
    tolerance = 1e-8
    raw_n_list = np.arange(interval_start, interval_end, interval_step, dtype=int)  # contains x values
    n_list = np.tile(raw_n_list, repeats_for_avg)  # repeats x values

    start_value_list = [np.ones(element, dtype=float) * init_value for element in n_list]
    f_list = [np.zeros(element, dtype=float).tolist() for element in n_list]
    J_list = [np.zeros((element, element), dtype=float).tolist() for element in n_list]
    r_list = [np.ones(element, dtype=float).tolist() for element in n_list]

    f_array = [np.zeros(element, dtype=float) for element in n_list]
    J_array = [np.zeros((element, element), dtype=float) for element in n_list]
    r_array = [np.ones(element, dtype=float) for element in n_list]

    # prepare for measurement
    time_symbolic_list = []
    time_symbolic_list_math = []
    time_symbolic_list_numpy = []
    time_manual_list = []
    time_manual_list_array = []

    # start measurement
    print('Calculating...')
    for i, n in enumerate(tqdm(n_list)):
        data = (f_list[i], J_list[i], r_list[i])
        array_data = (f_array[i], J_array[i], r_array[i])

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
        start_symbolic_jacobian_math = time.time()
        calc_case_symbolic_jacobian_math.approximate
        end_symbolic_jacobian_math = time.time()
        time_symbolic_list_math.append(end_symbolic_jacobian_math - start_symbolic_jacobian_math)

        calc_case_symbolic_jacobian_numpy = CalculationCase(
            funcIntersectingSpheres,
            start_value_list[i],
            max_iterations,
            tolerance,
            args=data,
            sympy_method='numpy'
            )
        start_symbolic_jacobian_numpy = time.time()
        calc_case_symbolic_jacobian_math.approximate
        end_symbolic_jacobian_numpy = time.time()
        time_symbolic_list_numpy.append(end_symbolic_jacobian_numpy - start_symbolic_jacobian_numpy)

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

        calc_case_manual_jacobian_array = CalculationCase(
            funcIntersectingSpheres,
            start_value_list[i],
            max_iterations,
            tolerance,
            manual_jacobian=JIntersectingSpheres,
            args=array_data
            )
        start_manual_jacobian_array = time.time()
        calc_case_manual_jacobian_array.approximate
        end_manual_jacobian_array = time.time()
        time_manual_list_array.append(end_manual_jacobian_array - start_manual_jacobian_array)

    # combine results
    measurement_list = [
        time_symbolic_list,
        time_symbolic_list_math,
        time_symbolic_list_numpy,
        time_manual_list,
        time_manual_list_array
        ]

    # save results
    print('Saving results...')
    data = (measurement_list, n_list, raw_n_list, repeats_for_avg)
    with open('data/data.pkl', 'wb') as f:
        pickle.dump(data, f)


def plot_results(save=False, load_file=False, show=True):
    # load data
    if load_file:
        import tkinter
        from tkinter import filedialog

        tkinter.Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

        print('Waiting for user selection of file...')
        folder_path = filedialog.askopenfilename(
            title='Select file',
            filetypes=[('pickle files', '*.pkl')],
            initialdir='./data',
            initialfile='data.pkl'
            )
        with open(folder_path, 'rb') as f:
            data = pickle.load(f)
    else:
        with open('data/data.pkl', 'rb') as f:
            data = pickle.load(f)
    
    measurement_list, n_list, raw_n_list, repeats_for_avg = data

    # reshape lists to 2D array and calculate average values and standard errors
    mean_list = []
    std_error_list = []
    for i, measurement_list_element in enumerate(measurement_list):
        measurement_list[i] = np.array(measurement_list_element).reshape((repeats_for_avg, len(raw_n_list))).transpose()
        inner_mean_list = []
        inner_std_error_list = []
        for j in range(len(raw_n_list)):
            inner_mean_list.append(np.mean(measurement_list[i][j]))
            inner_std_error_list.append(np.std(measurement_list[i][j])/np.sqrt(repeats_for_avg))
        mean_list.append(inner_mean_list)
        std_error_list.append(inner_std_error_list)

    # use scipy to fit curve to data
    # define functions for fitting
    def func_exponential(x, a, b):
        return a * np.exp(b * x)

    def func_quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c

    def func_cubic(x, a, b, c, d):
        return a * x ** 3 + b * x ** 2 + c * x + d

    # start curve fitting
    fit_list = []
    for i, element in enumerate(mean_list):
        inner_fit_list = []
        inner_fit_list.append(curve_fit(func_exponential, raw_n_list, element, p0=[1, 1])[0])
        inner_fit_list.append(curve_fit(func_quadratic, raw_n_list, element, p0=[1, 1, 1])[0])
        inner_fit_list.append(curve_fit(func_cubic, raw_n_list, element, p0=[1, 1, 1, 1])[0])
        fit_list.append(inner_fit_list)

    # prepare fitted curves for plotting
    fitted_list = []
    for i in range(len(fit_list)):
        inner_fitted_list = []
        inner_fitted_list.append(func_exponential(raw_n_list, *fit_list[i][0]))
        inner_fitted_list.append(func_quadratic(raw_n_list, *fit_list[i][1]))
        inner_fitted_list.append(func_cubic(raw_n_list, *fit_list[i][2]))
        fitted_list.append(inner_fitted_list)

    # define labels
    label_list = ['symbolic', 'symbolic math', 'symbolic numpy', 'manual', 'manual array']
    color_list = ['blue', 'green', 'red', 'black', 'orange']
    show and print('Plotting results...')

    # plot time comparison
    plt.figure('Time Comparison')
    plt.style.use('seaborn-whitegrid')
    for i, label in enumerate(label_list):
        plt.errorbar(
            raw_n_list,
            mean_list[i],
            yerr=std_error_list[i],
            fmt='.',
            label=label,
            capsize=3,
            color=color_list[i]
        )

    plt.xlabel('number of equations')
    plt.ylabel('time [s]')
    plt.legend()
    plt.yscale('log')
    save and plt.savefig('docs/graphics/' + datetime.today().strftime('%Y-%m-%dT%H-%M-%S') + '_time_comparison.svg')

    plt.figure('Exponential fit')
    plt.style.use('seaborn-whitegrid')
    for i, label in enumerate(label_list):
        plt.errorbar(
            raw_n_list,
            mean_list[i],
            yerr=std_error_list[i],
            fmt='.',
            label=label,
            capsize=3,
            color=color_list[i]
        )
        plt.plot(raw_n_list, fitted_list[i][0], label='Exponential fit '+label, color=color_list[i])

    plt.xlabel('number of equations')
    plt.ylabel('time [s]')
    plt.legend()
    plt.yscale('log')
    save and plt.savefig('docs/graphics/' + datetime.today().strftime('%Y-%m-%dT%H-%M-%S') + '_exponential_fit.svg')

    plt.figure('Quadratic fit')
    plt.style.use('seaborn-whitegrid')
    for i, label in enumerate(label_list):
        plt.errorbar(
            raw_n_list,
            mean_list[i],
            yerr=std_error_list[i],
            fmt='.',
            label=label,
            capsize=3,
            color=color_list[i]
        )
        plt.plot(raw_n_list, fitted_list[i][1], label='Quadratic fit '+label, color=color_list[i])

    plt.xlabel('number of equations')
    plt.ylabel('time [s]')
    plt.legend()
    plt.yscale('log')
    save and plt.savefig('graphics/' + datetime.today().strftime('%Y-%m-%dT%H-%M-%S') + '_quadratic_fit.svg')

    plt.figure('Cubic fit')
    plt.style.use('seaborn-whitegrid')
    for i, label in enumerate(label_list):
        plt.errorbar(
            raw_n_list,
            mean_list[i],
            yerr=std_error_list[i],
            fmt='.',
            label=label,
            capsize=3,
            color=color_list[i]
        )
        plt.plot(raw_n_list, fitted_list[i][2], label='Cubic fit '+label, color=color_list[i])

    plt.xlabel('number of equations')
    plt.ylabel('time [s]')
    plt.legend()
    plt.yscale('log')
    save and plt.savefig('graphics/' + datetime.today().strftime('%Y-%m-%dT%H-%M-%S') + '_cubic_fit.svg')

    save and print('Saving graphics to folder "graphics"')
    show and plt.show()

if __name__ == '__main__':
    print('What do you want to do?')
    action = str(input('You can type "calculate", "plot", "saveplot" or "all"\n'))
    if action == 'calculate':
        print(f'You chose "{action}"')
        calculate()
        print('Done!')

    elif action == 'plot':
        print(f'You chose "{action}"')
        plot_results(load_file=True)
        print('Done!')

    elif action == 'all' or action == '':
        if action == '':
            action = 'all'
        print(f'You chose "{action}"')
        print('Calculating...')
        calculate()
        plot_results(save=True)
        print('Done!')

    elif action == 'saveplot':
        print(f'You chose "{action}"')
        plot_results(save=True, load_file=True, show=False)
        print('Done!')

    else:
        print('Invalid input!')
