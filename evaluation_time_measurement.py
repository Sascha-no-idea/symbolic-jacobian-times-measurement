import numpy as np
import time
import matplotlib.pyplot as plt

def time_measurement_numpy_float(n):
    """
    Measure the time it takes to calculate the sum of a numpy array of floats
    """
    a = np.random.rand(n)
    a = a.astype(np.float64)
    start = time.time()
    b = a.sum()
    end = time.time()
    return end - start

def time_measurement_numpy_object(n):
    """
    Measure the time it takes to calculate the sum of a numpy array of objects
    """
    a = np.random.rand(n)
    a = a.astype(np.object)
    start = time.time()
    b = a.sum()
    end = time.time()
    return end - start

def time_measurement_list_float(n):
    """
    Measure the time it takes to calculate the sum of a list of floats
    """
    a = np.random.rand(n)
    a = a.tolist()
    start = time.time()
    b = sum(a)
    end = time.time()
    return end - start

 # run every function for differet n values multiple times
 # and create a plot of the time over n
n_list = []
time_measurement_numpy_float_avg = []
time_measurement_numpy_object_avg = []
time_measurement_list_float_avg = []
for i in range(100):
    n = np.random.randint(1000, 1000000)
    n_list.append(n)
    # create empty lists
    time_measurement_numpy_float_list = []
    time_measurement_numpy_object_list = []
    time_measurement_list_float_list = []
    for j in range(10):
        # fill lists with time measurements
        time_measurement_numpy_float_list.append(time_measurement_numpy_float(n))
        time_measurement_numpy_object_list.append(time_measurement_numpy_object(n))
        time_measurement_list_float_list.append(time_measurement_list_float(n))
    time_measurement_numpy_float_avg.append(sum(time_measurement_numpy_float_list)/len(time_measurement_numpy_float_list))
    time_measurement_numpy_object_avg.append(sum(time_measurement_numpy_object_list)/len(time_measurement_numpy_object_list))
    time_measurement_list_float_avg.append(sum(time_measurement_list_float_list)/len(time_measurement_list_float_list))

# plot results
plt.scatter(n_list, time_measurement_numpy_float_avg, label="numpy float")
plt.scatter(n_list, time_measurement_numpy_object_avg, label="numpy object")
plt.scatter(n_list, time_measurement_list_float_avg, label="list float")
plt.xlabel("n")
plt.ylabel("time")
plt.legend()
plt.show()