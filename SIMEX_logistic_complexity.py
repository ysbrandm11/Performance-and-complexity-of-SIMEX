"""
File that performs the SIMEX algorithm on a logistic regression model, for
different choises of parameters. The time and memory usage of the algorithm is
monitored and plotted at the end. The parameters that are varied are:
- The amount of iterations per zeta.
- The amount of datapoints in the dataset.
- The size of the dataset.

The time and space complexity are plotted and saved in the following files:
- Simulations per zeta.png
- number of datapoints.png
- size of the dataset.png

IJsbrand Meeter
University of Amsterdam
student id: 13880624
30 June 2024
Part of thesis: "The complexity and performance of the SIMEX algorithm."
"""

import numpy as np
import matplotlib.pyplot as plt
from SIMEX import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import time
import tracemalloc

# Data mean and standard deviation
mu = 0
variance = 1

clean_list = [100,200,300]
simulations_per_zeta = clean_list + [50,100,200,300,500,800, 1000]
clean_list = [2,2,3]
numbers_of_datapoints = clean_list + [4,6,8,10]
clean_list = [100,100,100]
sizes_of_dataset = clean_list + [100, 500, 1000]

simulation_per_zeta = 300
number_of_datapoints = 5
size_of_dataset = 1000

# number of repetitions
k = 200


execution_time = []
memory_usage = []

for num_sim in simulations_per_zeta:
    true_beta0 = 2
    true_beta1 = 3
    exc_time = 0
    mem = 0
    for _ in range(k):
        h = 0.3
        # Generate data
        x, y = make_logistic_data(true_beta1 ,true_beta0, mu=mu, sigma=variance, n=size_of_dataset)
        true_est_beta0, true_est_beta1 = fit_model(x, y)
        print(true_est_beta0, true_est_beta1)

        x_values = np.linspace(np.min(x), np.max(x), 100)

        # Add noise to data
        x_noise,y_noise  = add_additive_noise(x, h), y
        naif_beta0, naif_beta1 = fit_model(x_noise, y_noise)

        fitted_beta0, fitted_beta1 = fit_model(x, y)

        """
        UNCOMMENT to show the differences of the estimators.
        """
        # x_values = np.linspace(np.min(x), np.max(x), 100)
        # plt.figure(figsize=(10, 6))
        # plt.scatter(x, y, color='red', label='Original Data')
        # plt.plot(x_values, logistic_function(x_values, true_beta0, true_beta1),
        #          color='green', linewidth=3, label='Original Fitted Curve')
        # plt.plot(x_values, logistic_function(x_values, true_est_beta0,
        #                                      true_est_beta1), color='red',
        #                                      linewidth=3, label='Original Fitted Curve')
        # # plt.scatter(x_noise, y_noise, color='green', label='Noisy Data')
        # plt.plot(x_values, logistic_function(x_values, naif_beta0, naif_beta1),
        #          color='green', linewidth=3, label='Noisy Fitted Curve')
        # plt.xlabel('X')
        # plt.ylabel('Probability')
        # plt.title('Logistic Regression')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # start the tracker
        tracemalloc.start()
        print("memory for", end=" ")
        print(num_sim)
        # start the clock
        start = time.time()

        # Perform SIMEX to get a new estimate of beta
        simex_beta_0, simex_beta_1, zetas = SIMEX(x_noise, y_noise, h, amount_of_zeta=number_of_datapoints,
                                                   simulations_per_zeta=num_sim)
        # Fit a and b according to beta_0x^2+beta_1x+beta_2
        fitbeta0_a, fitbeta0_b, fitbeta0_c = np.polyfit(zetas, simex_beta_0, 2)
        fitbeta1_a, fitbeta1_b, fitbeta1_c = np.polyfit(zetas, simex_beta_1, 2)

        """
        UNCOMMENT TO SHOW PLOTS of the extrapolation.
        """
        # x_lin = np.linspace(-1, 5, 100)
        # plt.scatter(zetas, simex_beta_0)
        # plt.plot(x_lin, fitbeta0_a*x_lin**2 + fitbeta0_b*x_lin+fitbeta0_c,label="simex_beta0",  c='r')
        # plt.scatter(zetas, simex_beta_1)
        # plt.plot(x_lin, fitbeta1_a*x_lin**2 + fitbeta1_b*x_lin+fitbeta1_c, label="simex_beta1", c='r')
        # plt.title('simex estimators for beta0 and beta1')
        # plt.show()

        # Calculate the simex estimators, by filling in -1 in the polynomial.
        simex_beta_0 = fitbeta0_a * (-1)**2 + fitbeta0_b * (-1) + fitbeta0_c
        simex_beta_1 = fitbeta1_a * (-1)**2 + fitbeta1_b * (-1) + fitbeta1_c
        # stop the clock()
        end = time.time()
        # stop the memory tracker
        _, memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        exc_time += end-start
        mem += memory

    execution_time.append(exc_time/k)
    memory_usage.append(mem/k)

# Plotting data on the subplot
fig, axes = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(12, 6))
print(simulations_per_zeta[3:])
print(memory_usage[3:])

# Plotting data on each subplot
axes[0].plot(simulations_per_zeta[3:], execution_time[3:], color="black", label='Execution time')
axes[0].set_ylabel("Time in seconds")
axes[0].grid()
axes[0].legend()

axes[1].plot(simulations_per_zeta[3:], memory_usage[3:], color="black", label='memory usage')
axes[1].set_ylabel("Memory in bytes")
axes[1].grid()
axes[1].legend()

# Adding common X and Y labels
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Amount of iterations per zeta's")
#plt.title("Time and space usage for different amount of iteration per zeta")
plt.tight_layout()
plt.savefig('Simulations per zeta.png')
plt.show()

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Amount of runs of SIMEX.
execution_time = []
memory_usage = []

for num_data in numbers_of_datapoints:
    true_beta0 = 2
    true_beta1 = 3
    exc_time = 0
    mem = 0
    for _ in range(k):
        h = 0.3
        # Generate data
        x, y = make_logistic_data(true_beta1 ,true_beta0, mu=mu, sigma=variance, n=size_of_dataset)
        true_est_beta0, true_est_beta1 = fit_model(x, y)

        x_values = np.linspace(np.min(x), np.max(x), 100)

        # Add noise to data
        x_noise,y_noise  = add_additive_noise(x, h), y
        naif_beta0, naif_beta1 = fit_model(x_noise, y_noise)
        fitted_beta0, fitted_beta1 = fit_model(x, y)

        """
        UNCOMMENT to show the differences of the estimators.
        """
        # plt.scatter(x,y)
        # plt.plot(x_values, logistic_function(x_values, fitted_beta1, fitted_beta0), c='black', label="True")
        # plt.show()
        # x_values = np.linspace(np.min(x), np.max(x), 100)
        # plt.figure(figsize=(10, 6))
        # plt.scatter(x, y, color='red', label='Original Data')
        # plt.plot(x_values, logistic_function(x_values, true_beta0, true_beta1), color='green', linewidth=3, label='Original Fitted Curve')
        # plt.plot(x_values, logistic_function(x_values, true_est_beta0, true_est_beta1), color='red', linewidth=3, label='Original Fitted Curve')
        # plt.scatter(x_noise, y_noise, color='green', label='Noisy Data')
        # plt.plot(x_values, logistic_function(x_values, naif_beta0, naif_beta1), color='green', linewidth=3, label='Noisy Fitted Curve')
        # plt.xlabel('X')
        # plt.ylabel('Probability')
        # plt.title('Logistic Regression')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # start the tracker
        tracemalloc.start()
        print("memory for", end=" ")
        print(num_data)
        # start the clock
        start = time.time()

        # Perform SIMEX to get a new estimate of beta
        simex_beta_0, simex_beta_1, zetas = SIMEX(x_noise, y_noise, h,
                                                  amount_of_zeta=num_data,
                                                  simulations_per_zeta=simulation_per_zeta)
        # Fit a and b according to beta_0x^2+beta_1x+beta_2
        fitbeta0_a, fitbeta0_b, fitbeta0_c = np.polyfit(zetas, simex_beta_0, 2)
        fitbeta1_a, fitbeta1_b, fitbeta1_c = np.polyfit(zetas, simex_beta_1, 2)
        """
        UNCOMMENT TO SHOW PLOTS of the extrapolation.
        """
        # x_lin = np.linspace(-1, 5, 100)
        # plt.scatter(zetas, simex_beta_0)
        # plt.plot(x_lin, fitbeta0_a*x_lin**2 + fitbeta0_b*x_lin+fitbeta0_c,label="simex_beta0",  c='r')
        # plt.scatter(zetas, simex_beta_1)
        # plt.plot(x_lin, fitbeta1_a*x_lin**2 + fitbeta1_b*x_lin+fitbeta1_c, label="simex_beta1", c='r')
        # plt.title('simex estimators for beta0 and beta1')
        # plt.show()

        # Calculate the simex estimators, by filling in -1 in the polynomial.
        simex_beta_0 = fitbeta0_a * (-1)**2 + fitbeta0_b * (-1) + fitbeta0_c
        simex_beta_1 = fitbeta1_a * (-1)**2 + fitbeta1_b * (-1) + fitbeta1_c

        # stop the clock()
        end = time.time()
        # stop the memory tracker
        _, memory = tracemalloc.get_traced_memory()
        exc_time += end-start
        mem += memory

    execution_time.append(exc_time/k)
    memory_usage.append(mem/k)

# Plotting data on the subplot
fig, axes = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(12, 6))

# Plotting data on each subplot
axes[0].plot(numbers_of_datapoints[3:], execution_time[3:], color="black", label='Execution time')
axes[0].set_ylabel("Time in seconds")
axes[0].grid()
axes[0].legend()

axes[1].plot(numbers_of_datapoints[3:], memory_usage[3:], color="black", label='memory usage')
axes[1].set_ylabel("Memory in bytes")
axes[1].grid()
axes[1].legend()

# Adding common X and Y labels
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Number of datapoints")
#plt.title("Time and space usage for different amount of iteration per zeta")
plt.tight_layout()
plt.savefig('number of datapoints.png')
plt.show()

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# Amount of runs of SIMEX.
execution_time = []
memory_usage = []

for size_dataset in sizes_of_dataset:
    true_beta0 = 2
    true_beta1 = 3
    exc_time = 0
    mem = 0
    for _ in range(k):
        h = 0.3
        # Generate data
        x, y = make_logistic_data(true_beta1 ,true_beta0, mu=mu, sigma=variance, n=size_dataset)
        true_est_beta0, true_est_beta1 = fit_model(x, y)

        x_values = np.linspace(np.min(x), np.max(x), 100)

        # Add noise to data
        x_noise,y_noise  = add_additive_noise(x, h), y
        naif_beta0, naif_beta1 = fit_model(x_noise, y_noise)
        fitted_beta0, fitted_beta1 = fit_model(x, y)

        """
        UNCOMMENT to show the differences of the estimators.
        """
        # plt.scatter(x,y)
        # plt.plot(x_values, logistic_function(x_values, fitted_beta1, fitted_beta0), c='black', label="True")
        # plt.show()
        # UNCOMMET TO SHOW PLOTS
        # x_values = np.linspace(np.min(x), np.max(x), 100)
        # plt.figure(figsize=(10, 6))
        # plt.scatter(x, y, color='red', label='Original Data')
        # plt.plot(x_values, logistic_function(x_values, true_beta0, true_beta1), color='green', linewidth=3, label='Original Fitted Curve')
        # plt.plot(x_values, logistic_function(x_values, true_est_beta0, true_est_beta1), color='red', linewidth=3, label='Original Fitted Curve')
        # plt.scatter(x_noise, y_noise, color='green', label='Noisy Data')
        # plt.plot(x_values, logistic_function(x_values, naif_beta0, naif_beta1), color='green', linewidth=3, label='Noisy Fitted Curve')
        # plt.xlabel('X')
        # plt.ylabel('Probability')
        # plt.title('Logistic Regression')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # start the tracker
        tracemalloc.start()
        # start the clock
        start = time.time()

        # Perform SIMEX to get a new estimate of beta
        simex_beta_0, simex_beta_1, zetas = SIMEX(x_noise, y_noise, h,
                                                  amount_of_zeta=number_of_datapoints,
                                                  simulations_per_zeta=simulation_per_zeta)
        # Fit a and b according to beta_0x^2+beta_1x+beta_2
        fitbeta0_a, fitbeta0_b, fitbeta0_c = np.polyfit(zetas, simex_beta_0, 2)
        fitbeta1_a, fitbeta1_b, fitbeta1_c = np.polyfit(zetas, simex_beta_1, 2)

        """
        UNCOMMENT TO SHOW PLOTS of the extrapolation.
        """
        # x_lin = np.linspace(-1, 5, 100)
        # plt.scatter(zetas, simex_beta_0)
        # plt.plot(x_lin, fitbeta0_a*x_lin**2 + fitbeta0_b*x_lin+fitbeta0_c,label="simex_beta0",  c='r')
        # plt.scatter(zetas, simex_beta_1)
        # plt.plot(x_lin, fitbeta1_a*x_lin**2 + fitbeta1_b*x_lin+fitbeta1_c, label="simex_beta1", c='r')
        # plt.title('simex estimators for beta0 and beta1')
        # plt.show()

        # Calculate the simex estimators, by filling in -1 in the polynomial.
        simex_beta_0 = fitbeta0_a * (-1)**2 + fitbeta0_b * (-1) + fitbeta0_c
        simex_beta_1 = fitbeta1_a * (-1)**2 + fitbeta1_b * (-1) + fitbeta1_c

        # stop the clock()
        end = time.time()
        # stop the memory tracker
        _, memory = tracemalloc.get_traced_memory()
        exc_time += end-start
        mem += memory

    execution_time.append(exc_time/k)
    memory_usage.append(mem/k)

# Plotting data on the subplot
fig, axes = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(12, 6))
print(simulations_per_zeta[3:])
print(memory_usage[3:])
# Plotting data on each subplot
axes[0].plot(sizes_of_dataset[3:], execution_time[3:], color="black", label='Execution time')
axes[0].set_ylabel("Time in seconds")
axes[0].grid()
axes[0].legend()

axes[1].plot(sizes_of_dataset[2:], memory_usage[2:], color="black", label='memory usage')
axes[1].set_ylabel("Memory in bytes")
axes[1].grid()
axes[1].legend()

# Adding common X and Y labels
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Size of the dataset")
#plt.title("Time and space usage for different amount of iteration per zeta")
plt.tight_layout()
plt.savefig('size of the dataset.png')
plt.show()