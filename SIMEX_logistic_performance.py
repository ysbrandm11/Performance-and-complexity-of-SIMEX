"""
File that performs the SIMEX algorithm on a logistic regression model, for
different choises of parameters. The constructed estimators are written to a
file, which can be processed by the "Process_data.py" file. The files it creates
are:
- b_iterations
- b_number_datapoints
- b_size_dataset

The estimators are written to the file for different choises of the respective
parameter. The parameters that are varied are:
- The amount of iterations per zeta.
- The amount of datapoints in the dataset.
- The size of the dataset.

IJsbrand Meeter
University of Amsterdam
student id: 13880624
30 June 2024
Part of thesis: "The complexity and performance of the SIMEX algorithm."
"""
import numpy as np
from SIMEX import *
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LogisticRegression

# Data mean and standard deviation
mu = 0
variance = 1

simulations_per_zeta = [50,100,200,300,500,800, 1000]
numbers_of_datapoints =  [4,6,8,10]
sizes_of_dataset =  [100, 500, 1000]


simulation_per_zeta = 300
number_of_datapoints = 5
size_of_dataset = 1000

# number of repetitions
k = 200

f = open("b_iterations", "a")

for num_sim in simulations_per_zeta:
    true_beta0 = 2
    true_beta1 = 3
    for Case in range(k):
        h = 0.3
        # Generate data
        x, y = make_logistic_data(true_beta1 ,true_beta0, mu=mu,
                                  sigma=variance, n=size_of_dataset)
        x_noise, y_noise  = add_additive_noise(x, h), y

        simex_beta_0, simex_beta_1, zetas = SIMEX(x_noise, y_noise, h,
                                                  amount_of_zeta=number_of_datapoints,
                                                   simulations_per_zeta=num_sim)

        # Fit a and b according to beta_0x^2+beta_1x+beta_2
        fitbeta0_a, fitbeta0_b, fitbeta0_c = np.polyfit(zetas, simex_beta_0, 2)
        fitbeta1_a, fitbeta1_b, fitbeta1_c = np.polyfit(zetas, simex_beta_1, 2)

        true_est_beta0, true_est_beta1 = fit_model(x, y)
        naif_beta0, naif_beta1 = fit_model(x_noise, y_noise)
        simex_beta_0 = fitbeta0_a * (-1)**2 + fitbeta0_b * (-1) + fitbeta0_c
        simex_beta_1 = fitbeta1_a * (-1)**2 + fitbeta1_b * (-1) + fitbeta1_c

        f.write(str(naif_beta0))
        f.write(str(naif_beta1))
        f.write("\n")
        f.write(str(true_est_beta0))
        f.write(str(true_est_beta1))
        f.write("\n")
        f.write(str(simex_beta_0))
        f.write(str(simex_beta_1))
        f.write("\n")
f.write("end")
f.close()

f = open("b_number_datapoints", "a")
for num_data in numbers_of_datapoints:
    true_beta0 = 2
    true_beta1 = 3
    exc_time = 0
    mem = 0
    for Case in range(k):
        h = 0.3

        # Generate data
        x, y = make_logistic_data(true_beta1 ,true_beta0, mu=mu, sigma=variance,
                                  n=size_of_dataset)
        x_noise, y_noise  = add_additive_noise(x, h), y

        simex_beta_0, simex_beta_1, zetas = SIMEX(x_noise, y_noise, h,
                                                  amount_of_zeta=num_data,
                                                   simulations_per_zeta= simulation_per_zeta)

        # Fit a and b according to beta_0x^2+beta_1x+beta_2
        fitbeta0_a, fitbeta0_b, fitbeta0_c = np.polyfit(zetas, simex_beta_0, 2)
        fitbeta1_a, fitbeta1_b, fitbeta1_c = np.polyfit(zetas, simex_beta_1, 2)

        true_est_beta0, true_est_beta1 = fit_model(x, y)
        naif_beta0, naif_beta1 = fit_model(x_noise, y_noise)
        simex_beta_0 = fitbeta0_a * (-1)**2 + fitbeta0_b * (-1) + fitbeta0_c
        simex_beta_1 = fitbeta1_a * (-1)**2 + fitbeta1_b * (-1) + fitbeta1_c

        f.write(str(naif_beta0))
        f.write(str(naif_beta1))
        f.write("\n")
        f.write(str(true_est_beta0))
        f.write(str(true_est_beta1))
        f.write("\n")
        f.write(str(simex_beta_0))
        f.write(str(simex_beta_1))
        f.write("\n")
f.write("end")
f.close()

f = open("b_size_dataset", "a")
for size_data in sizes_of_dataset:
    true_beta0 = 2
    true_beta1 = 3
    exc_time = 0
    mem = 0
    for Case in range(k):
        print("Working on size of dataset:", end=" ")
        print(Case+1, end=" ")
        print(size_data)
        h = 0.3
        # Generate data
        x, y = make_logistic_data(true_beta1 ,true_beta0, mu=mu,
                                  sigma=variance, n=size_data)
        x_noise, y_noise  = add_additive_noise(x, h), y

        simex_beta_0, simex_beta_1, zetas = SIMEX(x_noise, y_noise, h,
                                                  amount_of_zeta=number_of_datapoints,
                                                   simulations_per_zeta=simulation_per_zeta)
        # Fit a and b according to beta_0x^2+beta_1x+beta_2
        fitbeta0_a, fitbeta0_b, fitbeta0_c = np.polyfit(zetas, simex_beta_0, 2)
        fitbeta1_a, fitbeta1_b, fitbeta1_c = np.polyfit(zetas, simex_beta_1, 2)

        true_est_beta0, true_est_beta1 = fit_model(x, y)
        naif_beta0, naif_beta1 = fit_model(x_noise, y_noise)
        simex_beta_0 = fitbeta0_a * (-1)**2 + fitbeta0_b * (-1) + fitbeta0_c
        simex_beta_1 = fitbeta1_a * (-1)**2 + fitbeta1_b * (-1) + fitbeta1_c

        f.write(str(naif_beta0))
        f.write(str(naif_beta1))
        f.write("\n")
        f.write(str(true_est_beta0))
        f.write(str(true_est_beta1))
        f.write("\n")
        f.write(str(simex_beta_0))
        f.write(str(simex_beta_1))
        f.write("\n")
f.write("end")
f.close()