"""
This file contains the SIMEX algorithm and a way of creating logistic data.

IJsbrand Meeter
University of Amsterdam
student id: 13880624
30 June 2024
Part of thesis: "The complexity and performance of the SIMEX algorithm."
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

"""
INPUT:  x: the x values of the logistic data.
        coefficients: the coefficients of the logistic function.
        intercept: the intercept of the logistic function.
OUTPUT: Logistic function output.
Function calculates the standard regression function.
"""
def logistic_function(x, coefficients, intercept):
    return 1 / (1 + np.exp(-(coefficients * x + intercept)))

"""
INPUT:  beta0: the intercept of the logistic function.
        beta1: the coefficient of the logistic function.
        mu: the mean of the data.
        sigma: the standard deviation of the data.
        sigma_e: the standard deviation of the noise.
        n: the amount of data points.
OUTPUT: X: the x values of the logistic data.
        Y: the y values of the logistic data. With Y being binary.

Function creates a logistic dataset, with the given parameters. In the follwing
way:
1. Generate n x_i values from a normal distribution with given mu and sigma.
2. Calculate p_i for each x_i according to the logistic function.
3. Calculate y_i for each x_i according to the binomial distribution with p_i.
"""
def make_logistic_data(beta0,beta1, mu=0, sigma=1, sigma_e=0.1, n=100):
    x = np.random.normal(mu, sigma, size=n)
    y = logistic_function(x, beta1, beta0)
    y = [np.random.binomial(1, p) for p in y]
    return np.array(x), np.array(y)

"""
INPUT:  x: the x values of the logistic data.
        y: the y values of the logistic data.
OUTPUT: beta0: the intercept of the logistic function.
        beta1: the coefficient of the logistic function.
Function fits a logistic regression model to the data using the sklearn library.
Uncomment different solvers to use different solvers. SIMEX wants a solver that
does not use penalties. But due to a bug in the sklearn library, the lbfgs
solver does not have the desired memory usage. The liblinear solver with a high
C value is used to mimic a non penalized solver with correct memory usage.
"""
def fit_model(x, y):
    x = x.reshape(-1, 1)
    #model = LogisticRegression(solver='liblinear',C = 10**100 ,random_state=0)
    model = LogisticRegression(solver='lbfgs', random_state=0, penalty=None)
    model.fit(x, y)
    return model.coef_[0], model.intercept_[0]

"""
INPUT:  X: the x values of the logistic data.
        sigma: the standard deviation of the ME.
OUTPUT: X: the x values of the logistic data with added noise.

Function adds additive noise to the data, with the given standard deviation.
Drawn from a standard normal distribution. With mean 0.
"""
def add_additive_noise(x, sigma):
    return x + np.random.normal(0, sigma, len(x))


"""
INPUT:  x_noise: the x values of the logistic data with noise.
        y: the true y values of the logistic data.
        sigma_u: the standard deviation of the ME.
        amount_of_zeta: the amount of zeta values to use.
        simulations_per_zeta: the amount of repetitions per zeta.
OUTPUT: beta_0: the beta0 estimators of the SIMEX algorithm.
        beta_1: the beta1 estimators of the SIMEX algorithm.
        zetas: the zeta values used in the SIMEX algorithm.
This function perform SIMEX. To read more about the SIMEX algorithm, its
mathematical background and how it is implemented, read the thesis.
"""
def SIMEX(x_noise, y, sigma_u, amount_of_zeta=5, simulations_per_zeta =1000):
    beta_0 = []
    beta_1 = []
    zetas = range(0, amount_of_zeta)
    for zeta in zetas:
        beta_0_prime = 0
        beta_1_prime = 0
        for _ in range(simulations_per_zeta):
            # Make data with (1+zeta)*sigma_u^2 noise
            x_new = add_additive_noise(x_noise, (np.sqrt(zeta)*sigma_u))
            # Calculate new beta0 and beta1
            beta_0_est, beta_1_est = fit_model(x_new, y)
            beta_0_prime += beta_0_est
            beta_1_prime += beta_1_est
        # Take average a,b of the simulations
        if simulations_per_zeta == 0:
            beta_0.append(0)
            beta_1.append(0)
        else:
            beta_0.append(beta_0_prime/simulations_per_zeta)
            beta_1.append(beta_1_prime/simulations_per_zeta)
    return beta_0,beta_1, zetas