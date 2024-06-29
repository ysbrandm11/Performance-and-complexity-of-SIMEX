"""
File that processes the data from the output of the python code, written to
files. The way the data is stored and created can be found in the "SIMEX
logistic" file. This code reads the data and calculates the bias and variance.

File is run by one of the following commands:
python3 Process_data.py < b_iterations
python3 Process_data.py < b_number_datapoints
python3 Process_data.py < b_size_dataset

IJsbrand Meeter
University of Amsterdam
student id: 13880624
30 June 2024
Part of thesis: "The complexity and performance of the SIMEX algorithm."
"""
import numpy as np

#Defined in previous file.
beta0 = 2
beta1 = 3

beta_0_true = []
beta_1_true = []
beta_0_naif = []
beta_1_naif = []
beta_0_simex = []
beta_1_simex = []

"""
INPUT: beta_list: list of lists of beta estimators.
       beta: the true value of beta.
OUTPUT: list of the bias of the beta estimators for each list.

The function calculates the bias of the beta estimators for each list of beta,
by calculating the absolute difference between the true value and the estimator.
"""
def find_bias(beta, beta_list):
    error = []
    for b in beta_list:
        temp = 0
        for j in b:
            temp+= abs(j-beta)
        temp = temp/len(beta_list[0])
        error.append(temp)
    return error


"""
INPUT: beta_list: list of repetitions of beta estimators.
OUTPUT: list of lists of beta estimators, where each list contains 200 beta
        estimators.

The function creates a matrix of beta estimators, where each list contains 200
beta estimators. The size of these lists is 200, because the experiment is
repeated 200 times.
"""
def make_matrix(beta_list):
    res = []
    for i in range(0, len(beta_list), 200):
        temp = []
        for j in range(200):
            temp.append(beta_list[i+j])
        res.append(temp)
    return res

# Read the data stored in the file.
while(True):
    a = input()
    if a == "end":
        break
    a = a.replace('[', '').replace(']', ' ')
    a, b = map(float, a.split(" "))
    beta_0_naif.append(a)
    beta_1_naif.append(b)

    a = input()
    a = a.replace('[', '').replace(']', ' ')
    a, b = map(float, a.split(" "))
    beta_0_true.append(a)
    beta_1_true.append(b)

    a = input()
    a = a.replace('[', '').replace(']', ' ')
    a, b = map(float, a.split(" "))
    beta_0_simex.append(a)
    beta_1_simex.append(b)

beta_0_naif = make_matrix(beta_0_naif)
beta_0_simex = make_matrix(beta_0_simex)
beta_1_naif = make_matrix(beta_1_naif)
beta_1_simex = make_matrix(beta_1_simex)

print("Bias of Naive beta0:", end=" ")
print(find_bias(beta0, beta_0_naif)[0])
print("Bias of SIMEX beta0:")
print(find_bias(beta0, beta_0_naif))

print("Variance of Naive beta0:", end=" ")
print(np.var(beta_0_naif))
print("Variance of SIMEX beta0:")
print([np.var(b) for b in beta_0_naif])

print("Bias of Naive beta1:", end=" ")
print(find_bias(beta1, beta_1_naif)[0])
print("Bias of SIMEX beta1:")
print(find_bias(beta1, beta_1_naif))

print("Variance of Naive beta1:", end=" ")
print(np.var(beta_1_naif))
print("Variance of SIMEX beta1:")
print([np.var(b) for b in beta_1_naif])
