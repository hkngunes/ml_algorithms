'''
Taken from: https://www.kaggle.com/code/sudhirnl7/linear-regression-tutorial/notebook
(x_i, y_i) ith data in dataset
x_i = (x_i_1, x_i_2, ...., x_i_n) j = 1, 2, ..., n
x_i_j - jth independent variable of ith data in dataset - jth feature and ith training example
y_i - dependent variable of ith data in dataset
h_theta - hypothesis function
h_theta(x_i) = theta_0 + theta_1*x_i_1 + theta_2*x_i_2 + ... + theta_m*x_i_m
theta_j - jth parameter of hypothesis function
'''

import pandas as pd                 #data manipulation
import numpy as np                  #data manipulation
import matplotlib.pyplot as plt     #visiulization
#import seaborn as sns #Visualization

plt.rcParams['figure.figsize'] = [8, 5]
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
# plt.style.use('seaborn-whitegrid')

path = 'data/'
df = pd.read_csv(path+'insurance.csv')
print('\nNumber of rows and columns in the data sets: ', df.shape)
