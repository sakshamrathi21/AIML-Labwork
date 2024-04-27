import numpy as np
import matplotlib.pyplot as plt

def generate_data(n, sigma):
    x_1 = np.random.uniform(size=(n, 1))
    x_2 = np.random.uniform(size=(n, 1))
    y = 3*x_1 + 4*x_2 + np.random.normal(scale=sigma, size=(n, 1))
    return x_1, x_2, y

np.random.seed(2)

sz = 1000
x_1, x_2, y = generate_data(sz, 0.5)


file = open('data.csv', 'w')
file.write('x_1,x_2,y\n')
for i in range(sz):
    file.write(str(x_1[i,0]) + ',' + str(x_2[i,0]) + ',' + str(y[i,0]) + '\n')
