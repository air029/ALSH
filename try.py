import numpy as np
import random
import math
from read_MNIST import *

num_neighbours = 10
radius = 50
r_range = 10 * radius

# original datas
# d = 100#dimension
# test



d = 3
xmin = -10
xmax = 10
num_datas = 6
#num_queries = num_datas / 10
num_queries = 1

datas = [[random.randint(xmin, xmax) for i in range(d)] for j in range(num_datas)]

#test
print(datas)


queries = []
for point in datas[:int(num_queries)]:#queries num
    queries.append([x + random.randint(0,1) for x in point])#100 dimension

print(queries)



#MNIST
datas, lbs = load_mnist("")
datas = datas.astype(np.int)




print(datas[0])
temp = 0
for ix in range(len(datas[0])):
    temp += datas[0][ix] ** 2
print(temp)
print(np.dot(datas[0],datas[0]))
print(np.linalg.norm(datas[0]))
datas = [datas[0]]

def dot(u, v):    #向量内积
    # return sum(ux * vx for ux, vx in zip(u,v))
    return np.dot(u, v)
    #return sum(ux * vx for ux, vx in zip(u,v))

# get max norm for two-dimension list
def g_max_norm(datas):
    norm_list = [math.sqrt(dot(dd, dd)) for dd in datas]
    return max(norm_list)

def g_transformation(datas):
    # U < 1  ||xi||2 <= U <= 1. recommend for 0.83
    U = 0.83
    #U = 0.75
    max_norm = g_max_norm(datas)
    ratio = float(U / max_norm)
    return ratio, max_norm, [[ratio * dx for dx in dd] for dd in datas]

norm_datas = g_transformation(datas)
print(norm_datas)