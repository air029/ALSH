# -*- coding: utf-8 -*-

"""
# Locality Sensitive Hashing for MIPS

## Reference:

- LSH:
[Locality-Sensitive Hashing Scheme Based on p-Stable Distributions](http://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/p253-datar.pdf)

- L2-ALSH:
[Asymmetric LSH (ALSH) for Sublinear Time Maximum Inner Product Search (MIPS)](https://papers.nips.cc/paper/5329-asymmetric-lsh-alsh-for-sublinear-time-maximum-inner-product-search-mips.pdf)

- Sign-ALSH(Cosine):
[Improved Asymmetric Locality Sensitive Hashing (ALSH) for Maximum Inner Product Search (MIPS)](http://auai.org/uai2015/proceedings/papers/96.pdf)

- Simple-ALSH(Cosine):
[On Symmetric and Asymmetric LSHs for Inner Product Search](https://arxiv.org/abs/1410.5518)

"""

import random
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from operator import itemgetter

from lsh import *
from lsh_wrapper import *
from lsh_tester import *
from global_mgr import gol
from read_MNIST import *
import datetime


def lsh_test(datas, queries, rand_num, num_neighbours, mips = False):


    """
    type = 'l2'
    tester = LshTesterFactory.createTester(type, mips, datas, queries, rand_num, num_neighbours)
    args = {
                'type':      type,
                'k_vec':     [1, 2, 4, 8],
                #'k_vec':    [2],
                'l_vec':     [2, 4, 8, 16, 32]
                #'l_vec':    [3]
            }
    tester.run(**args)
    """


    type = 'cosine'
    #k_vec = [i for i in range(1, 500)]
    #k_vec = [16, 32, 64, 128, 256, 512]
    #l_vec = [i for i in range(1, 21, 4)]
    tester = LshTesterFactory.createTester(type, mips, datas, queries, rand_num, num_neighbours)

    args = {
                'type':      type,
                #test
                #'k_vec':    [1, 2, 4, 8, 16, 32, 64],
                #'k_vec':     [4, 8],
                'k_vec':     [2],
                #'k_vec':    k_vec,
                #test
                'l_vec':    [2]
                #'l_vec':     [16, 64, 128, 256]
                #'l_vec':       [8]
                #'l_vec':    l_vec
            }
    tester.run(**args)
    """


    type = 'simple'
    tester = LshTesterFactory.createTester(type, mips, datas, queries, rand_num, num_neighbours)

    args = {
                'type':      type,
                'k_vec':    [1, 2],
                #'k_vec':     [2],
                'l_vec':    [2, 4]
                #'l_vec':     [3]
            }
    tester.run(**args)
    """


if __name__ == "__main__":
    #test
    #starttime = datetime.datetime.now()

    gol._init()
    #test
    #gol.set_value('DEBUG', True)

    # create a test dataset of vectors of non-negative integers
    #num_neighbours = 1
    #test

    #setting
    num_neighbours = 2
    radius = 0
    r_range = 10 * radius


    #xmin???
    #original datas
    d = 2
    xmin = 0
    xmax = 10
    num_datas = 12
    num_queries = 1

    
    datas = [[random.randint(xmin, xmax) for i in range(d)] for j in range(num_datas)]

    queries = []
    for point in datas[:int(num_queries)]:#queries num
        queries.append([x + random.uniform(-radius, radius) for x in point])#100 dimension


    """

    #kddalgebra_sample.csv
    #d = read_dimension("kddalgebra_sample.csv")
    #num_datas = read_data("kddalgebra_sample.csv")
    num_queries = 1

    datas = read_data("kddalgebra_sample.csv", False)
    #queries = read_queries("kddalgebra_sample.csv", 1)
    queries = [random.randint(0,1) for i in range(d)]
    """

    """
    #MNIST
    #train
    train, lbs_train = load_mnist("/home/nsklab/mjh/data", "train")
    train = train.astype(np.int)


    d_train = train.shape[1]
    print("d_train")
    print(d_train)
    num_train = train.shape[0]
    print("num_train")
    print(num_train)

    #test
    test, lbs_test = load_mnist("/home/nsklab/mjh/data", "t10k")
    test = test.astype(np.int)


    d_test = test.shape[1]
    print("d_test")
    print(d_test)
    num_test = test.shape[0]
    print("num_test")
    print(num_test)
    
    """

    """
    #test
    #convertion
    for x in range((datas.shape[0])):
        for y in range((datas.shape[1])):
            if datas[x][y] == 0:
                datas[x][y] = 1

    """

    """
    queries = []
    for i in range(num_queries):
        queries.append(datas[i])
    """
    """
    queries = []
    for point in test[:int(num_queries)]:#queries num
        queries.append([x for x in point])
    """


    #lsh_test(datas, queries, r_range, num_neighbours)#???

    # MIPS
    print(datas)
    print(queries)
    lsh_test(datas, queries, r_range, num_neighbours, True)#True???



    #endtime = datetime.datetime.now()

