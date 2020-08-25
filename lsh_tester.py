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
import numpy as np
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from operator import itemgetter

from lsh import *
from lsh_wrapper import *
from write import *

class LshTester():
    'LSH Test parameters'

    # datas: datas for build hash index
    # queries: query datas
    # rand_range: random range for norm
    # num_neighbours: query top num_neighbours
    def __init__(self, datas, queries, rand_range = 1.0, num_neighbours = 1):
        kdata = len(datas[0])#datas dimension
        qdata = len(queries[0])#queries dimension
        assert kdata == qdata

        self.d = kdata
        self.datas = datas
        self.queries = queries
        self.rand_range = rand_range
        self.num_neighbours = num_neighbours

        self.q_num = len(self.queries)#num queries

        if True == gol.get_value('DEBUG'):#DEBUG mdoe
            print ('LshTester:\td: ' + str(self.d) + '\trand_range: ' + str(self.rand_range) \
            + '\tnum_beighbours: ' + str(self.num_neighbours) + '\tq_num: ' + str(self.q_num))

            print ('LshTester datas:\n')
            for i, vd in enumerate(self.datas):
                print (str(i) + '\t->\t' + str(vd))
            print ('LshTester queries:\n')
            for i, vq in enumerate(self.queries):
                print (str(i) + '\t->\t' + str(vq))

    def linear(self, q, metric, max_results):   #找出距离q最近的max_results个结果（这是真的，精准的）
        """ brute force search by linear scan """#暴力强解
        candidates = [(ix, metric(q, p)) for ix, p in enumerate(self.datas)]   #enumerate 返回的是索引加数据
        x =  sorted(candidates, key=itemgetter(1))[:max_results]     #最近的top max_results个结果

        if True == gol.get_value('DEBUG'):
            print ('LshTester Sorted Validation:')
            for i, vx in enumerate(x):
                print (str(i) + '\t->\t' + str(vx))#i 原索引 vx metric

        return x

    # type: hash class type
    # k_vec: k vector for hash wrapper. default as [2]
    # l_vec: L vector for hash wrapper. default as [2]
    def run(self, type, k_vec = [2], l_vec = [2]):   #测试hash算法的精确度
        # set distance func object
        try:
            if 'l2' == type:
                metric = L2Lsh.distance
            elif 'cosine' == type:
                metric = CosineLsh.distance
            else:
                raise ValueError
        except ValueError:
            print ('LshTester: type error: ' + str(type))
            return

        exact_hits = [[ix for ix, dist in self.linear(q, metric, self.num_neighbours)] for q in self.queries]   #exact_hits是距离q最近的那些点（这是真的，精准的）
        #exact_hits 是 真实的top n 个点
        #对于每一个q [ix ix ix ix ...]应该是q的top n个点
        print ('==============================')
        print (type + ' TEST:')
        print ('L\tk\tacc\ttouch')

        # concatenating more hash functions increases selectivity
        for k in k_vec:
            lsh = LshWrapper(type, self.d, self.rand_range, k, 0)

            # using more hash tables increases recall
            for L in l_vec:
                lsh.resize(L)#the number of hash table
                lsh.index(self.datas)

                correct = 0
                for q, hits in zip(self.queries, exact_hits):#exact_hits 真实的点
                    lsh_hits = [ix for ix, dist in lsh.query(q, metric, self.num_neighbours)]
                    if lsh_hits == hits:
                        correct += 1

                    if True == gol.get_value('DEBUG'):
                        print ('Queried Sorted Validation:')
                        for j, vd in enumerate(lsh_hits):
                            print (str(q) + '\t->\t' + str(vd))

                # print 'correct: ' + str(correct)
                print ("{0}\t{1}\t{2}\t{3}".format(L, k, float(correct) / self.q_num, float(lsh.get_avg_touched()) / len(self.datas)))
                # print "{0}\t{1}\t{2}\t".format(L, k, float(correct) / self.q_num)

class L2AlshTester(LshTester):
    'L2-ALSH for MIPS Test parameters'

    # datas: datas for build hash index
    # queries: query datas
    # rand_range: random range for norm
    # num_neighbours: query top num_neighbours
    # m: ALSH extend metrix length. default 3
    def __init__(self, datas, queries, rand_range = 1.0, num_neighbours = 1, m = 3):
        kdata = len(datas[0])
        qdata = len(queries[0])
        assert (kdata == qdata)

        self.m = m
        self.half_extend = g_ext_half(self.m)

        # storage original datas & queries. used for validation
        self.origin_datas = datas
        self.origin_queries = queries

        self.q_num = len(self.origin_queries)

        # datas & queries transformation
        dratio, dmax_norm, self.norm_datas = g_transformation(self.origin_datas)
        self.norm_queries = g_normalization(self.origin_queries)
        assert (kdata == len(self.norm_datas[0]))
        assert (qdata == len(self.norm_queries[0]))
        assert (len(self.origin_datas) == len(self.norm_datas))
        assert (len(self.origin_queries) == len(self.norm_queries))

        # expand k dimension into k+2m dimension
        self.ext_datas = g_index_extend(self.norm_datas, self.m)
        self.ext_queries = g_query_extend(self.norm_queries, self.m)
        new_len = kdata + 2 * m
        assert (new_len == len(self.ext_datas[0]))
        assert (new_len == len(self.ext_queries[0]))
        assert (len(self.origin_datas) == len(self.ext_datas))
        assert (len(self.origin_queries) == len(self.ext_queries))

        if True == gol.get_value('DEBUG'):
            print ('L2AlshTester:\td: ' + str(kdata) + '\trand_range: ' + str(rand_range) \
            + '\tnum_beighbours: ' + str(num_neighbours) + '\tq_num: ' + str(self.q_num))

            print ('\tdatas_ratio: ' + str(dratio) + '\tdmax_norm: ' + str(dmax_norm) + '\n')
            print ('L2AlshTester origin_datas:\n')
            for i, vd in enumerate(self.origin_datas):
                print (str(i) + '\t->\t' + str(vd))
            print ('L2AlshTester origin_queries:\n')
            for i, vq in enumerate(self.origin_queries):
                print (str(i) + '\t->\t' + str(vq))

            print ('L2AlshTester norm_datas:\n')
            for i, vd in enumerate(self.norm_datas):
                print (str(i) + '\t->\t' + str(vd))
            print ('L2AlshTester norm_queries:\n')
            for i, vq in enumerate(self.norm_queries):
                print (str(i) + '\t->\t' + str(vq))

        LshTester.__init__(self, self.ext_datas, self.ext_queries, rand_range, num_neighbours)

    # MIPS
    def linear(self, q, metric, max_results):
        """ brute force search by linear scan """
        # print 'MipsLshTester linear:'
        candidates = [(ix, dot(q, p)) for ix, p in enumerate(self.origin_datas)]
        x = sorted(candidates, key=itemgetter(1), reverse=True)[:max_results]

        if True == gol.get_value('DEBUG'):
            print ('L2AlshTester Sorted Validation:')
            for i, vx in enumerate(x):
                print (str(i) + '\t->\t' + str(vx))

        return x

    # type: hash class type
    # k_vec: k vector for hash wrapper. default as [2]
    # l_vec: L vector for hash wrapper. default as [2]
    def run(self, type, k_vec = [2], l_vec = [2]):
        if True == gol.get_value('DEBUG'):
            print ('L2AlshTester Run:')

        try:
            if 'l2' == type:
                pass
            else:
                raise ValueError
        except ValueError:
            print ('L2AlshTester: type error: ' + str(type))
            return

        validate_metric = dot
        compute_metric = L2Lsh.distance

        exact_hits = [[ix for ix, dist in self.linear(q, validate_metric, self.num_neighbours)] for q in self.origin_queries]

        print ('==============================')
        print ('L2AlshTester ' + type + ' TEST:')
        print ('L\tk\tacc\ttouch')

        # concatenating more hash functions increases selectivity
        for k in k_vec:
            lsh = LshWrapper(type, self.d, self.rand_range, k, 0)

            # using more hash tables increases recall
            for L in l_vec:
                lsh.resize(L)
                lsh.index(self.ext_datas)

                correct = 0
                for q, hits in zip(self.ext_queries, exact_hits):
                    lsh_hits = [ix for ix, dist in lsh.query(q, compute_metric, self.num_neighbours)]
                    if lsh_hits == hits:
                        correct += 1

                    if True == gol.get_value('DEBUG'):
                        print ('Queried Sorted Validation:')
                        for j, vd in enumerate(lsh_hits):
                            print (str(q) + '\t->\t' + str(vd))

                # print 'correct: ' + str(correct)
                print ("{0}\t{1}\t{2}\t{3}".format(L, k, float(correct) / self.q_num, float(lsh.get_avg_touched()) / len(self.datas)))
                # print "{0}\t{1}\t{2}\t".format(L, k, float(correct) / self.q_num)

class CosineAlshTester(LshTester):
    'Cosine-ALSH for MIPS Test parameters'

    # datas: datas for build hash index
    # queries: query datas
    # rand_range: random range for norm
    # num_neighbours: query top num_neighbours
    # m: ALSH extend metrix length. default 3
    def __init__(self, datas, queries, rand_range = 1.0, num_neighbours = 1, m = 2):#m=2
        kdata = len(datas[0])#data dimension
        qdata = len(queries[0])#query dimension
        assert (kdata == qdata)

        self.m = m
        self.half_extend = g_ext_half(self.m)#0.5 0.5 0.5

        # storage original datas & queries. used for validation
        self.origin_datas = datas
        self.origin_queries = queries

        self.q_num = len(self.origin_queries)#num_queries

        # datas & queries transformation
        dratio, dmax_norm, self.norm_datas = g_transformation(self.origin_datas)#return ratio, max_norm, new_datas
        self.norm_queries = g_normalization(self.origin_queries)
        assert (kdata == len(self.norm_datas[0]))#dimension no change
        assert (qdata == len(self.norm_queries[0]))#dimension no change
        assert (len(self.origin_datas) == len(self.norm_datas))#num no change
        assert (len(self.origin_queries) == len(self.norm_queries))#num no change

        # expand k dimension into k+2m dimension
        self.ext_datas = g_index_cosine_extend(self.norm_datas, self.m)
        self.ext_queries = g_query_cosine_extend(self.norm_queries, self.m)
        new_len = kdata + 2 * m#new length
        assert (new_len == len(self.ext_datas[0]))
        assert (new_len == len(self.ext_queries[0]))
        assert (len(self.origin_datas) == len(self.ext_datas))
        assert (len(self.origin_queries) == len(self.ext_queries))

        if True == gol.get_value('DEBUG'):
            print ('CosineAlshTester:\td: ' + str(kdata) + '\trand_range: ' + str(rand_range) \
            + '\tnum_neighbours: ' + str(num_neighbours) + '\tq_num: ' + str(self.q_num))
            #write('RE', 'CosineAlshTester:\td: ' + str(kdata) + '\trand_range: ' + str(rand_range) \
            #+ '\tnum_neighbours: ' + str(num_neighbours) + '\tq_num: ' + str(self.q_num))

            print ('\tdatas_ratio: ' + str(dratio) + '\tdmax_norm: ' + str(dmax_norm) + '\n')
            #write('RE', '\tdatas_ratio: ' + str(dratio) + '\tdmax_norm: ' + str(dmax_norm) + '\n')
            print ('CosineAlshTester origin_datas:\n')
            #write('RE', )
            for i, vd in enumerate(self.origin_datas):#origin datas
                print (str(i) + '\t->\t' + str(vd))
            print ('CosineAlshTester origin_queries:\n')
            for i, vq in enumerate(self.origin_queries):#origin queries
                print (str(i) + '\t->\t' + str(vq))

            print ('CosineAlshTester norm_datas:\n')
            for i, vd in enumerate(self.norm_datas):#new datas
                print (str(i) + '\t->\t' + str(vd))
            print ('CosineAlshTester norm_queries:\n')
            for i, vq in enumerate(self.norm_queries):#new queries
                print (str(i) + '\t->\t' + str(vq))

        LshTester.__init__(self, self.ext_datas, self.ext_queries, rand_range, num_neighbours)

    # MIPS
    def linear(self, q, metric, max_results):
        """ brute force search by linear scan """
        # print 'MipsLshTester linear:'
        candidates = [(ix, dot(q, p)) for ix, p in enumerate(self.origin_datas)]
        x = sorted(candidates, key=itemgetter(1), reverse=True)[:max_results]

        if True == gol.get_value('DEBUG'):
            print ('CosineAlshTester Sorted Validation:')
            for i, vx in enumerate(x):
                print (str(i) + '\t->\t' + str(vx))

        return x


    # type: hash class type
    # k_vec: k vector for hash wrapper. default as [2]
    # l_vec: L vector for hash wrapper. default as [2]
    def run(self, type, k_vec = [2], l_vec = [2]):
        if True == gol.get_value('DEBUG'):
            print ('CosineAlshTester Run:')

        try:
            if 'cosine' == type:
                pass
            else:
                raise ValueError
        except ValueError:
            print ('CosineAlshTester: type error: ' + str(type))
            return

        validate_metric = dot
        compute_metric = L2Lsh.distance#???

        #top num_neighbor for each point
        exact_hits = [[ix for ix, dist in self.linear(q, validate_metric, self.num_neighbours)] for q in self.origin_queries]

        #top_N for each point
        N = 10
        top_N = [[ix for ix, dist in self.linear(q, validate_metric, N)] for q in self.origin_queries]


        #print ('==============================')
        write('RE', '==============================\n')
        #print ('CosineAlshTester ' + type + ' TEST:')
        write('RE', 'CosineAlshTester ' + type + ' TEST:\n')
        #print ('L\tk\tacc\ttouch')


        #precision and recall

        K = 6
        R = []
        d = len((self.ext_queries[0]))
        for i in range(K):
            R.append([random.gauss(0, 1) for i in range(d)])
        #print(R)
        print(self.origin_datas)
        print(self.origin_queries)
        print(self.datas)
        print(self.queries)
        print(self.ext_datas)
        print(self.ext_queries)

        #hash(data)
        """
        data_code = []
        for i in range(len(self.ext_datas)):
            data_code.append([np.sign(dot(self.ext_datas[i], R[j])) for j in range(K)])
        """
        data_code = [[np.sign(dot(self.ext_datas[i], R[j])) for j in range(K)] for i in range(len(self.ext_datas))]
        print(data_code)

        #hash(query)
        """
        query_code = []
        for i in range(len(self.ext_queries)):
            query_code.append([np.sign(dot(self.ext_queries[i], R[j])) for j in range(K)])
        """
        query_code = [[np.sign(dot(self.ext_queries[i], R[j])) for j in range(K)] for i in range(len(self.ext_queries))]
        print(query_code)

        #compute matches
        matches = [[0 for i in range(len(self.ext_datas))] for j in range(len(self.ext_queries))]
        print(matches)

        #print("Query code")
        #print(query_code)
        #print("Data code")
        #print(data_code)

        temp_matches = 0

        for i in range(len(self.ext_queries)):
            for j in range(len(self.ext_datas)):
                for m in range(K):
                    if query_code[i][m] == data_code[j][m]:
                        temp_matches += 1

                matches[i][j] = temp_matches
                temp_matches = 0


        print(matches)

        """
        #sort all the points in the set //(index, maches) for each query point
        query_index = 0
        points = [(ix, p) for ix, p in enumerate(matches[query_index])]
        #print("points")
        #print(points)
        sorted_points = sorted(points, key=itemgetter(1), reverse=True)
        #print("sorted_points")
        #print(sorted_points)

        #cumpute precision and recall
        relevant_seen = 0
        k = 0
        for pair in sorted_points:
            if pair[0] in set(top_N[query_index]):
                relevant_seen += 1
            k += 1
            precision = float(relevant_seen) / float(k)
            recall = float(relevant_seen) / float(N)
            print('precision:' + str(precision) + ':recall:' + str(recall))
        """

        print(len(self.ext_queries))


        print("top"+str(N))
        print(top_N)
        print("\n")
        for query_index in range(len(self.ext_queries)):
            # sort all the points in the set //(index, maches) for each query point
            points = [(ix, p) for ix, p in enumerate(matches[query_index])]
            print("points-start")
            print(points)

            sorted_points = sorted(points, key=itemgetter(1), reverse=True)
            print(sorted_points)
            print("points-end")


            # cumpute precision and recall
            relevant_seen = 0
            k = 0
            for pair in sorted_points:
                if pair[0] in set(top_N[query_index]):
                    relevant_seen += 1
                k += 1
                precision = float(relevant_seen) / float(k)
                recall = float(relevant_seen) / float(N)
                print('precision:' + str(precision) + ':recall:' + str(recall))
                #write('RE', "precision:" + str(precision) + ":recall:" + str(recall)+'\n')
                if 1 == recall:
                    break

            print("\n")
            #write('RE', '\n\n')


        return







        # concatenating more hash functions increases selectivity
        for k in k_vec:
            lsh = LshWrapper(type, self.d, self.rand_range, k, 0)

            # using more hash tables increases recall
            for L in l_vec:
                lsh.resize(L)
                lsh.index(self.ext_datas)

                correct = 0
                index_q = 0
                for q, hits in zip(self.ext_queries, exact_hits):
                    lsh_hits = [ix for ix, dist in lsh.query(q, compute_metric, self.num_neighbours)]
                    #test
                    #print('test')
                    #print(lsh_hits) #index in order???与hits长度有时候不相等
                    #print(hits)  #index in order, exact hits
                    if lsh_hits == hits:
                        correct += 1

                    #mine
                    copy_lsh_hits = lsh_hits
                    copy_hits = hits
                    #copy_lsh_hits.sort()
                    #copy_hits.sort()
                    print("hits")
                    #write('RE', "hits\n")
                    print(copy_hits)
                    #write('RE', str(copy_hits)+'\n')

                    #print("lsh_hits")
                    #write('RE', "lsh_hits")
                    #print(copy_lsh_hits)
                    #write('RE', str(copy_lsh_hits))
                    #print(lsh_hits)

                    #precision and recall
                    return_lsh_hits = sorted(copy_lsh_hits, key = lambda x:dot(self.origin_datas[x],self.origin_queries[index_q]), reverse=True)
                    print("return_lsh_hits")
                    #write('RE', "return_lsh_hits\n")
                    print(return_lsh_hits)
                    #write('RE', str(return_lsh_hits)+'\n')

                    """
                    relevant_seen = 0
                    for i in range(len(return_lsh_hits)):
                        if return_lsh_hits[i] in set(top_10[index_q]):
                            relevant_seen += 1
                        precision = float(100 * relevant_seen / (i + 1))
                        recall = float(100 * relevant_seen / 10)
                        print("precision:" + str(precision) + ":recall:" + str(recall))
                        #write('RE', "precision:" + str(precision) + ":recall:" + str(recall)+'\n')
                        #print(precision)
                        #print(recall)
                        if 10 == (i+1):
                            break

                    """
                    """
                    ip = []
                    for i in range(self.num_neighbours):
                        ip.append(dot(self.origin_queries, self.origin_datas[copy_hits[i]]))
                    #ip.sort(reverse=True)
                    print("ip")
                    print(ip)

                    lsh_ip = []
                    for i in range(len(copy_lsh_hits)):
                        lsh_ip.append(dot(self.origin_queries, self.origin_datas[copy_lsh_hits[i]]))
                    #lsh_ip.sort(reverse=True)
                    print("lsh_ip")
                    print(lsh_ip)
                    """

                    #assert (len(copy_lsh_hits) == len(copy_hits))
                    #assert (len(copy_hits) == self.num_neighbours)
                    common_set = set(copy_lsh_hits) & set(copy_hits)

                    print('L\tk')
                    #write('RE', 'L\tk\n')

                    print("{0}\t{1}".format(L, k))
                    #write('RE', "{0}\t{1}".format(L, k)+'\n')

                    #print('precision\t')
                    #write('RE', 'precision')

                    #print("{0}\t".format(len(common_set) / self.num_neighbours))
                    #print("{0}\t".format(len(common_set) / len(hits)))

                    #write('RE', "{0}\t".format(len(common_set) / len(hits)))

                    print('\n')
                    #write('RE', '\n\n')


                    """
                    if (len(common_set) / len(hits)) > 0.6:
                        print('L\tk')
                        print("{0}\t{1}".format(L, k))
                        print('precision\t')
                        # print("{0}\t".format(len(common_set) / self.num_neighbours))
                        print("{0}\t".format(len(common_set) / len(hits)))
                    else:
                        print('precision\t')
                        # print("{0}\t".format(len(common_set) / self.num_neighbours))
                        print("{0}\t".format(len(common_set) / len(hits)))
                    if True == gol.get_value('DEBUG'):
                        print ('Queried Sorted Validation:')
                        for j, vd in enumerate(lsh_hits):
                            print (str(q) + '\t->\t' + str(vd))

                    """

                    index_q += 1

                #print('correct: ' + str(correct))
                #print("{0}\t{1}\t{2}\t{3}".format(L, k, float(correct) / self.q_num, float(lsh.get_avg_touched()) / len(self.datas)))
                # print "{0}\t{1}\t{2}\t".format(L, k, float(correct) / self.q_num)


class SimpleAlshTester(LshTester):
    'Simple-ALSH for MIPS Test parameters'

    # datas: datas for build hash index
    # queries: query datas
    # rand_range: random range for norm
    # num_neighbours: query top num_neighbours
    def __init__(self, datas, queries, rand_range = 1.0, num_neighbours = 1):
        kdata = len(datas[0])
        qdata = len(queries[0])
        assert (kdata == qdata)

        # m: ALSH extend metrix length. default 1
        self.m = 1#???
        self.half_extend = g_ext_half(self.m)

        # storage original datas & queries. used for validation
        self.origin_datas = datas
        self.origin_queries = queries

        self.q_num = len(self.origin_queries)

        # datas & queries transformation
        dratio, dmax_norm, self.norm_datas = g_transformation(self.origin_datas)
        self.norm_queries = g_normalization(self.origin_queries)
        assert (kdata == len(self.norm_datas[0]))
        assert (qdata == len(self.norm_queries[0]))
        assert (len(self.origin_datas) == len(self.norm_datas))
        assert (len(self.origin_queries) == len(self.norm_queries))

        # expand k dimension into k+2m dimension
        self.ext_datas = g_index_simple_extend(self.norm_datas, self.m)
        self.ext_queries = g_query_simple_extend(self.norm_queries, self.m)
        new_len = kdata + 2 * self.m
        assert (new_len == len(self.ext_datas[0]))
        assert (new_len == len(self.ext_queries[0]))
        assert (len(self.origin_datas) == len(self.ext_datas))
        assert (len(self.origin_queries) == len(self.ext_queries))

        if True == gol.get_value('DEBUG'):
            print ('SimpleAlshTester:\td: ' + str(kdata) + '\trand_range: ' + str(rand_range) \
            + '\tnum_beighbours: ' + str(num_neighbours) + '\tq_num: ' + str(self.q_num))

            print ('\tdatas_ratio: ' + str(dratio) + '\tdmax_norm: ' + str(dmax_norm) + '\n')
            print ('SimpleAlshTester origin_datas:\n')
            for i, vd in enumerate(self.origin_datas):
                print (str(i) + '\t->\t' + str(vd))
            print ('SimpleAlshTester origin_queries:\n')
            for i, vq in enumerate(self.origin_queries):
                print (str(i) + '\t->\t' + str(vq))

            print ('SimpleAlshTester norm_datas:\n')
            for i, vd in enumerate(self.norm_datas):
                print (str(i) + '\t->\t' + str(vd))
            print ('SimpleAlshTester norm_queries:\n')
            for i, vq in enumerate(self.norm_queries):
                print (str(i) + '\t->\t' + str(vq))

        LshTester.__init__(self, self.ext_datas, self.ext_queries, rand_range, num_neighbours)

    # MIPS
    def linear(self, q, metric, max_results):
        """ brute force search by linear scan """
        # print 'MipsLshTester linear:'
        candidates = [(ix, dot(q, p)) for ix, p in enumerate(self.origin_datas)]
        x = sorted(candidates, key=itemgetter(1), reverse=True)[:max_results]

        if True == gol.get_value('DEBUG'):
            print ('SimpleAlshTester Sorted Validation:')
            for i, vx in enumerate(x):
                print (str(i) + '\t->\t' + str(vx))

        return x

    # type: hash class type
    # k_vec: k vector for hash wrapper. default as [2]
    # l_vec: L vector for hash wrapper. default as [2]
    def run(self, type, k_vec = [2], l_vec = [2]):
        if True == gol.get_value('DEBUG'):
            print ('SimpleAlshTester Run:')

        try:
            if 'simple' == type:
                pass
            else:
                raise ValueError
        except ValueError:
            print ('SimpleAlshTester: type error: ' + str(type))
            return

        validate_metric = dot
        compute_metric = L2Lsh.distance

        #top num_neighbor for each point
        exact_hits = [[ix for ix, dist in self.linear(q, validate_metric, self.num_neighbours)] for q in self.origin_queries]

        # top10 for each point
        top_10 = [[ix for ix, dist in self.linear(q, validate_metric, 10)] for q in self.origin_queries]

        print ('==============================')
        write('RE', '==============================\n')
        print ('SimpleAlshTester ' + type + ' TEST:')
        write('RE', 'SimpleAlshTester ' + type + ' TEST:\n')
        #print ('L\tk\tacc\ttouch')

        # concatenating more hash functions increases selectivity
        for k in k_vec:
            lsh = LshWrapper('cosine', self.d, self.rand_range, k, 0)

            # using more hash tables increases recall
            for L in l_vec:
                lsh.resize(L)
                lsh.index(self.ext_datas)

                correct = 0
                index_q = 0
                for q, hits in zip(self.ext_queries, exact_hits):
                    lsh_hits = [ix for ix, dist in lsh.query(q, compute_metric, self.num_neighbours)]
                    if lsh_hits == hits:
                        correct += 1

                    if True == gol.get_value('DEBUG'):
                        print ('Queried Sorted Validation:')
                        for j, vd in enumerate(lsh_hits):
                            print (str(q) + '\t->\t' + str(vd))

                    #mine
                    copy_lsh_hits = lsh_hits
                    copy_hits = hits
                    #copy_lsh_hits.sort()
                    #copy_hits.sort()
                    #print("hits")
                    write('RE', "hits\n")
                    #print(copy_hits)
                    write('RE', str(copy_hits)+'\n')

                    #print("lsh_hits")
                    #write('RE', "lsh_hits")
                    #print(copy_lsh_hits)
                    #write('RE', str(copy_lsh_hits))
                    #print(lsh_hits)

                    #precision and recall
                    return_lsh_hits = sorted(copy_lsh_hits, key = lambda x:dot(self.origin_datas[x],self.origin_queries[index_q]), reverse=True)
                    #print("return_lsh_hits")
                    write('RE', "return_lsh_hits\n")
                    #print(return_lsh_hits)
                    write('RE', str(return_lsh_hits)+'\n')

                    """
                    relevant_seen = 0
                    for i in range(len(return_lsh_hits)):
                        if return_lsh_hits[i] in set(top_10[index_q]):
                            relevant_seen += 1
                        precision = float(100 * relevant_seen / (i + 1))
                        recall = float(100 * relevant_seen / 10)
                        #print("precision:" + str(precision) + ":recall:" + str(recall))
                        write('RE', "precision:" + str(precision) + ":recall:" + str(recall)+'\n')
                        #print(precision)
                        #print(recall)
                        if 10 == (i+1):
                            break

                    """

                    """
                    ip = []
                    for i in range(self.num_neighbours):
                        ip.append(dot(self.origin_queries, self.origin_datas[copy_hits[i]]))
                    #ip.sort(reverse=True)
                    print("ip")
                    print(ip)

                    lsh_ip = []
                    for i in range(len(copy_lsh_hits)):
                        lsh_ip.append(dot(self.origin_queries, self.origin_datas[copy_lsh_hits[i]]))
                    #lsh_ip.sort(reverse=True)
                    print("lsh_ip")
                    print(lsh_ip)
                    """

                    #assert (len(copy_lsh_hits) == len(copy_hits))
                    #assert (len(copy_hits) == self.num_neighbours)
                    common_set = set(copy_lsh_hits) & set(copy_hits)

                    #print('L\tk')
                    write('RE', 'L\tk\n')

                    #print("{0}\t{1}".format(L, k))
                    write('RE', "{0}\t{1}".format(L, k)+'\n')

                    #print('precision\t')
                    #write('RE', 'precision')

                    #print("{0}\t".format(len(common_set) / self.num_neighbours))
                    #print("{0}\t".format(len(common_set) / len(hits)))

                    #write('RE', "{0}\t".format(len(common_set) / len(hits)))

                    #print('\n')
                    write('RE', '\n\n')


                    """
                    if (len(common_set) / len(hits)) > 0.6:
                        print('L\tk')
                        print("{0}\t{1}".format(L, k))
                        print('precision\t')
                        # print("{0}\t".format(len(common_set) / self.num_neighbours))
                        print("{0}\t".format(len(common_set) / len(hits)))
                    else:
                        print('precision\t')
                        # print("{0}\t".format(len(common_set) / self.num_neighbours))
                        print("{0}\t".format(len(common_set) / len(hits)))
                    if True == gol.get_value('DEBUG'):
                        print ('Queried Sorted Validation:')
                        for j, vd in enumerate(lsh_hits):
                            print (str(q) + '\t->\t' + str(vd))

                    """

                    index_q += 1

                # print 'correct: ' + str(correct)
                #print ("{0}\t{1}\t{2}\t{3}".format(L, k, float(correct) / self.q_num, float(lsh.get_avg_touched()) / len(self.datas)))
                # print "{0}\t{1}\t{2}\t".format(L, k, float(correct) / self.q_num)

class LshTesterFactory():
    'LSH Test factory'

    @staticmethod
    # type: l2 & cosine
    # mips: True for ALSH
    def createTester(type, mips, datas, queries, rand_num, num_neighbours):
        try:
            if False == mips:
                return LshTester(datas, queries, rand_num, num_neighbours)

            if 'l2' == type:
                return L2AlshTester(datas, queries, rand_num, num_neighbours)
            elif 'cosine' == type:
                return CosineAlshTester(datas, queries, rand_num, num_neighbours)
            elif 'simple' == type:
                return SimpleAlshTester(datas, queries, rand_num, num_neighbours)
            else:
                raise ValueError
        except ValueError:
            print ("LshTesterFactory type error: " + type)
            return
