from matplotlib import pyplot as plt
import numpy as np


#setting
filename = 'result22.txt'
top_k = 10


precision = [float(0) for i in range(top_k + 1)]
avg_pre = [float(0) for i in range(top_k + 1)]
count_pre = [int(0) for i in range(top_k + 1)]
recall = [i/top_k for i in range(top_k + 1)]
point = 0
p=0

with open(filename, 'r') as f:
    for line in f:
        if 'p' == line[0]:
            #print(line)

            # get_data
            word = line.split(':')
            temp_precision = float(word[1])
            temp_recall = word[3].split("repr(\)")
            temp_recall = float(temp_recall[0])

            # renew_data
            precision[int(top_k * temp_recall + 0.5)] += temp_precision
            count_pre[int(top_k * temp_recall + 0.5)] += 1

            if temp_recall == 0.1:
                point += temp_precision
                p += 1

            """
            if ((100 * temp_recall)) == 29.0:
                print('yes')

            if ((temp_recall) * top_k) == (0.29 * top_k):
                print(int(0.29 * top_k+ 0.5) )
            """

print(point)
print(p)

"""
for i in range(top_k+1):
    if count_pre[i] == 0:
        print(i)
"""



for i in range(top_k):
    avg_pre[i] = precision[i] / float(count_pre[i])



recall.pop(0)
avg_pre.pop(0)

print(recall)
print(avg_pre)


fig = plt.figure(figsize = (15, 15))
plt.title('Result')
plt.xlabel('Recall')
plt.ylabel('Precision')
#plt.xscale = 5


plt.grid(linestyle='-.')
plt.plot(recall, avg_pre, 'blue', label = 'sign-ALSH')

plt.legend()
plt.show()
