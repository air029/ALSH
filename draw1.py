from matplotlib import pyplot as plt
import numpy as np


#setting
filename = 'RE.txt'


precision, recall = [], []
avg_pre, avg_recall = [], []
line_index = 1
data_index = 0
num_queries = 400
top_k = 10






for i in range(top_k):
    precision.append(float(0))
    recall.append(float(0))
    avg_pre.append(float(0))
    avg_recall.append(float(0))


with open(filename, 'r') as f:
    for line in f:
        if ((line_index - 2) % 18) >= 5 and ((line_index - 2) % 18) <= 14 and line_index >= 2:
            #print(line)

            #renew_data_index
            data_index %= 10

            #get_data
            word = line.split(':')
            temp_precision = float(word[1])
            temp_recall = word[3].split("repr(\)")
            temp_recall = float(temp_recall[0])
            #print(temp_precision)
            #print(temp_recall)

            #renew_data
            precision[data_index] += temp_precision
            recall[data_index] += temp_recall

            #renew_data_index
            data_index += 1

        #renew_line_index
        line_index += 1


#print(precision)
#print(recall)


for i in range(top_k):
    avg_pre[i] = precision[i] / float(num_queries)
    avg_recall[i] = recall[i] / float(num_queries)


print(avg_pre)
print(avg_recall)


fig = plt.figure(figsize = (15, 15))
plt.title('Result')
plt.xlabel('Recall%')
plt.ylabel('Precision%')
#plt.xscale = 5

plt.plot(avg_recall, avg_pre, 'blue', label = 'sign-ALSH')

plt.legend()
plt.show()
