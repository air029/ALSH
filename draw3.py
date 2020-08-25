from matplotlib import pyplot as plt
import numpy as np



recall = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
avg_pre = [0.011475651941599385, 0.010010261735354731, 0.008978784610987845, 0.008203116772933805, 0.007015562179827442, 0.005991009572506365, 0.004844401126207759, 0.00384624207194464, 0.002595133285106773, 0.0]


print(recall)
print(avg_pre)


fig = plt.figure(figsize = (15, 15))
plt.title('Result')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xticks(recall)
#plt.xscale = 5


plt.grid(linestyle='-.')
plt.plot(recall, avg_pre, 'blue', label = 'sign-ALSH')

plt.legend()
plt.show()
