import matplotlib.pyplot as plt
import os
import numpy as np
import json
import shutil
from scipy import optimize as op

plt.style.use("seaborn")



train=np.load("train_reward.npy")
test=np.load("test_reward.npy")

show_len=1
plot_train = []
plot_t = []
for i in range(int(len(train) / show_len)):
    plot_train.append(np.mean(train[(i) * show_len:(i + 1) * show_len]))
    plot_t.append(i * show_len)

plt.figure()
plt.plot(plot_t, plot_train)
plt.legend(["Reward"])
file_name = "train.png"
plt.savefig(file_name)


plot_test = []
plot_t_ = []
for i in range(int(len(test) / show_len)):
    plot_test.append(np.mean(test[(i) * show_len:(i + 1) * show_len]))
    plot_t_.append(i * show_len)
plt.figure()
plt.plot(plot_t_, plot_test)
plt.legend(["Reward"])
file_name = "test.png"
plt.savefig(file_name)