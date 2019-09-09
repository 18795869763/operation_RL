import matplotlib.pyplot as plt
import os
import numpy as np
import json
import shutil
from scipy import optimize as op

plt.style.use("seaborn")


def plot(info, dic_agent_conf, dic_exp_conf, dic_path, mode, t):
    reward = info[0]
    # print(reward, reward.shape)
    reward = np.array(reward).flatten()
    #reward = np.clip(reward, 0, 1)

    show_len=1
    plot_reward=[]
    plot_t=[]
    for i in range(int(len(reward)/show_len)):
        plot_reward.append(np.mean(reward[(i)*show_len:(i+1)*show_len]))
        plot_t.append(i*show_len)

    plt.figure()
    plt.plot(plot_t,plot_reward)
    plt.title(dic_exp_conf["AGENT_NAME"] + " " + mode)
    #plt.ylim([0, 1])
    plt.legend(["Reward"])

    # year, month, day, hour, minute,
    time = "{}_{}_{}_{}_{}".format(t[0], t[1], t[2], t[3], t[4], t[5])
    path = os.path.join(dic_path[dic_exp_conf["AGENT_NAME"]], str(dic_agent_conf["DELAY_PROB"]), "") + time + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = path + mode + ".png"
    plt.savefig(file_name)


def record(train_info,test_info, dic_agent_conf, dic_exp_conf, dic_env_conf, dic_path, t):
    if dic_exp_conf["AGENT_NAME"] == "DDPG":
        test_reward = test_info[0]


    train_reward = train_info[0]

    result = {}



    time = "{}_{}_{}_{}_{}".format(t[0], t[1], t[2], t[3], t[4], t[5])
    path = os.path.join(dic_path[dic_exp_conf["AGENT_NAME"]], str(dic_agent_conf["DELAY_PROB"]), "") + time + "/"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + "result.txt", "w") as result_file:
        for key, value in result.items():
            result_file.writelines(key + ":" + str(value) + "\n")
    result_file.close()

    with open(path + "agent_conf.json", "w") as agent_conf_file:
        json.dump(dic_agent_conf, agent_conf_file)
    agent_conf_file.close()

    with open(path + "exp_conf.json", "w") as exp_conf_file:
        json.dump(dic_exp_conf, exp_conf_file)
    exp_conf_file.close()

    with open(path + "env_conf.json", "w") as env_conf_file:
        json.dump(dic_env_conf, env_conf_file)
    env_conf_file.close()

    path = path + "/" + "NUMPY FILE/"
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path+"/test_reward", np.array(test_reward))
    np.save(path + "/train_reward", np.array(train_reward))



def clear_record(dic_agent_conf, dic_exp_conf, dic_path):
    path = os.path.join(dic_path[dic_exp_conf["AGENT_NAME"]], str(dic_agent_conf["DELAY_PROB"]))
    shutil.rmtree(path)


def solve_linprog(dic_agent_conf, normal_flow, total_flow, b_ub):
    LB = np.zeros(dic_agent_conf["NUM_AGENTS"])
    UB = np.ones(dic_agent_conf["NUM_AGENTS"])
    bounds = list(zip(LB, UB))
    actions = []
    c = - normal_flow / sum(normal_flow)
    A_ub = total_flow
    res = op.linprog(c, A_ub, b_ub, bounds=bounds)
    if res.success:
        actions.append(res.x)
    else:
        return False
    return 1 - np.array(actions)[0]