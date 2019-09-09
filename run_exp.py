from ddpg import DDPG
from env_com import Env
from util import plot, record, clear_record, solve_linprog
import time
import numpy as np


def main(dic_agent_conf, dic_exp_conf, dic_env_conf, dic_path):
    np.random.seed(dic_agent_conf["NUMPY_SEED"])
    t = time.localtime(time.time())

    flag = False
    #train_show=[]
    if dic_exp_conf["AGENT_NAME"] == "DDPG":
        agent = DDPG(dic_agent_conf, dic_exp_conf, dic_path)
        flag = True
    print("=== Build Agent: %s ===" % dic_exp_conf["AGENT_NAME"])

    # ===== train =====
    print("=== Train Start ===")
    train_reward = []

    env = Env(dic_env_conf)
    for cnt_train_iter in range(dic_exp_conf["TRAIN_ITERATIONS"]):
        s = env.reset()
        r_sum = 0
        cnt_train_step=0
        while(True):
            ##管理员先动作
            a = agent.choose_action(s, explore=True)
            action=0
            if a>=0.5:
                action=1
            s_, r = env.step(action)
            r_sum += r
            train_reward.append(r)

            s_ = env.step_user()
            if (s_ is None):
                break

            if "DDPG" in dic_exp_conf["AGENT_NAME"]:
                agent.store_transition(s, a, r, s_)

            s = s_

            if "DDPG" in dic_exp_conf["AGENT_NAME"]:
                if agent.memory_batch_full:
                    agent.learn()

            cnt_train_step=cnt_train_step+1
            if cnt_train_step%100==0:
                with open('result.txt', 'a+') as f:
                    f.write("train: iter:{}, step:{}, r_sum:{},rewrd:{},action:{},successAttackers:{},foundAttackers:{}\n".format(
                    cnt_train_iter,cnt_train_step,r_sum,r,a,len(env.successAttackers), len(env.foundAttackers)))
                print(s)
                print("train, step:{}, r_sum:{},rewrd:{},action:{},successAttackers:{},foundAttackers:{}".format(
                    cnt_train_step,r_sum,r,a,len(env.successAttackers), len(env.foundAttackers)))


        print("train, iter:{}, r_sum:{}".format(cnt_train_iter, r_sum))
        train_reward.append(r_sum)



    if not dic_agent_conf["TRAIN"] and flag:
        agent.save_model(t)
    print("=== Train End ===")


    # ==== test ====
    print("=== Test Start ===")
    test_reward = []
    if dic_exp_conf["AGENT_NAME"]:
        test_com_cnt = []

    for cnt_test_iter in range(dic_exp_conf["TEST_ITERATIONS"]):
        s = env.reset()
        r_sum = 0

        cnt_test_step = 0

        while (True):
            ##管理员先动作
            a = agent.choose_action(s, explore=False)
            action = 0
            if a > 0.5:
                action = 1
            s_, r = env.step(action)

            s_ = env.step_user()
            if (s_ is None):
                break



        #for cnt_test_step in range(dic_exp_conf["MAX_EPISODE_LENGTH"]):
         #   a = agent.choose_action(s, explore=False)


            # action = 0
            # if a > 0.5:
            #     action = 1
            # s_, r = env.step(action)

            #s_, r = env.step(a)
            r_sum += r

            #test_reward.append(r)

            s = s_
            #if cnt_test_step%50==0:
            cnt_test_step = cnt_test_step + 1
            if cnt_test_step % 100 == 0:
                #
                print("test, step:{}, r_sum:{},rewrd:{},action:{},successAttackers:{},foundAttackers:{}".format(cnt_test_step, r_sum, r, a, len(env.successAttackers), len(env.foundAttackers)))

        if dic_exp_conf["AGENT_NAME"] == "ComDDPG":
            test_com_cnt.append(agent.communicate_counter)
        #print("test, iter:{}, r_sum:{}".format(cnt_test_iter, r))
        print("test, iter:{}, r_sum:{}".format(cnt_test_iter, r))
        test_reward.append(r_sum)




    print("=== Test End ===")
    # ==== record ====
    print("=== Record Begin ===")

    train_info = [train_reward]
    if dic_exp_conf["AGENT_NAME"] == "DDPG":
        test_info = [test_reward]
    else:
        test_info = [test_reward]

    plot(train_info, dic_agent_conf, dic_exp_conf, dic_path, "TRAIN", t)
    plot(test_info, dic_agent_conf, dic_exp_conf, dic_path, "TEST", t)
    record(train_info,test_info, dic_agent_conf, dic_exp_conf, dic_env_conf, dic_path, t)
    print("=== Record End ===")
