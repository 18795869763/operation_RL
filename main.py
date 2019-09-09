from config import *
from run_exp import main
import tensorflow as tf

dic_exp_conf["AGENT_NAME"] = "DDPG"  #

cost=[5]
for i in range(len(cost)):
    dic_env_conf["CHECK_FIRE_COST"]=cost[i]
    print("cost:",dic_env_conf["CHECK_FIRE_COST"])
    main(dic_agent_conf, dic_exp_conf, dic_env_conf, dic_path)
    tf.reset_default_graph()
