
TENSORFLOW_SEED = 1
NUMPY_SEED = 1

NUM_terminal = 2#1,2
NUM_server=3#6,7,8
TIME_window=10
Attack_window=10
dic_agent_conf = {
    "MEMORY_SIZE": 7000,#-----

    "COMMUNICATION_NOISE": 0.2,
    "CRITIC_LEARNING_RATE": 1e-4,
    "ACTOR_LEARNING_RATE": 1e-3,
    "GAMMA": 0.99,
    "NUM_TERMINAL": NUM_terminal,
    "NUM_SERVER":NUM_server,
    "TIME_WINDOW":TIME_window,
    "ATTACK_WINDOW":Attack_window,
    "ACTION_NOISE": 0.3,
    "DQN_LEARNING_RATE": 5*1e-4,
    "TARGET_REPLACE_FREQ": 500,
    "TARGET_REPLACE_RATIO": 0.1,
    "BATCH_SIZE": 128,
    "REGULARIZATION": True,
    "L2_REGULARIZATION": 0.1,
    "DELAY": True,
    "DELAY_PROB": 0.0,
    "STATE_DIM": 2*TIME_window,
    "ACTION_DIM": 1,#0 1 if check the fire
    "TENSORFLOW_SEED": TENSORFLOW_SEED,
    "NUMPY_SEED": NUMPY_SEED,
    "DENSE_UNITS": 64,
    "TRAIN": True,
    "RNN": False,
    "SHARE": True,
    "PRIORITIZED_MEMORY":False
}

dic_env_conf = {
    "NUMPY_SEED": NUMPY_SEED,
    "NUM_TERMINAL": NUM_terminal,
    "NUM_SERVER":NUM_server,
    "TIME_WINDOW":TIME_window,
    "ATTACK_WINDOW":Attack_window,
    "ATTACKER_PRO":0.5,
    "ATTACK_PRO":0.3,
    "CHECK_FIRE_COST":5
}

dic_exp_conf = {
    "AGENT_NAME": "DDPG",
    "TRAIN_ITERATIONS": 100,
    "VALIDA_ITERATIONS": 20,
    "TEST_ITERATIONS": 50,
    "MAX_EPISODE_LENGTH": 100,
    "TIME": "2018_12_16_18_42",
    "IMITATION": False,
}

dic_path = {
    "DDPG": "records/DDPG",
    "ComDDPG": "records/Communication",
    "HUMAN_POLICY_1": "records/Human_Policy_1",
    "HUMAN_POLICY_2": "records/Human_Policy_2",
    "DQN": "records/DQN",
    "Linear":"records/Linear"
}