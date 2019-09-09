#
import numpy as np
from queue import Queue
import pandas as pd
import copy
import random


class Env:
    def __init__(self, dic_env_conf):
        self.attackerPro = dic_env_conf["ATTACKER_PRO"]
        self.attackPro = dic_env_conf["ATTACK_PRO"]
        # self.userNum_Experiment = dic_env_conf["MAX_USERNUM"]
        self.check_fire_cost=dic_env_conf["CHECK_FIRE_COST"]
        #self.attackerPro = 0.5
        #self.attackPro = 0.3
        self.userNum_Experiment = 300
        #self.check_fire_cost= 20#20
        self.userMaxTime=60
        self.reset()

    def reset(self):
        self.network = CNetwork()
        self.user = []
        self.aclModified = 0
        self.attackSuccess = 0
        self.states = np.zeros((10, 2))
        self.userAccumulatedNum = 0
        self.successAttackers = []
        self.foundAttackers = []
        stateTemp=copy.deepcopy(self.states)
        stateTemp=stateTemp.reshape(20)
        return stateTemp

    def createNewUser(self):
        userName = 'User' + str(self.userAccumulatedNum)
        isAttacker = random.randint(0,100)
        if (isAttacker> self.attackerPro * 100):
            isAttacker = 0
        else:
            isAttacker = 1
        attackPro = self.attackPro
        self.user.append(CUser(userName,isAttacker,attackPro))
        self.userAccumulatedNum=self.userAccumulatedNum+1
        return

    def step_user(self):
        ##如果用户为空且未达到最大实验用户数量
        if (len(self.user) <= 0 and self.userAccumulatedNum < self.userNum_Experiment):
            self.createNewUser()

        ##已经不再创建新用户，则退出
        if (len(self.user) <= 0):
            return None

        userActionMsg = ""
        teminalDestDic = {'C1': '', 'C2': ''}

        ##每一个用户选择一个动作
        for user in self.user:
            action = user.getNextAction(self)
            # print("User:{},isAttacker:{},Action:{},{},{},{},{}\n".format(user.userName, user.isAttacker, action.act,
            #                                                              action.src, action.dest, action.remark,
            #                                                              action.time))
            ##用户进入房间，将攻击标志和攻击成功标志均置为0
            if (action.act == 'move'):
                user.spaceNow = action.dest
                if (action.src == 'P0' and action.dest == 'P1'):
                    self.aclModified = 0
                    self.attackSuccess = 0
                ##用户退出房间，则删除用户
                elif (action.src == 'P1' and action.dest == 'P0'):
                    self.user.remove(user)
            elif (action.act == 'access'):
                ##标识每台终端访问的服务
                ##teminalDestDic[action.src].append(action.dest)
                teminalDestDic[action.src] = action.dest
                ##如果有更改acl的动作 ,更改aclModified的状态
                if (action.remark == 'S4-F1/start'):
                    self.aclModified = 1
                ##如果攻击成功，除更改变量状态外，用户退出
                elif (action.remark == 'S4-F1/success'):
                    self.aclModified = 0
                    self.attackSuccess = 1
                    ##增加攻击成功用户
                    self.successAttackers.append(user.userName)
                    ##该用户退出
                    user.nextActions.append(CAction('move', 'P1', 'P0', 1, ''))
                    user.nextActions.append(CAction('move', 'P2', 'P1', 1, ''))
            ##用户存在时间
            user.accumulatedTime = user.accumulatedTime + 1

        ##计算下一个state
        ##删除首行并新增一行，完成状态迭代,新状态在矩阵尾部
        self.states = np.delete(self.states, 0, axis=0)
        self.states = np.row_stack((self.states, [0, 0]))
        terminalUsed = ['C1', 'C2']
        shape = self.states.shape
        for i in range(len(terminalUsed)):
            if (teminalDestDic[terminalUsed[i]] == ''):
                self.states[shape[0] - 1, i] = 0
            else:
                serverNum = int(teminalDestDic[terminalUsed[i]][1:])
                self.states[shape[0] - 1, i] = serverNum

        stateTemp = copy.deepcopy(self.states)
        stateTemp = stateTemp.reshape(20)
        return stateTemp


    def step(self, agentAction):
        ##处理管理员动作并计算回报
        reward = 0.0
        # 0 do nothing ;1 check the fire;check has some cost
        if(agentAction==0):
            ##攻击成功
            if(self.attackSuccess ==1):
                reward=-10
                self.attackSuccess=0
        else:
            if(self.aclModified)==1:
                reward=10
                self.aclModified=0
                ##增加被发现的用户(目前只考虑一个用户)
               ## self.c.append(self.user[0].userName)
                self.foundAttackers.append(self.user[0].userName)
                ##强制用户退出
                self.user[0].nextActions=[]
                self.user[0].attackActions=[]
                self.user[0].nextActions.append(CAction('move', 'P1', 'P0', 1, ''))
                if (self.user[0].spaceNow != 'P1'):
                    self.user[0].nextActions.append(CAction('move', 'P2', 'P1', 1, ''))
            else:
                reward = -self.check_fire_cost

        stateTemp = copy.deepcopy(self.states)
        stateTemp = stateTemp.reshape(20)
        return stateTemp,reward


    ##actionKind==0 normal
    ##acrtionKind==1 attack
    def getAction(self,user,actionKind):
        if(actionKind==0):
            pos = random.randint(0, len(self.network.normalActions[user.spaceNow])-1)
            return self.network.normalActions[user.spaceNow][pos],0
        else:
            ##print(user.userName, user.spaceNow, user.isAttacker, user.accumulatedTime)
            pos = random.randint(0, len(self.network.attackActions[user.spaceNow]) - 1)
            return self.network.attackActions[user.spaceNow][pos], 0

class CNetwork:
    def __init__(self):
        self.clients=[]
        self.servers=[]
        self.otherDevices=[]
        self.normalActions=dict({'P0':[],'P1':[],'P2':[]})
        self.attackActions=dict({'P1':[],'P2':[]})
        self.initNetwork()

    def initNetwork(self):
        self.clients.append(('C1', 'P1'))
        self.clients.append(('C2', 'P2'))
        self.servers.append(('S1', 'P3'))
        self.servers.append(('S2', 'P4'))
        self.servers.append(('S3', 'P4'))
        self.servers.append(('S4', 'P4'))
        self.otherDevices.append(('F1', 'P4'))
        self.otherDevices.append(('W1', 'P4'))

        ##正常动作序列
        normalActionList = []
        normalActionList.append(CAction('move', 'P0', 'P1', 1, ''))
        self.normalActions['P0'].append(copy.deepcopy(normalActionList))

        normalActionList = []
        normalActionList.append(CAction('move', 'P1', 'P2',1,''))
        self.normalActions['P1'].append(copy.deepcopy(normalActionList))

        normalActionList = []
        normalActionList.append(CAction('move', 'P2', 'P1',1,''))
        self.normalActions['P2'].append(copy.deepcopy(normalActionList))

        normalActionList = []
        normalActionList.append(CAction('move', 'P1', 'P0',1, ''))
        self.normalActions['P1'].append(copy.deepcopy(normalActionList))

        normalActionList=[]
        normalActionList.append(CAction('access','C1','S2',1,''))
        self.normalActions['P1'].append(copy.deepcopy(normalActionList))

        normalActionList = []
        normalActionList.append(CAction('access','C1','S1',1,''))
        self.normalActions['P1'].append(copy.deepcopy(normalActionList))

        normalActionList = []
        normalActionList.append(CAction('access','C2','F1',1,''))
        self.normalActions['P2'].append(copy.deepcopy(normalActionList))

        normalActionList = []
        normalActionList.append(CAction('access','C2','S4',1,''))
        self.normalActions['P2'].append(copy.deepcopy(normalActionList))

        ##攻击动作序列
        attackActionList=[]
        attackActionList.append(CAction('move', 'P1', 'P2', 1, ''))
        attackActionList.append(CAction('access', 'C2', 'S4', 1, ''))
        attackActionList.append(CAction('access', 'C2', 'S4', 1, 'S4-F1/start'))
        attackActionList.append(CAction('move', 'P2', 'P1', 1, ''))
        attackActionList.append(CAction('access', 'C1', 'S1', 1, ''))
        attackActionList.append(CAction('access', 'C1', 'S1', 1,'S1-S3'))
        attackActionList.append(CAction('move', 'P1', 'P2', 1, ''))
        attackActionList.append(CAction('access', 'C2', 'S4', 1, ''))
        attackActionList.append(CAction('access', 'C2', 'S4', 1, 'S4-F1/success'))
        self.attackActions['P1'].append(copy.deepcopy(attackActionList))

        attackActionList = []
        attackActionList.append(CAction('access', 'C2', 'S4', 1,''))
        attackActionList.append(CAction('access', 'C2', 'S4', 1, 'S4-F1/start'))
        attackActionList.append(CAction('move', 'P2', 'P1', 1, ''))
        attackActionList.append(CAction('access', 'C1', 'S1', 1,''))
        attackActionList.append(CAction('access', 'C1', 'S1', 1,'S1-S3'))
        attackActionList.append(CAction('move', 'P1', 'P2', 1, ''))
        attackActionList.append(CAction('access', 'C2', 'S4', 1,''))
        attackActionList.append(CAction('access', 'C2', 'S4', 1, 'S4-F1/success'))
        self.attackActions['P2'].append(copy.deepcopy(attackActionList))

class CAction:
    def __init__(self, _act, _src, _dest, _time, _remark):
        self.act = _act
        self.src = _src
        self.dest = _dest
        self.time = _time
        self.remark = _remark


class CUser:
    def __init__(self, _userName, _isAttacker,_attackProbability):
        self.isAttacker = _isAttacker
        self.userName=_userName
        self.nextActions=[]
        self.attackActions=[]
        self.spaceNow='P0'
        self.terminalNow=-1
        self.accumulatedTime=0
        if(self.isAttacker):
            if(_attackProbability>=0 and _attackProbability<=1):
                self.attackProbability=_attackProbability
        else:
            self.attackProbability=0

        ##beingAttack=0代表未开始攻击，为1代表正在攻击（多步攻击未完成），为2代表攻击已经完成
        self.beingAttack = 0


    ##得到下一步动作
    def getNextAction(self,env):
        ##如果用户当前未进入房间，则动作为进入房间
        if (self.spaceNow == 'P0' and self.accumulatedTime==0):
            return CAction('move', 'P0', 'P1', 1,'')

        ##如果超过最长时间，则退出
        if (self.accumulatedTime >= env.userMaxTime-1 and self.spaceNow == 'P2'):
            self.nextActions = []
            return CAction('move','P2','P1',1,'')

        if (self.accumulatedTime >= env.userMaxTime-1 and self.spaceNow == 'P1'):
            return CAction('move', 'P1', 'P0', 1,'')

        ##如果上一个动作都已经完成了，选择一个新动作
        if(len(self.nextActions)<=0 and len(self.attackActions)<=0):
            self.chooseAction(env)

        ##如果上一个正常动作未完成，则完成相应的动作
        if(len(self.nextActions)>0):
            actionTemp = self.nextActions.pop()
            ##如果动作完成需要多个时间片
            if (actionTemp.time>1):
                actionTemp.time = actionTemp.time - 1
                self.nextActions.append(actionTemp)
                return CAction(actionTemp.act, actionTemp.src, actionTemp.dest, actionTemp.time,actionTemp.remark)
            else:
                return actionTemp

        ##如果上一个攻击动作未完成，则完成相应的动作
        if (len(self.attackActions) > 0):
            actionTemp = self.attackActions.pop()
            ##print(actionTemp)
            ##如果动作完成需要多个时间片
            if (actionTemp.time > 1):
                actionTemp.time = actionTemp.time - 1
                self.attackActions.append(actionTemp)
                return CAction(actionTemp.act, actionTemp.src, actionTemp.dest, actionTemp.time,actionTemp.remark)
            else:
                return actionTemp

    ##根据环境信息选择一个动作
    def chooseAction(self,env):
        ##如果攻击者还没有开始攻击
        if (self.isAttacker and self.beingAttack == 0):
            temp = random.randint(0, 100)
            if (temp < self.attackProbability * 100):
                actions, actionKind = env.getAction(self, 1)  ##attack
            else:
                actions, actionKind = env.getAction(self, 0)  ##normal
        ##如果已经开始攻击了，则选择继续攻击动作或一个正常动作
        # elif(self.beingAttack==1):
        #     temp = random.randint(0, 100)
        #     if (temp < self.attackProbability * 100):
        #         return env.getAction(1)##attack
        #     else:
        #         return env.getAction(0)##normal
        ##如果攻击者已经完成攻击或非攻击者
        else:
            actions, actionKind = env.getAction(self, 0)  ##normal

        ##保存动作，倒序拷贝
        # 如果是正常动作
        if (actionKind == 'normal'):
            self.nextActions = copy.deepcopy(actions)
            self.nextActions.reverse()
        else:  ##如果是攻击动作
            self.attackActions = copy.deepcopy(actions)
            self.attackActions.reverse()
        return



