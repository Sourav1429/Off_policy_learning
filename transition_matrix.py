#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 13:33:18 2022

@author: Sourav
"""
import numpy as np
class beh_pol_sd:
    def __init__(self,P,policy,nS,nA):
        self.P = P
        self.policy = policy
        self.nS = nS;
        self.nA = nA;
    
    def onehot(self):
        pol = np.zeros((self.nS,self.nA));
        for i in range(self.nS):
            pol[i][int(self.policy[i])]=1;
        return pol;
    def find_transition_matrix(self,onehot_encode=1):
        if(onehot_encode==1):
            self.policy = self.onehot()
        T_s_s_next = np.zeros((self.nS,self.nS));
        for s in range(self.nS):
            for s_next in range(self.nS):
                for a in range(self.nA):
                    #print(s,s_next,a);
                    #print(T[a,s,s_next]);
                    T_s_s_next[s,s_next]+=self.P[a,s,s_next]*self.policy[s,a];
        return T_s_s_next;
    def state_distribution_simulated(self,onehot_encode=1):
        P_policy = self.find_transition_matrix(onehot_encode)
        #print(P_policy);
        P_dash = np.append(P_policy - np.eye(self.nS),np.ones((self.nS,1)),axis=1);
        #print(P_dash);
        P_last = np.linalg.pinv(np.transpose(P_dash))[:,-1]
        return P_last;





'''nS = 4
nA = 2
rep_cost = 0.7
obj = Machine_Replacement(rep_cost,nS,nA);
P = obj.gen_probability();
#P = np.array([[[0,1],[1,0]],
#                   [[1,0],[0,1]]])
print(P);
#behaviour_policy=np.array([[0.6,0.4],[0.3,0.7],[0.4,0.6],[0.1,0.9]]);
#P=np.array([
 #   [[0, 1],[1, 0]],
 #   [[1, 0],[0, 1]]]);
#behaviour_policy=np.array([[0.5,0.5],[1,0]]);'''
'''policies = np.array([[0,0,0,0],
            [0,0,0,1],
            [0,0,1,1],
            [0,1,1,1],
            [1,1,1,1]]);'''
#behaviour_policy = [[0.5,0.5],[0.7,0.3]]
'''policy = np.array([[0.2,0.8],
                        [0.4,0.6],
                        [0.9,0.1],
                        [0.5,0.5]]);
#policies = np.array([[0,0]]);
obj = beh_pol_sd(P, policy, nS, nA);'''

'''for i in policies:
    print(i,"=========>");
    P_policy=find_transition_matrix(nA,nS,P,i)
    print(P_policy);
    P_dash = np.append(P_policy - np.eye(nS),np.ones((nS,1)),axis=1);
    print(P_dash);
    P_last = np.linalg.pinv(np.transpose(P_dash))[:,-1]
    print(P_last)
    print("==============================================================");'''

'''print("Behaviour policy");
P_policy = obj.find_transition_matrix(0);
print(P_policy);
P_dash = np.append(P_policy - np.eye(nS),np.ones((nS,1)),axis=1);
print(P_dash);
P_last = np.linalg.pinv(np.transpose(P_dash))[:,-1]
print(P_last)'''

