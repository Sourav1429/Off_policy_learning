#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 18:27:08 2022

@author: Sourav
"""
import weights_parameterization #user made module
from transition_simulator import Simulate_episode #user made module and class
import numpy as np
from itertools import product
import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
  get_batch(): forms a batch. when random value less than equal to 0.5
  Input:
  ------
  batch_size : the size of the batch formed
  T         :  total size of the
'''
class average_case_state_dist:
    def __init__(self,isDeterministic,T,nS,nA,behaviour_policy,target_policy,state,batch_size,l_rate,rep_cost):
        self.isDeterministic = isDeterministic;
        self.T = T;
        self.nS = nS;
        self.nA = nA;
        self.behaviour_policy = behaviour_policy;
        self.target_policy = target_policy
        self.state = state;
        self.batch_size = batch_size;
        self.l_rate = l_rate;
        #self.data = data;
        self.rep_cost = rep_cost
        
    def one_hot(self):
        ret_policy = np.zeros((self.nS,self.nA));
        for i in range(len(self.target_policy)):
            ret_policy[i][int(self.target_policy[i])]=1;
        return ret_policy;
    
    def get_batch(self,data):
        i=1;
        j = np.random.choice(self.T);
        batch,reward = [],[];
        while(i<=self.batch_size):
            if np.random.random()<=0.5:
                batch.append([data['state'][j],data['action'][j],data['next_state'][j]])
                reward.append(data['reward'][j]);
                j=(j+1)%self.T;
                i+=1;
        return batch,reward;
    
    def get_w(self,data,weight_obj,m,pair=0):
        if(pair==1):
            Z_wstate = 0;
            for i in range(len(data)):
                val=weight_obj(data[i][0]);
                Z_wstate+=val;
            return Z_wstate/m;
        else:
            state1,state2,w_state1,w_state2,w_next_state1,w_next_state2,beta1,beta2 = list(),list(),list(),list(),list(),list(),list(),list();
            K=list();
            for i in range(len(data)):
                sample1 = data[i][0];
                sample2 = data[i][1];
                state1.append(sample1[0]);
                w_state1.append(weight_obj(sample1[0]));
                w_next_state1.append(weight_obj(sample1[2]));
                state2.append(sample2[0]);
                w_state2.append(weight_obj(sample2[0]));
                w_next_state2.append(weight_obj(sample2[2]));
                beta1.append((self.target_policy[sample1[0],sample1[1]])/self.behaviour_policy[sample1[0],sample1[1]]);
                beta2.append((self.target_policy[sample2[0],sample2[1]])/self.behaviour_policy[sample2[0],sample2[1]]);
                K.append(sample1[2]==sample2[2]);
            return (state1,state2,w_state1,w_state2,w_next_state1,w_next_state2,beta1,beta2,K);
    
    def get_state_distribution(self):
        weight_obj = weights_parameterization.weights(self.nS, 1)
        optimizerW = optim.Adam(weight_obj.parameters(),lr=self.l_rate)
        obj = Simulate_episode(self.T, self.nS, self.nA, self.rep_cost, self.behaviour_policy, self.state);
        data = obj.play_episode()
        if(self.isDeterministic):
            self.target_policy = self.one_hot();
        print("Behaviour policy:",self.behaviour_policy);
        print("Target_policy:",self.target_policy);
        for _ in range(self.T):
            batch,reward = self.get_batch(data);
            pairs = list(product(batch, repeat=2))
            state1,state2,w_state1,w_state2,w_next_state1,w_next_state2,beta1,beta2,K = self.get_w(pairs,weight_obj,len(batch));
            Z_w_state = self.get_w(batch, weight_obj, len(batch),1);
            #print(len(state1)," is the number of samples used after pairing");
            W_loss = 0;
            for i in range(len(state1)):
                W_loss += (beta1[i]*(w_state1[i]/Z_w_state) - (w_next_state1[i]/Z_w_state))*(beta2[i]*(w_state2[i]/Z_w_state) - (w_next_state2[i]/Z_w_state))*K[i]
            W_loss = W_loss/(2*self.batch_size);
            #loss_store['pol:1111'].append(W_loss);
            optimizerW.zero_grad()
            W_loss.backward()
            optimizerW.step()
            optimizerW.zero_grad()
        state_distribution=[];
        for i in range(self.nS):
            w_state = weight_obj(i);
            w_state = w_state.detach().numpy()[0];
            state_distribution.append(w_state);
        return (state_distribution,data);
        
    
            
'''
T = 1000 # number of times episode is run
nS = 4 #enter number of states
nA = 2 #let this be 2 only
rep_cost = 0.7 #replacement cost
behaviour_policy = np.array([[0.6,0.4],[0.3,0.7],[0.4,0.6],[0.1,0.9]]);
#behaviour_policy = np.array([[0.5,0.5],[1,0]]);
#behaviour_policy = np.array([[0.5,0.5],[0.7,0.3]])
state = 0
batch_size = 50
l_rate=0.1

isDeterministic = False;

#target_policy = np.array([[0,0]]);

#target_policy = np.array([1,1,1,1])

target_policy=np.array([[0.2,0.8],
                        [0.4,0.6],
                        [0.9,0.1],
                        [0.5,0.5]]);
obj = average_case_state_dist(isDeterministic, T, nS, nA, behaviour_policy, target_policy, state, batch_size, l_rate, rep_cost);
print(obj.get_state_distribution())

#input("Should we continu,?");
#loss_store={'pol:1111':[]}


##Now calculating reward

'''
    


'''
import csv
with open('policy_1111.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = ['policy:1111'])
    writer.writeheader()
    writer.writerows(loss_store);
print("Written in csv");
object_rew = est_rew(state_distribution,target_policy,behaviour_policy,data,nS);
print(object_rew.find_reward());'''

