#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:49:23 2022

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt

class plots:
    def __init__(self,optimal_val,nS,nA,state_distribution,P,R,state_ratio,T,runs,state,K_pol,behaviour_policy):
        self.optimal_val = optimal_val;
        self.nS = nS;
        self.nA = nA;
        self.state_ratio = state_ratio;
        self.state_distribution = state_distribution
        self.P = P;
        self.R = R;
        self.T = T
        self.runs = runs;
        self.start = state;
        self.K = len(K_pol)
        self.K_pol = K_pol;
        self.behaviour_policy = behaviour_policy
    
    def play_and_plot(self):
        r_est = np.zeros(self.T);
        for run in range(self.runs):
            state = self.start;
            rew_state = np.zeros(self.nS);
            n = np.zeros(self.nS);
            #k=np.random.choice(self.K);
            for t in range(self.T):
                action = np.argmax(np.random.multinomial(1,self.behaviour_policy[state,:]))
                next_state = np.argmax(np.random.multinomial(1,self.P[action,state,:]));
                rew = self.R[action,state];
                n[state]+=1;
                rew_state[state]=(rew_state[state]+rew)/n[state];
                #print(t,"=>",rew);
                #print(rew_state);
                #input('continue');
                #print(self.state_ratio)
                rew_val = [np.dot(rew_state,self.state_ratio[i]) for i in range(self.nS)]
                print(t,"=>",rew_val);
                #input('continue');
                k=np.argmin(rew_val);
                r=np.min(rew_val);
                #print(t,"=>",r,"==>",k,'===>',state);
                r_est[t]+=r;
                self.policy = self.K_pol[k]
                state = next_state;
            #print("Policy played:",self.policy)
            input("Continue:[y/n]");
        r_est=r_est/self.runs;
        cumu_est = np.cumsum(r_est);
        print(cumu_est);
        plt.figure(1)
        plt.plot(cumu_est);
        plt.title('Cost plot')
        plt.xlabel('Time');
        plt.ylabel('Cost')
        reg_val = cumu_est - self.optimal_val*np.arange(1,self.T+1)
        plt.figure(2)
        plt.plot(reg_val);
        plt.title('Regret plot');
        plt.xlabel('Time');
        plt.ylabel('Regret');
            
            