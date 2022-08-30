#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 10:37:40 2022

@author: Sourav
"""
from average_case_discrete_step_size import average_case_state_dist
from find_target_policy import Target_Policy
import numpy as np
from Machine_replacement_env_create import Machine_Replacement
from transition_matrix import beh_pol_sd
T = 500 # number of times episode is run
nS = 4 #enter number of states
nA = 2 #let this be 2 only
rep_cost = 0.7 #replacement cost
behaviour_policy = np.array([[0.6,0.4],[0.3,0.7],[0.4,0.6],[0.1,0.9]]);
#behaviour_policy = np.array([[0.5,0.5],[1,0]]);
#behaviour_policy = np.array([[0.5,0.5],[0.7,0.3]])
state = 0
batch_size = 50
runs = 10;
l_rate=0.1

isDeterministic = True;
mr_obj = Machine_Replacement(rep_cost,nS,nA);
P = mr_obj.gen_probability()
R = mr_obj.gen_expected_reward();
print("Probability matrix:");
print(P);
print("Reward matrix:");
print(R);
start = 0;
tar_obj = Target_Policy(np.arange(nS), np.arange(nA), P, R, start)
#target_policy = tar_obj.find_optimum_policy(runs, T)[-1];
target_policy = np.array([[0,0,0,1],
                          [0,0,1,1],
                          [0,1,1,1],
                          [1,1,1,1]]);
print("Target policy:");
print(target_policy);
beh_sd_obj = beh_pol_sd(P, behaviour_policy, nS, nA);
behaviour_policy_state_distribution = beh_sd_obj.state_distribution_simulated(onehot_encode=0);
for policy in target_policy:
    sd_obj = average_case_state_dist(isDeterministic, T, nS, nA, behaviour_policy, policy, state, batch_size, l_rate, rep_cost)
    state_distribution_ratios,data = sd_obj.get_state_distribution();
    print("State distribution ratios:");
    print(state_distribution_ratios)

    print("Behaviour policy state distribution:");
    print(behaviour_policy_state_distribution);
    target_policy_distribution = state_distribution_ratios * behaviour_policy_state_distribution;
    target_policy_distribution = target_policy_distribution/np.sum(target_policy_distribution);
    
    '''cost_dist = np.zeros(nS);
    state_visitation = np.zeros(nS);
    for i in range(T):
        s = data['state'][i];
        state_visitation[s]+=1;
        cost_dist[s]+=R[target_policy[s],s];
    print("Cost_distribution");
    print(cost_dist/state_visitation);'''
    print("Output_cost:");
    cost = 0;
    for i in range(nS):
        cost = cost + target_policy_distribution[i]*R[policy[i],i];
    print(cost);
    
    
