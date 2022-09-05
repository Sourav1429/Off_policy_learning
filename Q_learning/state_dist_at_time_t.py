#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 23:06:40 2022

@author: Sourav
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

path1 = "/home/user/.config/spyder-py3/pReinforce and off policy learning/Q_learning/State_distribution_policy_1111_state_0.xlsx"
path2 = "/home/user/.config/spyder-py3/pReinforce and off policy learning/Q_learning/State_distribution_policy_1111_state_1.xlsx"
path3 = "/home/user/.config/spyder-py3/pReinforce and off policy learning/Q_learning/State_distribution_policy_1111_state_2.xlsx"
path4 = "/home/user/.config/spyder-py3/pReinforce and off policy learning/Q_learning/State_distribution_policy_1111_state_3.xlsx"

data1 = pd.read_excel(path1);
data2 = pd.read_excel(path2);
data3 = pd.read_excel(path3);
data4 = pd.read_excel(path4);

T=1000;
runs = 10
state_0_time_dist,state_1_time_dist,state_2_time_dist,state_3_time_dist = list(),list(),list(),list()
for t in range(T):
    state_0_time_dist.append(np.sum(data1[t])/runs);
    state_1_time_dist.append(np.sum(data2[t])/runs);
    state_2_time_dist.append(np.sum(data3[t])/runs);
    state_3_time_dist.append(np.sum(data4[t])/runs);
state_0_time_dist = np.array(state_0_time_dist)
state_1_time_dist = np.array(state_1_time_dist)
state_2_time_dist = np.array(state_2_time_dist)
state_3_time_dist = np.array(state_3_time_dist)

plt.figure();
plt.plot(np.cumsum(state_0_time_dist)/np.arange(1,T+1));
plt.plot(np.cumsum(state_1_time_dist)/np.arange(1,T+1));
plt.plot(np.cumsum(state_2_time_dist)/np.arange(1,T+1));
plt.plot(np.cumsum(state_3_time_dist)/np.arange(1,T+1));
plt.legend(['State 0','State 1','State 2','State 3']);
plt.title('Policy 1111');
plt.xlabel('Time');
plt.ylabel('State_visitation');