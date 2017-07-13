# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:47:07 2017

@author: wsn
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:59:23 2017

@author: wsn
"""
from frozen_lake import FrozenLakeEnv
env = FrozenLakeEnv()
print(env.__doc__)

# Some basic imports and setup
import numpy as np, numpy.random as nr, gym
np.set_printoptions(precision=3)
def begin_grading(): print("\x1b[43m")
def end_grading(): print("\x1b[0m")

# Seed RNGs so you get the same printouts as me
env.seed(0); from gym.spaces import prng; prng.seed(10)
# Generate the episode
env.reset()
for t in range(100):
    env.render()
    a = env.action_space.sample()
    ob, rew, done, _ = env.step(a)
    if done:
        break
assert done
env.render();

class MDP(object):
    def __init__(self, P, nS, nA, desc=None):
        self.P = P # state transition and reward probabilities, explained below
        self.nS = nS # number of states
        self.nA = nA # number of actions
        self.desc = desc # 2D array specifying what each grid cell means (used for plotting)

    
mdp = MDP( {s : {a : [tup[:3] for tup in tups] for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()}, env.nS, env.nA, env.desc)

#print("mdp.P is a two-level dict where the first key is the state and the second key is the action.")
#print("The 2D grid cells are associated with indices [0, 1, 2, ..., 15] from left to right and top to down, as in")
#print(np.arange(16).reshape(4,4))
#print("mdp.P[state][action] is a list of tuples (probability, nextstate, reward).\n")
#print("For example, state 0 is the initial state, and the transition information for s=0, a=0 is \nP[0][0] =", mdp.P[0][0], "\n")
#print("As another example, state 5 corresponds to a hole in the ice, which transitions to itself with probability 1 and reward 0.")
#print("P[5][0] =", mdp.P[5][0], '\n')

def greed_policy(mdp, state, gamma, valuePrev):
    
    """
    Inputs:
    state: scalar, represents single state
    valuePrev: vector <number of states>-by-1
    gamma: scalar, discount factor
    
    Outputs:
    pi:scalar, represents specific action
    value: scalar, represents correspondent value w.r.t pi
    """    
    max_value=0 #max value for givin state
    action=0    #best action to be performed w.r.t givin state
        
    for id in range(mdp.nA):
        temp=[x[0]*(x[2]+gamma*valuePrev[x[1]]) for x in mdp.P[state][id]]
        if np.sum(temp)>max_value: 
            max_value=np.sum(temp)
            action=id 
    return max_value, action
    

def value_iteration(mdp, gamma, nIt):
    """
    Inputs:
        mdp: MDP
        gamma: discount factor
        nIt: number of iterations, corresponding to n above
    Outputs:
        (value_functions, policies)
        
    len(value_functions) == nIt+1 and len(policies) == n
    """
    print("Iteration | max|V-Vprev| | # chg actions | V[0]")
    print("----------+--------------+---------------+---------")
    Vs = [np.zeros(mdp.nS)] # list of value functions contains the initial value function V^{(0)}, which is zero
    pis = []
    
    for it in range(nIt):
        oldpi = pis[-1] if len(pis) > 0 else None # \pi^{(it)} = Greedy[V^{(it-1)}]. Just used for printout
        Vprev = Vs[-1] # V^{(it)}
        
        # YOUR CODE HERE
        # Your code should define the following two variables
        # pi: greedy policy for Vprev, 
        #     corresponding to the math above: \pi^{(it)} = Greedy[V^{(it)}]
        #     numpy array of ints
        # V: bellman backup on Vprev
        #     corresponding to the math above: V^{(it+1)} = T[V^{(it)}]
        #     numpy array of floats

        #V, pi = [greed_policy(mdp, x, gamma, Vprev) for x in range(mdp.nS)]
        V = np.zeros(mdp.nS)
        pi = np.zeros(mdp.nS)       
        
        for idx in range(mdp.nS):
            V[idx],pi[idx] = greed_policy(mdp, idx, gamma, Vprev)
        
        max_diff = np.abs(V - Vprev).max()
        nChgActions="N/A" if oldpi is None else (pi != oldpi).sum()
        print("%4i      | %6.5f      | %4s          | %5.3f"%(it, max_diff, nChgActions, V[0]))
        Vs.append(V)
        pis.append(pi)
    return Vs, pis

GAMMA=0.95 # we'll be using this same value in subsequent problems
begin_grading()
Vs_VI, pis_VI = value_iteration(mdp, gamma=GAMMA, nIt=20)
end_grading()

import matplotlib.pyplot as plt
#matplotlib inline
#for (V, pi) in zip(Vs_VI[:10], pis_VI[:10]):
#    plt.figure(figsize=(3,3))
#    plt.imshow(V.reshape(4,4), cmap='gray', interpolation='none', clim=(0,1))
#    ax = plt.gca()
#    ax.set_xticks(np.arange(4)-.5)
#    ax.set_yticks(np.arange(4)-.5)
#    ax.set_xticklabels([])
#    ax.set_yticklabels([])
#    Y, X = np.mgrid[0:4, 0:4]
#    a2uv = {0: (-1, 0), 1:(0, -1), 2:(1,0), 3:(-1, 0)}
#    Pi = pi.reshape(4,4)
#    for y in range(4):
#        for x in range(4):
#            a = Pi[y, x]
#            u, v = a2uv[a]
#            plt.arrow(x, y,u*.3, -v*.3, color='m', head_width=0.1, head_length=0.1) 
#            plt.text(x, y, str(env.desc[y,x].item().decode()),
#                     color='g', size=12,  verticalalignment='center',
#                     horizontalalignment='center', fontweight='bold')
#    plt.grid(color='b', lw=2, ls='-')
#plt.figure()
#plt.plot(Vs_VI)
#plt.title("Values of different states");

def compute_vpi(pi, mdp, gamma):
    # YOUR CODE HERE
    """
    State-value function make convefence of action change very fast,
    because the update of V^pi(s) for all s doesn't use the previous V(s)
    """    
    coeff=[]
    b=[]
    
    for idx in range(mdp.nS):
        #Construct coefficient matrix
        coeff_temp=np.zeros(mdp.nS)
        b_temp=np.zeros(mdp.nS)
        coeff_temp[idx]=1
                
        l=len(mdp.P[idx][pi[idx]])
        for idx2 in range(l):
            s_prime=mdp.P[idx][pi[idx]][idx2]
            coeff_temp[s_prime[1]]=-s_prime[0]*gamma+coeff_temp[s_prime[1]]
                
        b_temp=np.sum([x[0]*x[2] for x in mdp.P[idx][pi[idx]]])
        coeff.append(coeff_temp)
        b.append(b_temp)
       
    V=np.linalg.solve(coeff, b)
    return V

#GAMMA=0.95 # we'll be using this same value in subsequent problems
#begin_grading()
#print(compute_vpi(np.ones(16), mdp, gamma=GAMMA))
#end_grading()
   
Vpi=compute_vpi(pis_VI[15], mdp, gamma=GAMMA)
V_vi = Vs_VI[15]
print("From compute_vpi", Vpi)
print("From value iteration", V_vi)
print("Difference", Vpi - V_vi)    

def unit_func(vpi, mdp, state, action, gamma):
    #Calculate Q_pi()
    q_v=sum([x[0]*(x[2]+gamma*vpi[x[1]]) for x in mdp.P[state][action]])
    return q_v
        
def compute_qpi(vpi, mdp,  gamma):
    # YOUR CODE HERE
    Qpi=np.zeros((mdp.nS, mdp.nA))
    Qpi=[[unit_func(vpi, mdp, x, y, gamma) for y in range(mdp.nA)] for x in range(mdp.nS)]
    return Qpi

begin_grading()
Qpi = compute_qpi(np.arange(mdp.nS), mdp, gamma=0.95)
print("Qpi:\n", Qpi)
end_grading()


def policy_iteration(mdp, gamma, nIt):
    """
    Fast than value iteration, also be impacted with the value of gamma,
    RL can only be used in cases that the value of other states matters, 
    otherwise the value of other states doesn't help with updating value 
    of current state.
    """
    Vs = []
    pis = []
    pi_prev = np.zeros(mdp.nS,dtype='int')
    pis.append(pi_prev)
    print("Iteration | # chg actions | V[0]")
    print("----------+---------------+---------")
    for it in range(nIt):        
        vpi = compute_vpi(pi_prev, mdp, gamma)
        qpi = compute_qpi(vpi, mdp, gamma)
        pi = np.argmax(qpi, axis=1)
        print("%4i      | %6i        | %6.5f"%(it, (pi != pi_prev).sum(), vpi[0]))
        Vs.append(vpi)
        pis.append(pi)
        pi_prev = pi
    return Vs, pis
Vs_PI, pis_PI = policy_iteration(mdp, gamma=0.95, nIt=20)
plt.plot(Vs_PI);

for s in range(5):
    plt.figure()
    plt.plot(np.array(Vs_VI)[:,s])
    plt.plot(np.array(Vs_PI)[:,s])
    plt.ylabel("value of state %i"%s)
    plt.xlabel("iteration")
    plt.legend(["value iteration", "policy iteration"], loc='best')

