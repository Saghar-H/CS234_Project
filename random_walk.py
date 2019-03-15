import gym
import gym_walk
import pdb
import numpy as np
import random
from lstd import LSTD
from pprint import pprint


def get_discounted_return(episode_rewards, gamma):
    discounted_rewards = [0] * (len(episode_rewards) + 1)
    for i in range(len(episode_rewards)-1,-1,-1):
        discounted_rewards[i] = discounted_rewards[i+1] * gamma + episode_rewards[i]
    return discounted_rewards[:-1]
    
def init_env(env_name, seed):
    env = gym.make(env_name)
    env.reset()
    random.seed(seed)
    env.seed(seed)
    return env

def compute_P(transition_probs, num_actions, num_states):
    ret = np.zeros((num_states, num_states))
    for s in transition_probs.keys():
        for a in transition_probs[s]:
            for tup in transition_probs[s][a]:
                sp = tup[1]
                p_sasp = tup[0]
                ret[s,sp] += 1.0/num_actions * p_sasp

    return ret

done = False
seed = 1358
env_name = 'WalkFiveStates-v0'
env = init_env(env_name, seed)
num_features = 5
num_states = 5
num_episodes = 80

gamma = 0.95
lambda_ = 0.5

transition_probs = env.env.P

# One hot vector representations:
Phi = np.eye(num_states)

'''
Computes monte carlo estimates of D and V:
'''
# def run_env_episodes(num_episodes):
#     D = np.zeros(env.observation_space.n)
#     V = np.zeros(env.observation_space.n)

#     total_steps = 0
#     for ep in range(num_episodes):
#         env.reset()
#         done = False
#         ep_rewards = []
#         ep_states = []
#         while not done:
#             #env.render()
#             state, reward, done, info = env.step(random.randint(0, env.action_space.n - 1))
#             D[state] +=1
#             total_steps +=1
#             ep_rewards.append(reward)
#             ep_states.append(state)


#         ep_discountedrewards = get_discounted_rewards(ep_rewards, gamma)
#         for i in range(len(ep_states)):
#             V[ep_states[i]] += ep_discountedrewards[i]

#     print('Monte Carlo D:{0}'.format(D / total_steps, total_steps))
#     print('Monte Carlo V:{0}'.format(V / D))
#     return np.diag(D / total_steps), V/D

# print('Generate Monte Carlo Estimates of D and V...')
# D,V = run_env_episodes(100000)
# print('Done finding D and V!')

D = np.diag([0.12443139 ,0.24981192 ,0.25088312, 0.25018808 ,0.12468549])
V = np.array([0, 0.01776151, 0.071083, 0.26708894 ,1])

'''
Now compute the MRP value of P: P(s'|s)
'''
P = compute_P(transition_probs, env.action_space.n, env.observation_space.n)

print('starting the main loop...')

G = []


# LSTD operator:
LSTD_lambda = LSTD(num_features, epsilon=0.0)

loss = []
for ep in range(num_episodes):
    env.reset()
    ep_rewards = []
    ep_states = []
    state, reward, done, info = env.step(np.random.randint(env.action_space.n))
    episode_loss = 0
    timestep = 1
    LSTD_lambda.reset()
    while not done:
        #env.render()
        ep_rewards.append(reward)
        ep_states.append(state)
        state_next, reward_next, done, info = env.step(np.random.randint(env.action_space.n))
        LSTD_lambda.update(Phi[state,:], reward, Phi[state_next,:], gamma, lambda_, timestep)
        theta = LSTD_lambda.theta
        #print("A is: {0}".format(LSTD_lambda.A))
        #print("b is: {0}".format(LSTD_lambda.b))
        #print("z is: {0}".format(LSTD_lambda.z))
        #print("Theta is: {0}".format(theta))
        #print("State is: {0}".format(state))
        state = state_next
        reward = reward_next
        timestep += 1
    ep_rewards.append(reward)
    ep_states.append(state)
    LSTD_lambda.update(Phi[state,:], reward, Phi[state,:], gamma, lambda_, timestep)
    ep_discountedrewards = get_discounted_return(ep_rewards, gamma)
    ep_loss = np.mean([(np.dot(Phi[ep_states[t],:], theta) - ep_discountedrewards[t])**2 for t in range(len(ep_states))])
    #print('states:{0}'.format(ep_states))
    #print('Episode {0} loss is {1}'.format(ep, ep_loss))
    #print('Episode {0} rewards are {1}'.format(ep, ep_rewards))
    G.append(ep_discountedrewards)
    loss.append(ep_loss)
    

print('episode loss:{0}'.format(loss))
print ("average loss: ", sum(loss) / num_episodes)
print("#########")


