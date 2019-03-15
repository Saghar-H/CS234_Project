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


done = False
seed = 1358
env_name = 'WalkFiveStates-v0'
env = init_env(env_name, seed)
num_features = 5
num_states = 5
num_episodes = 80

gamma = 0.95
lambda_ = 0.5

# One hot vector representations:
Phi = np.eye(num_states)



G = []
transition_probs = env.env.P

# LSTD operator:
LSTD_lambda = LSTD(num_features, epsilon=0.0001)

loss = []
for ep in range(num_episodes):
    env.reset()
    ep_rewards = []
    ep_states = []
    state, reward, done, info = env.step(np.random.randint(env.action_space.n))
    episode_loss = 0
    timestep = 1

    while not done:
        #env.render()
        ep_rewards.append(reward)
        ep_states.append(state)
        state_next, reward_next, done, info = env.step(np.random.randint(env.action_space.n))
        LSTD_lambda.update(Phi[state-1,:], reward, Phi[state_next-1,:], gamma, lambda_, timestep)
        theta = LSTD_lambda.theta
        #print("A is: {0}".format(LSTD_lambda.A))
        print("b is: {0}".format(LSTD_lambda.b))
        #print("z is: {0}".format(LSTD_lambda.z))
        print("Theta is: {0}".format(theta))
        #print("State is: {0}".format(state))
        state = state_next
        reward = reward_next
        timestep += 1
    ep_rewards.append(reward)
    ep_states.append(state)
    ep_discountedrewards = get_discounted_return(ep_rewards, gamma)
    ep_loss = np.mean([(np.dot(Phi[ep_states[t]-1,:], theta) - ep_discountedrewards[t])**2 for t in range(len(ep_states))])
    #print('Episode {0} loss is {1}'.format(ep, ep_loss))
    #print('Episode {0} rewards are {1}'.format(ep, ep_rewards))
    G.append(ep_discountedrewards)
    loss.append(ep_loss)
    

print('episode loss:{0}'.format(loss))
print ("average loss: ", sum(loss) / num_episodes)
print("#########")


