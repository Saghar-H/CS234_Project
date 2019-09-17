import numpy as np
from autograd_cls import AutoGrad 
from compute_utils import compute_lcv_lambda_gradient, compute_epsilon_lambda_gradient, compute_hjj, compute_z, compute_z_gradient, compute_eps_t, compute_hjj_gradient, get_discounted_return, calculate_batch_loss
from lstd import LSTD, MiniBatchLSTDLambda
from adam import ADAM
import copy
import pdb


def minibatch_LSTD(trajectories, Phi, num_features, gamma=0.4, lambda_=0.2):
    LSTD_lambda = MiniBatchLSTDLambda(gamma, lambda_, Phi)
    G = {}
    running_loss = []
    num_episodes = len(trajectories)
    for ep in range(num_episodes):
        G[ep] = []
        traj = trajectories[ep]
        ep_rewards = []
        ep_states = []
        cur_state, reward, next_state, done = traj[0]
        LSTD_lambda.update(None, 0 , cur_state)
        cur_state = next_state
        #LSTD_lambda.reset_boyan(Phi[cur_state, :])
        for timestep in range(len(traj)):
            cur_state, reward, next_state, done = traj[timestep]
            LSTD_lambda.update(cur_state, reward, next_state)
            ep_rewards.append(reward)
            ep_states.append(cur_state)
            if done:
                LSTD_lambda.update(next_state, 0, None)
        theta = LSTD_lambda.theta
        ep_discountedrewards = get_discounted_return(ep_rewards, gamma)
        # print('ep_discounted:{0}'.format(ep_discountedrewards))
        if len(ep_discountedrewards) > 0:
            ep_loss = np.mean(
                [(np.dot(Phi[ep_states[t], :], theta) - ep_discountedrewards[t]) ** 2 for t in range(len(ep_states))])

            G[ep] = ep_discountedrewards
            running_loss.append(ep_loss)
    # After we calculated the Theta parameter from the training data
    loss, rmse = calculate_batch_loss(trajectories, G, theta, Phi)

    average_loss = rmse
    return LSTD_lambda, theta, average_loss, G

def LSTD_algorithm(trajectories, Phi, num_features, gamma=0.4, lambda_=0.2):
    # LSTD operator:
    LSTD_lambda = LSTD(num_features)
    G = {}
    running_loss = []
    num_episodes = len(trajectories)
    for ep in range(num_episodes):
        G[ep] = []
        traj = trajectories[ep]
        if len(traj) <= 4:
            continue
        ep_rewards = []
        ep_states = []
        cur_state = traj[0][0]
        LSTD_lambda.reset_boyan(Phi[cur_state, :])
        for timestep in range(len(traj)):
            cur_state, reward, next_state, done = traj[timestep]
            LSTD_lambda.update_boyan(Phi[cur_state, :], reward, Phi[next_state, :], gamma, lambda_, timestep)
            ep_rewards.append(reward)
            ep_states.append(cur_state)
        theta = LSTD_lambda.theta
        ep_discountedrewards = get_discounted_return(ep_rewards, gamma)
        # print('ep_discounted:{0}'.format(ep_discountedrewards))
        if len(ep_discountedrewards) > 0:
            ep_loss = np.mean(
                [(np.dot(Phi[ep_states[t], :], theta) - ep_discountedrewards[t]) ** 2 for t in range(len(ep_states))])
            # print('Episode {0} loss is {1}'.format(ep, ep_loss))
            # print('Episode {0} rewards are {1}'.format(ep, ep_rewards))
            G[ep] = ep_discountedrewards
            running_loss.append(ep_loss)
    # After we calculated the Theta parameter from the training data
    loss, rmse = calculate_batch_loss(trajectories, G, theta, Phi)
    # print('episode loss:{0}'.format(loss))
    # print(LSTD_lambda.A, LSTD_lambda.b)

    # print("average running loss in training: ", sum(running_loss) / num_episodes)
    # print("average loss after training: ", sum(loss) / num_episodes)
    average_loss = rmse
    return LSTD_lambda, theta, average_loss, G


def Adaptive_LSTD_algorithm(trajectories, 
                            Phi, 
                            P, 
                            V, 
                            D, 
                            R, 
                            Gs, 
                            config
                           ):
    # LSTD operator:
    Auto_grad = AutoGrad(compute_CV_loss, 4)
    Auto_grad.gradient_fun()
    adaptive_LSTD_lambda = LSTD(config.num_features)
    G = {}
    loss = []
    running_loss = []
    num_episodes = len(trajectories)
    adam_optimizer = ADAM(x_init = config.default_lambda, alpha=config.lr)
    lambda_ = config.default_lambda
    
    for ep in range(config.num_episodes):
        G[ep] = []
        traj = trajectories[ep]
        if len(traj) <= 4:
            continue
        ep_rewards = []
        ep_states = []
        Z = np.zeros((config.num_features, config.num_states))
        Z_gradient = np.zeros((config.num_features, config.num_states))
        H_diag = np.zeros(config.num_states) # n 
        eps = np.zeros(config.num_states)
        states_count = np.zeros(config.num_states)
        epsilon_lambda_gradient = np.zeros(config.num_states)
        H_diag_gradient = np.zeros(config.num_states)
        episode_loss = 0
        cur_state = traj[0][0]
        adaptive_LSTD_lambda.reset_boyan(Phi[cur_state, :])

        for timestep in range(len(traj)):

            cur_state, reward, next_state, done = traj[timestep]
            adaptive_LSTD_lambda.update_boyan(Phi[cur_state, :], reward, Phi[next_state, :], config.gamma, lambda_, timestep)
            ep_rewards.append(reward)
            ep_states.append(cur_state)
        theta = adaptive_LSTD_lambda.theta
        print(theta)
        A = adaptive_LSTD_lambda.A
        b = adaptive_LSTD_lambda.b
        A_inv = np.linalg.pinv(A + np.eye(A.shape[0]) * config.A_inv_epsilon, rcond=.1)
        
        for timestep in range(len(traj)-1):
            cur_state, reward, next_state, done = traj[timestep]
            # To-do : change the following update to running average
            states_count[cur_state] += 1
            ct = states_count[cur_state]
            Z[:,cur_state] = (ct-1)/ct *Z[:,cur_state]+ 1/ct * compute_z(lambda_, config.gamma, Phi, ep_states, timestep )
            Z_gradient[:, cur_state] = (ct-1)/ct * Z_gradient[:, cur_state] + 1/ct * compute_z_gradient(lambda_, config.gamma, Phi, ep_states, timestep)            
            H_diag[cur_state] = (ct-1)/ct * H_diag[cur_state] + 1/ct * compute_hjj(Phi, lambda_, config.gamma, ep_states, timestep, A_inv)
            eps[cur_state] = (ct-1)/ct * eps[cur_state] + 1/ct * compute_eps_t(Phi, theta, config.gamma, reward, ep_states, timestep)
            
            epsilon_lambda_gradient[cur_state] = (ct-1)/ct * epsilon_lambda_gradient[cur_state] + \
                                                1/ct * compute_epsilon_lambda_gradient(Phi,
                                                                                       lambda_, 
                                                                                       config.gamma,
                                                                                       A, 
                                                                                       b,  
                                                                                       A_inv, 
                                                                                       Z, 
                                                                                       timestep, 
                                                                                       ep_states, 
                                                                                       ep_rewards
                                                                                      )
            
            H_diag_gradient[cur_state] = (ct-1)/ct * H_diag_gradient[cur_state] + 1/ct * compute_hjj_gradient(Phi, 
                                                                                                              lambda_, 
                                                                                                              config.gamma, 
                                                                                                              ep_states, 
                                                                                                              timestep, 
                                                                                                              A, 
                                                                                                              b,  
                                                                                                              A_inv
                                                                                                             )
        #grad = compute_cv_gradient(Phi, theta, gamma, lambda_, P, V, D, R)
        # Replaced the above update with:
        grad = compute_lcv_lambda_gradient(eps, 
                                           H_diag, 
                                           ep_states, 
                                           epsilon_lambda_gradient, 
                                           H_diag_gradient,
                                           grad_clip_max_norm = config.grad_clip_norm)
        if config.compute_autograd:    
            auto_grad = Auto_grad.loss_autograd_fun(trajectories, Phi, config.num_features, config.gamma, lambda_, Gs)
            print('gradient diff:{0}'.format(abs(grad-auto_grad)))

        # if ep > 1000 :
        # new_lambda = lambda_ -  lr * compute_cv_gradient(Phi, theta, gamma, lambda_, P, V, D)
        # print(new_lambda)
        # if new_lambda >= 0 and new_lambda <= 1:
        #   lambda_ = new_lambda
        #   print('current lambda:{0}'.format(lambda_))
        
        # grad = compute_cv_gradient2(Phi, theta, gamma, lambda_, R, A, b, z)
        if config.use_adam_optimizer:
            adam_optimizer.update(grad, ep)
            new_lambda = adam_optimizer.x
        else:
            new_lambda = lambda_ - config.lr * grad
        if new_lambda >= 0 and new_lambda <= 1:
            lambda_ = new_lambda
            print('current lambda:{0}'.format(lambda_))
        ep_discountedrewards = get_discounted_return(ep_rewards, config.gamma)
        # print('ep_discounted:{0}'.format(ep_discountedrewards))
        if len(ep_discountedrewards) > 0:
            ep_loss = np.mean(
                [(np.dot(Phi[ep_states[t], :], theta) - ep_discountedrewards[t]) ** 2 for t in range(len(ep_states))])
            # print('Episode {0} loss is {1}'.format(ep, ep_loss))
            # print('Episode {0} rewards are {1}'.format(ep, ep_rewards))
            G[ep] = ep_discountedrewards
            running_loss.append(ep_loss)
    # After we calculated the Theta parameter from the training data
    loss, rmse = calculate_batch_loss(trajectories, G, theta, Phi)
    # print('episode loss:{0}'.format(loss))
    # print(LSTD_lambda.A, LSTD_lambda.b)
    #print("Final Lambda: {0}".format(lambda_))
    #print("average running loss in training: ", np.mean(running_loss))
    #print("average loss after training: ", np.mean(loss))
    average_loss = rmse
    return adaptive_LSTD_lambda, theta, average_loss, G, lambda_


def Adaptive_LSTD_algorithm_batch(trajectories, 
                                    Phi, 
                                    P, 
                                    V, 
                                    D, 
                                    R, 
                                    Gs,
                                    logger, 
                                    config
                                 ):
    # LSTD operator:

    Auto_grad = AutoGrad(compute_CV_loss, 4)
    Auto_grad.gradient_fun()
    #adaptive_LSTD_lambda = LSTD(config.num_features)
    adaptive_LSTD_lambda = MiniBatchLSTDLambda(config.gamma, config.default_lambda, Phi)
    G = {}
    loss = []
    running_loss = []
    num_episodes = len(trajectories)
    adam_optimizer = ADAM(x_init = config.default_lambda, alpha=config.lr)
    lambda_ = config.default_lambda
    valid_episode_counter = 0
    for ep in range(config.num_episodes):
        traj = trajectories[ep]
        G[ep] = [] 
        if len(traj) <= 4:
            continue       
        cur_state, reward, next_state, done = traj[0]
        adaptive_LSTD_lambda.update(None, 0 , cur_state) 
        if valid_episode_counter % config.batch_size == 0:           
            ep_rewards = []
            ep_states = []
            Z = np.zeros((config.num_features, config.num_states))
            Z_gradient = np.zeros((config.num_features, config.num_states))
            H_diag = np.zeros(config.num_states) # n 
            eps = np.zeros(config.num_states)
            states_count = np.zeros(config.num_states)
            epsilon_lambda_gradient = np.zeros(config.num_states)
            H_diag_gradient = np.zeros(config.num_states)
            episode_loss = 0
            
        #cur_state = traj[0][0]
        #adaptive_LSTD_lambda.reset_boyan(Phi[cur_state, :])

        for timestep in range(len(traj)):
            cur_state, reward, next_state, done = traj[timestep]
            adaptive_LSTD_lambda.update(cur_state, reward, next_state)
            if done:
                adaptive_LSTD_lambda.update(next_state, 0, None)
            #adaptive_LSTD_lambda.update_boyan(Phi[cur_state, :], reward, Phi[next_state, :], config.gamma, lambda_, timestep)
            ep_rewards.append(reward)
            ep_states.append(cur_state)
        if logger:
            logger.log_scalar('average trajectories reward', np.mean(ep_rewards),  valid_episode_counter)
            logger.writer.flush()

        theta = adaptive_LSTD_lambda.theta
        A = adaptive_LSTD_lambda.A
        b = adaptive_LSTD_lambda.b.reshape((-1,1))
        A_inv = np.linalg.pinv(A + np.eye(A.shape[0]) * config.A_inv_epsilon, rcond=.1)
        
        for timestep in range(len(traj)-1):
            cur_state, reward, next_state, done = traj[timestep]
            states_count[cur_state] += 1
            ct = states_count[cur_state]
            Z[:,cur_state] = (ct-1)/ct *Z[:,cur_state]+ 1/ct * compute_z(lambda_, config.gamma, Phi, ep_states, timestep )
            Z_gradient[:, cur_state] = (ct-1)/ct * Z_gradient[:, cur_state] + \
                                        1/ct * compute_z_gradient(lambda_, config.gamma, Phi, ep_states, timestep)      
            
            H_diag[cur_state] = (ct-1)/ct * H_diag[cur_state] + \
                                1/ct * compute_hjj(Phi, lambda_, config.gamma, ep_states, timestep, A_inv)
            
            eps[cur_state] = (ct-1)/ct * eps[cur_state] + \
                             1/ct * compute_eps_t(Phi, theta, config.gamma, reward, ep_states, timestep)

            epsilon_lambda_gradient[cur_state] = (ct-1)/ct * epsilon_lambda_gradient[cur_state] + \
                                                1/ct * compute_epsilon_lambda_gradient(Phi,
                                                                                       lambda_, 
                                                                                       config.gamma,
                                                                                       A, 
                                                                                       b,  
                                                                                       A_inv, 
                                                                                       Z, 
                                                                                       timestep, 
                                                                                       ep_states, 
                                                                                       ep_rewards
                                                                                      )
            
            H_diag_gradient[cur_state] = (ct-1)/ct * H_diag_gradient[cur_state] + 1/ct * compute_hjj_gradient(Phi, 
                                                                                                              lambda_, 
                                                                                                              config.gamma, 
                                                                                                              ep_states, 
                                                                                                              timestep, 
                                                                                                              A, 
                                                                                                              b,  
                                                                                                              A_inv
                                                                                                             )
        # update the gradients of the batch:
        if valid_episode_counter % config.batch_size == 0:
            grad = compute_lcv_lambda_gradient(eps, 
                                               H_diag, 
                                               ep_states, 
                                               epsilon_lambda_gradient, 
                                               H_diag_gradient,
                                               grad_clip_max_norm = config.grad_clip_norm)
            if logger:
                logger.log_scalar('CV loss lambda gradients per batch', grad,  valid_episode_counter/config.batch_size)
                logger.writer.flush()

            if config.compute_autograd:    
                auto_grad = Auto_grad.loss_autograd_fun(trajectories, Phi, config.num_features, config.gamma, lambda_, Gs)
                print('gradient diff:{0}'.format(abs(grad-auto_grad)))

            if config.use_adam_optimizer:
                adam_optimizer.update(grad, valid_episode_counter+1)
                new_lambda = adam_optimizer.x
            else:
                new_lambda = lambda_ - config.lr * grad
            if new_lambda >= 0 and new_lambda <= 1:
                lambda_ = new_lambda
                print('gradient: {0}'.format(grad))
                print('current lambda:{0}'.format(lambda_))
                print('current theta:{0}'.format(theta))
            ep_discountedrewards = get_discounted_return(ep_rewards, config.gamma)
            # print('ep_discounted:{0}'.format(ep_discountedrewards))
            if len(ep_discountedrewards) > 0:
                ep_loss = np.mean(
                    [(np.dot(Phi[ep_states[t], :], theta) - ep_discountedrewards[t]) ** 2 for t in range(len(ep_states))])
                G[ep] = ep_discountedrewards
                running_loss.append(ep_loss)
        valid_episode_counter += 1
    # After we calculated the Theta parameter from the training data
    loss, rmse = calculate_batch_loss(trajectories, G, theta, Phi)
    print('Theta values: {0}'.format(theta))
    # print('episode loss:{0}'.format(loss))
    # print(LSTD_lambda.A, LSTD_lambda.b)
    #print("Final Lambda: {0}".format(lambda_))
    #print("average running loss in training: ", np.mean(running_loss))
    #print("average loss after training: ", np.mean(loss))
    return adaptive_LSTD_lambda, theta, rmse, G, lambda_


'''
The same as Adaptive_LSTD_algorithm_batch, except A and b
are calculated based on all the episodes.
'''
def Adaptive_LSTD_algorithm_batch_type2(trajectories, 
                                        Phi, 
                                        P, 
                                        V, 
                                        D, 
                                        R, 
                                        Gs,
                                        logger, 
                                        config
                                     ):
    # LSTD operator:

    Auto_grad = AutoGrad(compute_CV_loss, 4)
    Auto_grad.gradient_fun()
    adaptive_LSTD_lambda = LSTD(config.num_features)
    G = {}
    loss = []
    running_loss = []
    num_episodes = len(trajectories)
    adam_optimizer = ADAM(x_init = config.default_lambda, alpha=config.lr)
    lambda_ = config.default_lambda
    ##### First go over all the trajectories and calculate estimate A and b:
    for ep in range(config.num_episodes):
        traj = trajectories[ep]
        if len(traj) <= 4:
            continue 
        cur_state = traj[0][0]
        adaptive_LSTD_lambda.reset_boyan(Phi[cur_state, :])
        for timestep in range(len(traj)):
            cur_state, reward, next_state, done = traj[timestep]
            adaptive_LSTD_lambda.update_boyan(Phi[cur_state, :], 
                                              reward, 
                                              Phi[next_state, :], 
                                              config.gamma, 
                                              lambda_, 
                                              timestep
                                             )
        
#     theta = adaptive_LSTD_lambda.theta
#     A = adaptive_LSTD_lambda.A
#     b = adaptive_LSTD_lambda.b
#     A_inv = np.linalg.pinv(A + np.eye(A.shape[0]) * config.A_inv_epsilon)
    #pudb.set_trace()
 ######## Now use the above A and b to calculate optimal lambda:   
    valid_episode_counter = 0    
    for ep in range(config.num_episodes):
        traj = trajectories[ep]
        G[ep] = [] 
        if len(traj) <= 4:
            continue
        if valid_episode_counter % config.batch_size == 0:           
            ep_rewards = []
            ep_states = []
            Z = np.zeros((config.num_features, config.num_states))
            Z_gradient = np.zeros((config.num_features, config.num_states))
            H_diag = np.zeros(config.num_states) # n 
            eps = np.zeros(config.num_states)
            states_count = np.zeros(config.num_states)
            epsilon_lambda_gradient = np.zeros(config.num_states)
            H_diag_gradient = np.zeros(config.num_states)
            episode_loss = 0
        
        cur_state = traj[0][0]
        
        adaptive_LSTD_lambda.reset_boyan(Phi[cur_state, :])

        for timestep in range(len(traj)):

            cur_state, reward, next_state, done = traj[timestep]
            adaptive_LSTD_lambda.update_boyan(Phi[cur_state, :], reward, Phi[next_state, :], config.gamma, lambda_, timestep)
            ep_rewards.append(reward)
            ep_states.append(cur_state)
        #pudb.set_trace()
        theta = adaptive_LSTD_lambda.theta
        A = adaptive_LSTD_lambda.A
        b = adaptive_LSTD_lambda.b
        A_inv = np.linalg.pinv(A + np.eye(A.shape[0]) * config.A_inv_epsilon, rcond=.1)

        
        for timestep in range(len(traj)-1):
            cur_state, reward, next_state, done = traj[timestep]
            states_count[cur_state] += 1
            ct = states_count[cur_state]
            Z[:,cur_state] = (ct-1)/ct *Z[:,cur_state]+ 1/ct * compute_z(lambda_, config.gamma, Phi, ep_states, timestep )
            Z_gradient[:, cur_state] = (ct-1)/ct * Z_gradient[:, cur_state] + \
                                        1/ct * compute_z_gradient(lambda_, config.gamma, Phi, ep_states, timestep)      
            
            H_diag[cur_state] = (ct-1)/ct * H_diag[cur_state] + \
                                1/ct * compute_hjj(Phi, lambda_, config.gamma, ep_states, timestep, A_inv)
            
            eps[cur_state] = (ct-1)/ct * eps[cur_state] + \
                             1/ct * compute_eps_t(Phi, theta, config.gamma, reward, ep_states, timestep)
            
            epsilon_lambda_gradient[cur_state] = (ct-1)/ct * epsilon_lambda_gradient[cur_state] + \
                                                1/ct * compute_epsilon_lambda_gradient(Phi,
                                                                                       lambda_, 
                                                                                       config.gamma,
                                                                                       A, 
                                                                                       b,  
                                                                                       A_inv, 
                                                                                       Z, 
                                                                                       timestep, 
                                                                                       ep_states, 
                                                                                       ep_rewards
                                                                                      )
            
            H_diag_gradient[cur_state] = (ct-1)/ct * H_diag_gradient[cur_state] + 1/ct * compute_hjj_gradient(Phi, 
                                                                                                              lambda_, 
                                                                                                              config.gamma, 
                                                                                                              ep_states, 
                                                                                                              timestep, 
                                                                                                              A, 
                                                                                                              b,  
                                                                                                              A_inv
                                                                                                             )
        # update the gradients of the batch:
        if valid_episode_counter % config.batch_size == 0:
            grad = compute_lcv_lambda_gradient(eps, 
                                               H_diag, 
                                               ep_states, 
                                               epsilon_lambda_gradient, 
                                               H_diag_gradient,
                                               grad_clip_max_norm = config.grad_clip_norm)
            if config.compute_autograd:    
                auto_grad = Auto_grad.loss_autograd_fun(trajectories, Phi, config.num_features, config.gamma, lambda_, Gs)
                print('gradient diff:{0}'.format(abs(grad-auto_grad)))

            if config.use_adam_optimizer:
                adam_optimizer.update(grad, valid_episode_counter+1)
                new_lambda = adam_optimizer.x
            else:
                new_lambda = lambda_ - config.lr * grad
            #pudb.set_trace()
            if new_lambda >= 0 and new_lambda <= 1:
                lambda_ = new_lambda
                print('gradient: {0}'.format(grad))
                print('current lambda:{0}'.format(lambda_))
            ep_discountedrewards = get_discounted_return(ep_rewards, config.gamma)
            # print('ep_discounted:{0}'.format(ep_discountedrewards))
            if len(ep_discountedrewards) > 0:
                ep_loss = np.mean(
                    [(np.dot(Phi[ep_states[t], :], theta) - ep_discountedrewards[t]) ** 2 for t in range(len(ep_states))])
                G[ep] = ep_discountedrewards
                running_loss.append(ep_loss)
        valid_episode_counter += 1
    # After we calculated the Theta parameter from the training data
    loss, rmse = calculate_batch_loss(trajectories, G, theta, Phi)
    # print('episode loss:{0}'.format(loss))
    # print(LSTD_lambda.A, LSTD_lambda.b)
    #print("Final Lambda: {0}".format(lambda_))
    #print("average running loss in training: ", np.mean(running_loss))
    #print("average loss after training: ", np.mean(loss))
    return adaptive_LSTD_lambda, theta, rmse, G, lambda_

'''
This needs to be here to avoid having circular dependencies.
'''
def compute_CV_loss(trajectories, 
                    Phi, 
                    num_features, 
                    gamma, 
                    lambda_, 
                    Gs, 
                    logger=False,
                    epsilon=0.0):
    '''
    :param trajectories:
    :param num_features:
    :param gamma:
    :param epsilon:
    :return:
    '''
    total_num_tuples = sum([len(traj) for traj in trajectories])
    num_episodes = len(trajectories)
    loto_loss = []
    step = 0
    for i in range(min(1000,num_episodes)):
        traj = trajectories[i]
        if len(traj) <= 4:
            continue
        for j in range(len(traj)):
            # leave one tuple oto_trajectoriesout
            loto_trajectories = copy.deepcopy(trajectories)
            del loto_trajectories[i][j]
            model, _, loss, _ = LSTD_algorithm(loto_trajectories, Phi, num_features, gamma, lambda_)
            theta = model.theta
            #theta = [-24, -16, -8,0]
            # pdb.set_trace()
            tuple_loss = (np.dot(Phi[trajectories[i][j][0], :], theta) - Gs[i][j]) ** 2
            loto_loss.append(tuple_loss)
            
        if logger:
            logger.log_scalar('average trajectories loss', loss, step)
            logger.log_scalar('current tuple loto cv', tuple_loss, step)
            logger.log_scalar('mean loto cv', np.mean(loto_loss)**.5, step)
            logger.writer.flush()
            step += 1

        print('trajectory :{0}, current mean loto loss:{1}'.format(i, np.mean(loto_loss)**.5))
    cv_loss = np.mean(loto_loss) 
    return cv_loss ** 0.5