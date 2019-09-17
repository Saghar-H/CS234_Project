class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)
    
    def to_str(self):
        ret = 'seed_{0}_feats_{1}_st_{2}_eps_{3}_gamma_{4}_lambda_{5}_lr_{6}_batch_{7}_randlambda_{8}'.format(self.seed,
                                                                                                              self.num_features,
                                                                                                              self.num_states,
                                                                                                              self.num_episodes,
                                                                                                              self.gamma,
                                                                                                              self.default_lambda,
                                                                                                              self.lr,
                                                                                                              self.batch_size,
                                                                                                              self.random_init_lambda)

        return ret
    
    
    
        seed = 1357,
    #env_name = 'WalkFiveStates-v0',
    env_name = 'Boyan',
    num_features = 4,
    num_states = 13,
    num_episodes = 20,
    A_inv_epsilon = 1e-3,
    gamma = 1.0,
    default_lambda = 0.75,
    lr = .2,
    use_adaptive_lambda = True,
    grad_clip_norm = 10,
    compute_autograd = False,
    use_adam_optimizer = True,
    batch_size = 4,
    upsampling_rate = 1,
    step_size_gamma = 0.1,
    step_size_lambda = 0.05,
    seed_iterations=5, 
    seed_step_size=5, 
    random_init_lambda = False,