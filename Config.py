class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)
    
    def to_str(self):
        ret = 'seed_{0}_feats_{1}_st_{2}_eps_{3}_gamma_{4}_lambda_{5}_lr_{6}_batch_{7}_randlambda_{8}_walk_{9}'.format(self.seed,
                                                                                                              self.num_features,
                                                                                                              self.num_states,
                                                                                                              self.num_train_episodes,
                                                                                                              self.gamma,
                                                                                                              self.default_lambda,
                                                                                                              self.lr,
                                                                                                              self.batch_size,
                                                                                                              self.random_init_lambda,
                                                                                                              self.walk_type)

        return ret
    
    