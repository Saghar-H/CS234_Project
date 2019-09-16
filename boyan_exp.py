from mdp_env.blackhc import mdp

class BOYAN_MDP:
    def __init__(self, file_name):
            self.file_name = file_name
            self.spec = None
            self.env = None
            self.boyan_markov_chain_env()
            if not file_name is None:
                self.save_mdp_to_png()

    def save_mdp_to_png(self):
        mdp_spec = self.spec
#         with open(self.file_name, 'wb') as file:
#             file.write(mdp.graph_to_png(mdp_spec.to_graph()))
#        mdp.display_mdp(mdp_spec)

    def boyan_markov_chain_env(self):
        spec = mdp.MDPSpec()
        s_12 = spec.state('12')
        s_11 = spec.state('11')
        s_10 = spec.state('10')
        s_9 = spec.state('9')
        s_8 = spec.state('8')
        s_7 = spec.state('7')
        s_6 = spec.state('6')
        s_5 = spec.state('5')
        s_4 = spec.state('4')
        s_3 = spec.state('3')
        s_2 = spec.state('2')
        s_1 = spec.state('1')
        s_0 = spec.state('0', terminal_state=True)

        action_0 = spec.action()
        action_1 = spec.action()

        spec.transition(s_12, action_0, mdp.NextState(s_11))
        spec.transition(s_12, action_0, mdp.Reward(-3.0))
        spec.transition(s_12, action_1, mdp.NextState(s_10))
        spec.transition(s_12, action_1, mdp.Reward(-3.0))

        spec.transition(s_11, action_0, mdp.NextState(s_10))
        spec.transition(s_11, action_0, mdp.Reward(-3.0))
        spec.transition(s_11, action_1, mdp.NextState(s_9))
        spec.transition(s_11, action_1, mdp.Reward(-3.0))

        spec.transition(s_10, action_0, mdp.NextState(s_9))
        spec.transition(s_10, action_0, mdp.Reward(-3.0))
        spec.transition(s_10, action_1, mdp.NextState(s_8))
        spec.transition(s_10, action_1, mdp.Reward(-3.0))

        spec.transition(s_9, action_0, mdp.NextState(s_8))
        spec.transition(s_9, action_0, mdp.Reward(-3.0))
        spec.transition(s_9, action_1, mdp.NextState(s_7))
        spec.transition(s_9, action_1, mdp.Reward(-3.0))

        spec.transition(s_8, action_0, mdp.NextState(s_7))
        spec.transition(s_8, action_0, mdp.Reward(-3.0))
        spec.transition(s_8, action_1, mdp.NextState(s_6))
        spec.transition(s_8, action_1, mdp.Reward(-3.0))

        spec.transition(s_7, action_0, mdp.NextState(s_6))
        spec.transition(s_7, action_0, mdp.Reward(-3.0))
        spec.transition(s_7, action_1, mdp.NextState(s_5))
        spec.transition(s_7, action_1, mdp.Reward(-3.0))

        spec.transition(s_6, action_0, mdp.NextState(s_5))
        spec.transition(s_6, action_0, mdp.Reward(-3.0))
        spec.transition(s_6, action_1, mdp.NextState(s_4))
        spec.transition(s_6, action_1, mdp.Reward(-3.0))

        spec.transition(s_5, action_0, mdp.NextState(s_4))
        spec.transition(s_5, action_0, mdp.Reward(-3.0))
        spec.transition(s_5, action_1, mdp.NextState(s_3))
        spec.transition(s_5, action_1, mdp.Reward(-3.0))

        spec.transition(s_4, action_0, mdp.NextState(s_3))
        spec.transition(s_4, action_0, mdp.Reward(-3.0))
        spec.transition(s_4, action_1, mdp.NextState(s_2))
        spec.transition(s_4, action_1, mdp.Reward(-3.0))

        spec.transition(s_3, action_0, mdp.NextState(s_2))
        spec.transition(s_3, action_0, mdp.Reward(-3.0))
        spec.transition(s_3, action_1, mdp.NextState(s_1))
        spec.transition(s_3, action_1, mdp.Reward(-3.0))

        spec.transition(s_2, action_0, mdp.NextState(s_1))
        spec.transition(s_2, action_0, mdp.Reward(-3.0))
        spec.transition(s_2, action_1, mdp.NextState(s_0))
        spec.transition(s_2, action_1, mdp.Reward(-3.0))

        spec.transition(s_1, action_0, mdp.NextState(s_0))
        spec.transition(s_1, action_0, mdp.Reward(-2.0))
        spec.transition(s_1, action_1, mdp.NextState(s_0))
        spec.transition(s_1, action_1, mdp.Reward(-2.0))
        self.spec = spec
        self.env = spec.to_env()


    def visualize_spec(self):
        spec = self.spec
        spec_graph = spec.to_graph()
        spec_png = mdp.graph_to_png(spec_graph)
        mdp.display_mdp(spec)
        save_mdp_to_png(spec, self.file_name)




