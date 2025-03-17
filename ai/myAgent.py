class Agent:
    def __init__(self, model):
        super(Agent, self).__init__()
        self.model = model

    def step(self, obs, state, action_mask):
        prep = self.model.preprocess(obs, state, action_mask)
        # _, _, all_masks, _ = prep
        # if (~all_masks[:, :, 1:]).all():
        #     return [], None, None, None

        actions, logits, value = self.model(prep)
        return actions, logits, value, prep
