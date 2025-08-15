import torch

# Here 0 encodes cooperate
# 1 encode defect


class IPD:
    def __init__(self, device, batch_size, shared_rewards=False):
        self.device = device
        self.n_actions = 2
        self.batch_size = batch_size
        self.payout = torch.Tensor([[3.0, 0], [5, 1]]).repeat(batch_size, 1, 1).to(device)
        self.last_obs_1 = torch.zeros((batch_size, 4))
        self.last_obs_2 = torch.zeros((batch_size, 4))
        self.last_rew_1 = 0.01 * torch.ones((batch_size, 1))
        self.last_rew_2 = 0.01 * torch.ones((batch_size, 1))
        self.shared_rewards = shared_rewards

    def step(self, actions):
        a1, a2 = actions
        a1 = torch.nn.functional.one_hot(a1, 2)
        a2 = torch.nn.functional.one_hot(a2, 2)

        obs_1 = torch.cat([a1, a2], dim=1)
        obs_2 = torch.cat([a2, a1], dim=1)

        a1 = a1.reshape(self.batch_size, 1, a1.shape[1])
        a2 = a2.reshape(self.batch_size, a2.shape[1], 1)
        a1 = a1.float()
        a2 = a2.float()

        payout_T = torch.transpose(self.payout, 2, 1)
        r1 = torch.bmm(torch.bmm(a1, self.payout), a2)
        r2 = torch.bmm(torch.bmm(a1, payout_T), a2)

        self.last_obs_1 = obs_1.float()
        self.last_obs_2 = obs_2.float()
        self.last_rew_1 = r1
        self.last_rew_2 = r2

        r1 = r1.flatten()
        r2 = r2.flatten()

        if self.shared_rewards:
            ret_rew = (r1 + r2, r1 + r2)
        else:
            ret_rew = (r1, r2)

        return (obs_1.float(), obs_2.float()), ret_rew

    def reset(self):
        return torch.ones(self.batch_size, 2 * self.n_actions).to(self.device), {}

    def obs(self):
        return (
            self.last_obs_1.float().to(self.device),
            self.last_obs_2.float().to(self.device),
        )

    def legal_action_mask(self):
        return torch.ones((self.batch_size, self.n_actions)).to(self.device)

    def reward(self):
        return (
            self.last_rew_1.flatten().to(self.device),
            self.last_rew_2.flatten().to(self.device),
        )
