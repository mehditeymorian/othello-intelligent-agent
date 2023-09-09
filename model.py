import torch
import torch.nn as nn
from env import OthelloEnv


class DQN(nn.Module):

    def __init__(self, env: OthelloEnv, hidden_size):
        super(DQN, self).__init__()

        self.env = env

        self.l1 = nn.Linear(env.observation_space.shape, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, env.action_space.shape)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.l1(x))
        x = self.relu2(self.l2(x))
        x = self.relu3(self.l3(x))
        x = self.l3(x)

        return x

    def act(self, observation):
        observation_t = torch.as_tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            q_values = self(observation_t)

        move = torch.argmax(q_values).detach().item()
        ok_move = self.env.game.in_options((move // 8, move % 8))

        if ok_move:
            return move, ok_move

        action = self.env.valid_moves_max(q_values.tolist())

        return action, ok_move
