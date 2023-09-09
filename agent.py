import math
import os.path

from model import DQN
import random
import torch
import numpy as np
from collections import deque
from env import OthelloEnv

online_path = "../state-dic/online.pth"
target_path = "../state-dic/target.pth"


class Agent:

    def __init__(self, env: OthelloEnv, batch_size, gamma, buffer_size, epsilon_params,
                 save_frequency, tau, hidden_size):
        epsilon_decay, epsilon_start, epsilon_end = epsilon_params
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.env = env
        self.save_frequency = save_frequency
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = deque(maxlen=buffer_size)
        self.policy_net = DQN(env, hidden_size)
        self.target_net = DQN(env, hidden_size)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.load()

    def act(self, observation, step):
        epsilon = np.interp(step, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])

        rnd_sample = random.random()

        if rnd_sample <= epsilon:
            action = self.env.sample_valid_moves()
            ok_move = True
        else:
            action, ok_move = self.policy_net.act(observation)

        return action, ok_move

    def update_nets(self, step):
        TAU = self.tau
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = (policy_net_state_dict[key] * TAU) + (target_net_state_dict[key] * (1 - TAU))
        self.target_net.load_state_dict(target_net_state_dict)

        if step % self.save_frequency == 0:
            self.save()

    def calculate_quality_values(self):
        # start gradient step
        transitions = random.sample(self.replay_buffer, self.batch_size)
        #
        states, actions, rewards, dones, next_states = zip(*transitions)

        states_t = torch.as_tensor(states, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
        non_final_mask_t = 1 - torch.as_tensor(dones, dtype=torch.long)
        non_final_states_t = torch.as_tensor(next_states, dtype=torch.float32)[non_final_mask_t]

        values = self.policy_net(states_t).gather(dim=1, index=actions_t)

        # compute targets
        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask_t] = self.target_net(non_final_states_t).max(1)[0]

        expected_state_action_values = (next_state_values * self.gamma) + rewards_t

        return values, expected_state_action_values

    def add_transition(self, transition):
        self.replay_buffer.append(transition)

    def initialize_buffer(self, size):
        observation = self.env.reset()

        for _ in range(size):
            action = self.env.sample_valid_moves()

            _, new_observation, reward, done = self.env.step(action, True)

            transition = (observation, action, reward, done, new_observation)
            self.replay_buffer.append(transition)
            observation = new_observation

            if done:
                observation = self.env.reset()

    def load(self):
        pass
        # if not os.path.exists("../state-dic"):
        #     os.mkdir("../state-dic")
        # if os.path.exists(online_path) and os.path.exists(target_path):
        #     self.policy_net.load_state_dict(torch.load(online_path))
        #     self.target_net.load_state_dict(torch.load(target_path))

    def save(self):
        pass
        # torch.save(self.policy_net.state_dict(), online_path)
        # torch.save(self.target_net.state_dict(), target_path)
