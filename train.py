import torch
import torch.nn as nn
from collections import deque
import itertools
from agent import Agent
from env import OthelloEnv, TURN_BLACK
from torch.utils.tensorboard import SummaryWriter

GAMMA = 0.99
BATCH_SIZE = 128
BUFFER_SIZE = 15_000
MIN_REPLAY_SIZE = 5000
EPSILON_START = 0.9
EPSILON_END = 0.02
EPSILON_DECAY = 1000
SAVE_FREQUENCY = 10_000
LR = 1e-4
TAU = 0.009
MOMENTUM = 0.15
GAME_PLAY_ITERATION = 10_0000
HIDDEN_SIZE = 256
CLIP_GRAD = 200

writer = SummaryWriter(f'runs/agent-H3-{HIDDEN_SIZE}-RMSProp-CLIP-{CLIP_GRAD}-LR-{LR:0.4f}-TAU-{TAU:0.3f}-MO-{MOMENTUM:0.2f}-testtttt')
# writer = SummaryWriter(f'runs/agent-H3-{HIDDEN_SIZE}-Adam-CLIP-{CLIP_GRAD}-LR-{LR:0.4f}-TAU-{TAU:0.3f}-MO-{MOMENTUM:0.2f}')

reward_buffer = deque([0.0], maxlen=1000)

games_played = 0
wins = 0
draws = 0
agent_total_moves = 0
agent_correct_moves = 0

env = OthelloEnv(TURN_BLACK)

agent = Agent(env,
              batch_size=BATCH_SIZE,
              gamma=GAMMA,
              buffer_size=BUFFER_SIZE,
              epsilon_params=(EPSILON_DECAY, EPSILON_START, EPSILON_END),
              save_frequency=SAVE_FREQUENCY,
              tau=TAU,
              hidden_size=HIDDEN_SIZE)

writer.add_graph(agent.policy_net, torch.zeros(env.observation_space.shape))
writer.flush()

optimizer = torch.optim.AdamW(agent.policy_net.parameters(), lr=LR, amsgrad=True)

agent.initialize_buffer(MIN_REPLAY_SIZE)

observation = env.reset()

for step in itertools.count():
    action, ok_move = agent.act(observation, step)

    agent_total_moves += 1
    if ok_move:
        agent_correct_moves += 1

    _, new_observation, reward, done = env.step(action, ok_move)

    transition = (observation, action, reward, done, new_observation)
    agent.add_transition(transition)
    observation = new_observation

    policy_values, expected_values = agent.calculate_quality_values()
    loss = nn.functional.smooth_l1_loss(policy_values, expected_values.unsqueeze(1))
    # gradient descent
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(agent.policy_net.parameters(), CLIP_GRAD)
    optimizer.step()

    # update target network
    agent.update_nets(step)

    if done:
        score = env.agent_score()
        reward_buffer.append(score)
        if env.agent_win():
            wins += 1
        elif env.agent_draw():
            draws += 1
        observation = env.reset()
        games_played += 1

        win_ratio = 100 * wins // max(games_played, 1)
        win_draw_ratio = 100 * (wins + draws) // max(games_played, 1)
        correct_move_ratio = 100 * agent_correct_moves // max(agent_total_moves, 1)
        writer.add_scalar('win_ratio', win_ratio, games_played, double_precision=True, new_style=True)
        writer.add_scalar('win_draw_ratio', win_draw_ratio, games_played, double_precision=True, new_style=True)
        writer.add_scalar('score', score, games_played, new_style=True)
        writer.add_scalar('correct_move_ratio', correct_move_ratio, games_played, double_precision=True, new_style=True)

        if games_played == GAME_PLAY_ITERATION:
            break

    if step % 1000 == 0:
        print(f"step: {step}")
    #     win_ratio = 100 * wins // max(games_played, 1)
    #     win_draw_ratio = 100 * (wins + draws) // max(games_played, 1)
    #     correct_move_ratio = 100 * agent_correct_moves // max(agent_total_moves, 1)
    #     print()
    #     print(
    #         f">  correct_move_ratio: {correct_move_ratio:.2f}% wins: {wins}/{games_played} win ratio: [{win_ratio:.2f}%:{win_draw_ratio:.2f}%]")

# env = OthelloEnv(TURN_BLACK)
# observation = env.reset()
#
# while True:
#     observation_t = torch.as_tensor(observation, dtype=torch.float32)
#     action, ok_move = agent.policy_net.act(observation_t)
#     observation, _, done, _, _, _ = env.step(action)
#     env.render()
#     if done:
#         env.reset()
