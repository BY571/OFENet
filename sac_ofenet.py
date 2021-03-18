
import numpy as np
import random
from collections import deque
import time

import json
import gym
import pybullet_envs

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, MultivariateNormal

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import matplotlib.pyplot as plt
import wandb
from scripts.utils import timer, fill_buffer, pretrain_ofenet, get_target_dim
from scripts.replay_buffer import ReplayBuffer
from scripts.agent import REDQ_Agent
from scripts.ofenet import OFENet, DummyRepresentationLearner



def evaluate(frame, eval_runs=5, capture=False):
    """
    Makes an evaluation run with the current epsilon
    """

    reward_batch = []
    for i in range(eval_runs):
        state = eval_env.reset()

        rewards = 0
        while True:
            action = agent.eval_(np.expand_dims(state, axis=0))
            action_v = np.clip(action, action_low, action_high)
            state, reward, done, _ = eval_env.step(action_v)
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
    if capture == False:   
        writer.add_scalar("Test_Reward", np.mean(reward_batch), frame)

    
def train(steps, precollected, agent):
    scores_deque = deque(maxlen=100)
    average_100_scores = []
    scores = []
    losses = []

    state = env.reset()
    state = state.reshape((1, state_size))
    score = 0
    i_episode = 1
    for step in range(precollected+1, steps+1):

        # eval runs
        if step % args.eval_every == 0 or step == precollected+1:
            evaluate(step, args.eval_runs)

        action = agent.act(state)
        action_v = action.numpy()
        action_v = np.clip(action_v, action_low, action_high)
        next_state, reward, done, info = env.step(action_v)
        next_state = next_state.reshape((1, state_size))
        ofenet_loss, a_loss, c_loss = agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            scores_deque.append(score)
            scores.append(score)
            average_100_scores.append(np.mean(scores_deque))
            current_step = step - precollected
            writer.add_scalar("Average100", np.mean(scores_deque), current_step)
            writer.add_scalar("Train_Reward", score, current_step)
            writer.add_scalar("OFENet loss", ofenet_loss, current_step)
            writer.add_scalar("Actor loss", a_loss, current_step)
            writer.add_scalar("Critic loss", c_loss, current_step)
            print('\rEpisode {} Env. Step: [{}/{}] Reward: {:.2f}  Average100 Score: {:.2f} ofenet_loss: {:.3f}, a_loss: {:.3f}, c_loss: {:.3f}'.format(i_episode, step, steps, score, np.mean(scores_deque), ofenet_loss, a_loss, c_loss))
            state = env.reset()
            state = state.reshape((1, state_size))
            score = 0
            i_episode += 1

    return scores
        
    
parser = argparse.ArgumentParser(description="")
parser.add_argument("--env", type=str, default="HalfCheetahBulletEnv-v0",
                    help="Environment name, default = HalfCheetahBulletEnv-v0")
parser.add_argument("--info", type=str, default="SAC-OFENet",
                    help="Information or name of the run")
parser.add_argument("--steps", type=int, default=1_000_000,
                    help="The amount of training interactions with the environment, default is 1mio")
parser.add_argument("--N", type=int, default=2,
                    help="Number of Q-network ensemble, default is 10")
parser.add_argument("--M", type=int, default=2,
                    help="Numbe of subsample set of the emsemble for updating the agent, default is 2 (currently only supports 2!)")
parser.add_argument("--G", type=int, default=1,
                    help="Update-to-Data (UTD) ratio, updates taken per step with the environment, default=20")
parser.add_argument("--eval_every", type=int, default=10_000,
                    help="Number of interactions after which the evaluation runs are performed, default = 10.000")
parser.add_argument("--eval_runs", type=int, default=1,
                    help="Number of evaluation runs performed, default = 1")
parser.add_argument("--seed", type=int, default=0,
                    help="Seed for the env and torch network weights, default is 0")
parser.add_argument("--lr", type=float, default=3e-4,
                    help="Actor learning rate of adapting the network weights, default is 3e-4")
parser.add_argument("--layer_size", type=int, default=256,
                    help="Number of nodes per neural network layer, default is 256")
parser.add_argument("--replay_memory", type=int, default=int(1e6),
                    help="Size of the Replay memory, default is 1e6")
parser.add_argument("-bs", "--batch_size", type=int, default=256,
                    help="Batch size, default is 256")
parser.add_argument("-t", "--tau", type=float, default=0.005,
                    help="Softupdate factor tau, default is 0.005")
parser.add_argument("-g", "--gamma", type=float, default=0.99,
                    help="discount factor gamma, default is 0.99")
parser.add_argument("--ofenet_layer", type=int, default=8,
                    help="Number of dense layer in each (state/action) block of the ofenet network, (default: 8)")
parser.add_argument("--collect_random", type=int, default=10_000,
                    help="Number of randomly collected transitions to pretrain the OFENet, (default: 10.000)")
parser.add_argument("--batch_norm", type=int, default=1, choices=[0,1],
                    help="Add batch norm to the OFENet, default: 1")
parser.add_argument("--activation", type=str, default="SiLU", choices=["SiLU", "ReLU"],
                    help="Type of activation function for the ofenet network, choose between SiLU and ReLU, default: SiLU")
parser.add_argument("--ofenet", type=int, default=1, choices=[0,1], help="Using OFENet feature extractor, default: True")

args = parser.parse_args()
    
    
    
if __name__ == "__main__":
        
    writer = SummaryWriter("runs/"+args.info)
    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    action_high = env.action_space.high[0]
    seed = args.seed
    action_low = env.action_space.low[0]
    torch.manual_seed(seed)
    env.seed(seed)
    eval_env.seed(seed+1)
    np.random.seed(seed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target_dim = get_target_dim(args.env)

    replay_buffer = ReplayBuffer(action_size, state_size, args.replay_memory, args.batch_size, seed, device)
    
    if args.ofenet:
        ofenet_size = 30
        ofenet = OFENet(state_size,
                            action_size,
                            target_dim=target_dim,
                            num_layer=args.ofenet_layer,
                            hidden_size=ofenet_size,
                            batch_norm=args.batch_norm,
                            activation=args.activation,
                            device=device).to(device)
        print(ofenet)
    else:
        ofenet_size = 30
        ofenet = DummyRepresentationLearner(state_size,
                            action_size,
                            target_dim=target_dim,
                            num_layer=args.ofenet_layer,
                            hidden_size=ofenet_size,
                            batch_norm=args.batch_norm,
                            activation=args.activation,
                            device=device)
        
    agent = REDQ_Agent(state_size=state_size,
                action_size=action_size,
                replay_buffer=replay_buffer,
                ofenet=ofenet,
                random_seed=seed,
                lr=args.lr,
                hidden_size=args.layer_size,
                gamma=args.gamma,
                tau=args.tau,
                device=device,
                action_prior="uniform",
                N=args.N,
                M=args.M,
                G=args.G)

    fill_buffer(samples=args.collect_random,
                agent=agent,
                env=env)
    if args.ofenet:
        t0 = time.time()
        agent = pretrain_ofenet(agent=agent,
                        epochs=args.collect_random,
                        writer=writer,
                        target_dim=target_dim)
        t1 = time.time()
        timer(t0, t1, train_type="Pre-Training")
    # untrained eval run
    evaluate(0, args.eval_runs)
    
    t0 = time.time()
    final_average100 = train(steps=args.steps,
                             precollected=args.collect_random,
                             agent=agent)
    t1 = time.time()
    env.close()
    timer(t0, t1)
    
    # save parameter
    #with open('runs/'+args.info+".json", 'w') as f:
    #    json.dump(args.__dict__, f, indent=2)
    #hparams = vars(args)
    #metric = {"final average 100 train reward": final_average100}
    #writer.add_hparams(hparams, metric)