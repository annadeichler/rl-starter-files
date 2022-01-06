import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv

import utils
from utils import device


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="how many worst episodes to show")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")
parser.add_argument("--tile-size", type=int, default=8,
                    help="tile size")

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

print(f"Device: {device}\n")

# Load environment

env = ut.make_env(args.env, args.seed)

# Load agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    argmax=args.argmax, num_envs=args.procs,
                    use_memory=args.memory, use_text=args.text)
print("Agent loaded\n")

### save episodes

obs = env.reset()
obs_img = env.render('rgb_array', highlight=False, tile_size=args.tile_size)
print(f"obs.shape: {obs_img.shape}")
traj = None
episodes = args.episodes

ep_rew, ep_len = [], []

for ep in range(episodes):
    steps = 0
    rew = 0
    ep_traj = obs_img
    ep_traj = ep_traj[np.newaxis, ...]
    while True:
        with torch.no_grad():
            action = agent.get_action(obs)

        # Observation, reward and next obs
        obs, reward, done, _ = env.step(action)
        # traj = np.append(traj, obs.numpy(), axis=0)
        obs_img = env.render('rgb_array', highlight=False, tile_size=args.tile_size)
        ep_traj = np.append(ep_traj, obs_img[np.newaxis, ...], axis=0)

        rew += reward
        steps += 1

        if done:

            if rew >= 0.92 and steps < 16:
                print(f"episode {ep}, reward {rew}, steps {steps}")
                ep_rew.append(rew)
                ep_len.append(steps)
                if traj is None:
                    traj = ep_traj.copy()
                else:
                    traj = np.append(traj, ep_traj, axis=0)

            obs = env.reset()
            obs_img = env.render('rgb_array', highlight=False, tile_size=args.tile_size)
            break

print(traj.shape)
print(sum(ep_rew)/len(ep_rew))
print(sum(ep_len)/len(ep_len))

path = "storage/{}/".format(args.env)
with open(path + "/expert_{}.npy".format(args.env), "wb") as f:
    np.save(f, traj[:traj.shape[0]])
