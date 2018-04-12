import multiprocessing
from utils.replay_memory import Memory
from utils.torch import *
from torch.autograd import Variable
import math
import time


def collect_samples(pid, queue, env, policy, custom_reward, mean_action,
                    tensor, render, update_rs, min_batch_size):
    torch.randn(pid, )
    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0

    while num_steps < min_batch_size:
        obs = env.reset()
        reward_episode = 0

        for t in range(10000):
            obs_var = Variable(tensor(obs).unsqueeze(0), volatile=True)
            if mean_action:
                action = policy(obs_var)[0].data[0].numpy()
            else:
                action = policy.select_action(obs_var)[0].numpy()
            next_obs, reward, done, _ = env.step(action)
            reward_episode += reward

            if custom_reward is not None:
                reward = custom_reward(obs, action)
                total_c_reward += reward
                min_c_reward = min(min_c_reward, reward)
                max_c_reward = max(max_c_reward, reward)

            mask = 0 if done else 1

            memory.push(obs, action, mask, next_obs, reward)

            if render:
                env.render()
            if done:
                break

            obs = next_obs

        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        total_reward += reward_episode
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


class Agent:

    def __init__(self, env_factory, policy, custom_reward=None, mean_action=False, render=False,
                 tensor_type=torch.DoubleTensor, num_threads=1):
        self.env_factory = env_factory
        self.policy = policy
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.render = render
        self.tensor = tensor_type
        self.num_threads = num_threads
        self.env_list = []
        for i in range(num_threads):
            self.env_list.append(self.env_factory(i))

    def collect_samples(self, min_batch_size):
        t_start = time.time()
        if use_gpu:
            self.policy.cpu()
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads-1):
            worker_args = (i+1, queue, self.env_list[i + 1], self.policy, self.custom_reward, self.mean_action,
                           self.tensor, False, False, thread_batch_size)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        memory, log = collect_samples(0, None, self.env_list[0], self.policy, self.custom_reward, self.mean_action,
                                      self.tensor, self.render, True, thread_batch_size)

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            memory.append(worker_memory)
        batch = memory.sample()
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        if use_gpu:
            self.policy.cuda()
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch, log