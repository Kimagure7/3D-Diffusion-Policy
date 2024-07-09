import gym
from gym import spaces
import numpy as np
import torch
from collections import defaultdict, deque
import dill


def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x,axis=0),n,axis=0)

def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype
    )

def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')


def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    
    if isinstance(x[0], torch.Tensor):
        return torch.stack(x[-n:])
    else:
        return np.array(x[-n:])



def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result


def aggregate(data, method='max'):
    if isinstance(data[0], torch.Tensor):
        if method == 'max':
            # equivalent to any
            return torch.max(torch.stack(data))
        elif method == 'min':
            # equivalent to all
            return torch.min(torch.stack(data))
        elif method == 'mean':
            return torch.mean(torch.stack(data))
        elif method == 'sum':
            return torch.sum(torch.stack(data))
        else:
            raise NotImplementedError()
    else:
        if method == 'max':
            # equivalent to any
            return np.max(data)
        elif method == 'min':
            # equivalent to all
            return np.min(data)
        elif method == 'mean':
            return np.mean(data)
        elif method == 'sum':
            return np.sum(data)
        else:
            raise NotImplementedError()


def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    if isinstance(all_obs[0], np.ndarray):
        result = np.zeros((n_steps,) + all_obs[-1].shape, 
            dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = np.array(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
    elif isinstance(all_obs[0], torch.Tensor):
        result = torch.zeros((n_steps,) + all_obs[-1].shape, 
            dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = torch.stack(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
    else:
        raise RuntimeError(f'Unsupported obs type {type(all_obs[0])}')
    return result

"""
`MultiStepWrapper`在`SimpleVideoRecordingWrapper`基础上实现了以下功能：

1. **多步观察和动作处理**：它允许环境接收一系列连续的动作（`n_action_steps`），并返回一系列连续的观察（`n_obs_steps`）。这使得模型可以基于过去几步的动作和观察来做出决策，增强了模型对环境动态的理解。

2. **奖励聚合**：对于连续的步骤，它可以使用指定的方法（如最大值、最小值、平均值或总和）来聚合奖励。这有助于在多步决策中考虑长期奖励。

3. **终止和截断条件**：它维护了动作序列中的终止状态（done）列表，并能够处理环境的终止或截断条件。如果在序列中检测到终止状态，后续的动作将不再执行。

4. **信息字典管理**：它记录了每一步的信息字典，并提供了获取最近几步信息的功能。这有助于分析模型在不同步骤中的表现。

5. **自定义函数执行**：通过`run_dill_function`方法，它支持执行外部传入的函数，增加了环境的灵活性和扩展性。

6. **属性访问与信息获取**：提供了获取内部状态（如奖励序列、信息字典等）的方法，以便于调试和性能评估。

7. **适应不同的观察空间类型**：无论是Box型还是Dict型的空间，它都能够正确处理并堆叠最近的观察数据。

8. **初始化与重置功能增强**：重置环境中包含了初始化多步缓冲区的操作，确保每次开始新回合时环境处于正确的初始状态。

9. **动态调整观测窗口大小**：通过参数化的方式允许用户指定需要多少历史步骤的信息作为当前观测的一部分，增强了对不同任务场景的适应性。

动作重复空间定义：首先，MultiStepWrapper通过repeated_space函数扩展了环境的原始动作空间。这会创建一个新的Gym空间，其维度等于原始动作空间维度乘以指定的多步数（n_action_steps）。这意味着你可以在一步中提供一个包含多个连续动作序列的动作

与Diffusion Policy中完全保持一致
"""
class MultiStepWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            n_obs_steps, 
            n_action_steps, 
            max_episode_steps=None,
            reward_agg_method='max'
        ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.n_obs_steps = n_obs_steps

        self.obs = deque(maxlen=n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=n_obs_steps+1))
    
    def reset(self):
        """Resets the environment using kwargs."""
        obs = super().reset()

        self.obs = deque([obs], maxlen=self.n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=self.n_obs_steps+1))

        obs = self._get_obs(self.n_obs_steps)
        return obs

    def step(self, action):
        """
        actions: (n_action_steps,) + action_shape
        """
        for act in action:
            if len(self.done) > 0 and self.done[-1]:
                # termination
                break
            observation, reward, done, info = super().step(act)

            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) \
                and (len(self.reward) >= self.max_episode_steps):
                # truncation
                done = True
            self.done.append(done)
            self._add_info(info)

        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        done = aggregate(self.done, 'max')
        info = dict_take_last_n(self.info, self.n_obs_steps)
        return observation, reward, done, info

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)
    
    def get_rewards(self):
        return self.reward
    
    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)
    
    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result
