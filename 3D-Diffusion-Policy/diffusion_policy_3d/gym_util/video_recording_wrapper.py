import gym
import numpy as np
from termcolor import cprint

"""
`SimpleVideoRecordingWrapper` 类在 `MujocoPointcloudWrapperAdroit` 类上实现了视频录制的功能。具体来说，它通过以下方式扩展了功能：

1. **初始化参数**：在构造函数中，它接收了 `mode` 和 `steps_per_render` 参数，用于指定渲染模式和每多少步进行一次渲染。

2. **重写 reset 方法**：当环境被重置时，它会清空 frames 列表，并将当前环境的渲染结果保存到 frames 列表中。同时，它会将 step_count 重置为1。

3. **重写 step 方法**：每当执行一个动作并获得下一步的结果时，它都会增加 step_count 的值，并将当前环境的渲染结果添加到 frames 列表中。

4. **get_video 方法**：该方法用于获取整个 episode 的视频帧。它首先将所有帧堆叠成一个 numpy 数组，然后调整维度顺序以适应特定存储格式（如 wandb mp4 文件）的要求。

通过这些额外的功能，`SimpleVideoRecordingWrapper` 能够在不改变原始环境逻辑的情况下记录下整个 episode 的视觉表现，这对于分析策略行为、调试和可视化很有帮助。
"""
class SimpleVideoRecordingWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            mode='rgb_array',
            steps_per_render=1,
        ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        
        self.mode = mode
        self.steps_per_render = steps_per_render

        self.step_count = 0

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.frames = list()

        frame = self.env.render(mode=self.mode)
        assert frame.dtype == np.uint8
        self.frames.append(frame)
        
        self.step_count = 1
        return obs
    
    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        
        frame = self.env.render(mode=self.mode)
        assert frame.dtype == np.uint8
        self.frames.append(frame)
        
        return result
    
    def get_video(self):
        video = np.stack(self.frames, axis=0) # (T, H, W, C)
        # to store as mp4 in wandb, we need (T, H, W, C) -> (T, C, H, W)
        video = video.transpose(0, 3, 1, 2)
        return video

