import gym
import numpy as np
import pytorch3d.ops as torch3d_ops
import torch
from termcolor import cprint
from diffusion_policy_3d.gym_util.mujoco_point_cloud import PointCloudGenerator
from typing import NamedTuple, Any
from dm_env import StepType

ADROIT_PC_TRANSFORM = np.array([
                    [1, 0, 0],
                    [0, np.cos(np.radians(45)), np.sin(np.radians(45))],
                    [0, -np.sin(np.radians(45)), np.cos(np.radians(45))]])

ADROIT_HAND_FINGER_NAMES = {'palm',
            'ffknuckle', 'ffproximal', 'ffmiddle', 'ffdistal',
            'mfknuckle', 'mfproximal', 'mfmiddle', 'mfdistal',
            'rfknuckle', 'rfproximal', 'rfmiddle', 'rfdistal',
            'lfknuckle', 'lfproximal', 'lfmiddle', 'lfdistal',
            'thbase', 'thproximal', 'thmiddle', 'thdistal',}

ENV_POINT_CLOUD_CONFIG = {
    
    'adroit_hammer': {
        'min_bound': [-10, -10, -0.099],
        'max_bound': [10, 10, 10],
        'num_points': 512,
        'point_sampling_method': 'fps',
        'cam_names':['top'],
        'transform': ADROIT_PC_TRANSFORM,
        'scale': np.array([1, 1, 1]),
        'offset': np.array([0, 0, 1.]),
    },
    
    'adroit_door': {
       'min_bound': [-10, -10, -0.499],
        'max_bound': [10, 10, 10],
        'num_points': 512,
        'point_sampling_method': 'fps',
        'cam_names':['top'],
        'transform': ADROIT_PC_TRANSFORM,
        'scale': np.array([1, 1, 1]),
        'offset': np.array([0, 0, 1.]),
    },
    
    'adroit_pen': {
        'min_bound': [-10, -10, -0.79],
        'max_bound': [10, 10, 10],
        'num_points': 512,
        'point_sampling_method': 'fps',
        'cam_names':['vil_camera'],
        'transform': None,
        'scale': np.array([1, 1, 1]),
        'offset': np.array([0, 0, 0.]),
    },

}

def point_cloud_sampling(point_cloud:np.ndarray, num_points:int, method:str='uniform'):
    """
    support different point cloud sampling methods
    point_cloud: (N, 6), xyz+rgb
    """
    if num_points == 'all': # use all points
        return point_cloud
    if point_cloud.shape[0] <= num_points:
        # pad with zeros
        point_cloud = np.concatenate([point_cloud, np.zeros((num_points - point_cloud.shape[0], 6))], axis=0)
        return point_cloud
    
    if method == 'uniform':
        # uniform sampling
        sampled_indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        point_cloud = point_cloud[sampled_indices]
    elif method == 'fps':
        # fast point cloud sampling using torch3d
        point_cloud = torch.from_numpy(point_cloud).unsqueeze(0).cuda()
        num_points = torch.tensor([num_points]).cuda()
        # remember to only use coord to sample
        _, sampled_indices = torch3d_ops.sample_farthest_points(points=point_cloud[...,:3], K=num_points)
        point_cloud = point_cloud.squeeze(0).cpu().numpy()
        point_cloud = point_cloud[sampled_indices.squeeze(0).cpu().numpy()]
    else:
        raise NotImplementedError(f"point cloud sampling method {method} not implemented")

    return point_cloud
    

class ExtendedTimeStepAdroit(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    observation_sensor: Any
    observation_pointcloud: Any
    action: Any
    n_goal_achieved: Any
    time_limit_reached: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)

"""
这个MujocoPointcloudWrapperAdroit类主要在原始环境（env）的基础上增加了点云数据的生成、处理和整合到观察（observation）中，以增强环境的感知信息。

`MujocoPointcloudWrapperAdroit`在`PointCloudGenerator`的基础上实现了以下功能：
1. **点云裁剪**：根据环境特定的最小和最大边界（如果使用点云裁剪的话）来裁剪生成的点云，只保留那些落在指定边界内的点。
2. **点云变换**：对生成的点云应用旋转、缩放和偏移变换，这些变换参数是根据环境配置预先设定的。
3. **点云采样**：将生成的点云采样到固定数量的点。支持两种采样方法：均匀采样（uniform）和远thest point sampling (fps)。这确保了输出的点云具有统一的数量，便于后续处理。
4. **整合到观察结果中**：将处理后的点云数据添加到环境返回的观察结果字典中，这样智能体就可以利用这些信息进行决策。
5. **深度图获取**：同时获取与点云相关的深度图信息，并将其也整合到观察结果中。
"""
class MujocoPointcloudWrapperAdroit(gym.Wrapper):
    """
    fetch point cloud from mujoco and add it to obs
    """
    def __init__(self, env, env_name:str, use_point_crop=True):
        super().__init__(env)
        self.env_name = env_name
        # point cloud cropping
        self.min_bound = ENV_POINT_CLOUD_CONFIG[env_name].get('min_bound', None)
        self.max_bound = ENV_POINT_CLOUD_CONFIG[env_name].get('max_bound', None)
        
        self.use_point_crop = use_point_crop
        cprint(f"[MujocoPointcloudWrapper] use_point_crop: {self.use_point_crop}", 'green')

        
        # point cloud sampling
        self.num_points = ENV_POINT_CLOUD_CONFIG[env_name].get('num_points', 512)
        self.point_sampling_method = ENV_POINT_CLOUD_CONFIG[env_name].get('point_sampling_method', 'uniform')
        cprint(f"[MujocoPointcloudWrapper] sampling {self.num_points} points from point cloud using {self.point_sampling_method}", 'green')
        assert self.point_sampling_method in ['uniform', 'fps'], \
            f"point_sampling_method should be one of ['uniform', 'fps'], but got {self.point_sampling_method}"
        
        # point cloud generator
        self.pc_generator = PointCloudGenerator(sim=env.get_mujoco_sim(),
                                                cam_names=ENV_POINT_CLOUD_CONFIG[env_name]['cam_names'])
        self.pc_transform = ENV_POINT_CLOUD_CONFIG[env_name].get('transform', None)
        self.pc_scale = ENV_POINT_CLOUD_CONFIG[env_name].get('scale', None)
        self.pc_offset = ENV_POINT_CLOUD_CONFIG[env_name].get('offset', None)
    
    

    def get_point_cloud(self, use_RGB=True):
        save_img_dir = None
        point_cloud, depth = self.pc_generator.generateCroppedPointCloud(save_img_dir=save_img_dir) # (N, 6), xyz+rgb
        
        
        # do transform, scale, offset, and crop
        if self.pc_transform is not None:
            point_cloud[:, :3] = point_cloud[:, :3] @ self.pc_transform.T
        if self.pc_scale is not None:
            point_cloud[:, :3] = point_cloud[:, :3] * self.pc_scale
        if self.pc_offset is not None:    
            point_cloud[:, :3] = point_cloud[:, :3] + self.pc_offset

        if self.use_point_crop:
            if self.min_bound is not None:
                mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
                point_cloud = point_cloud[mask]
            if self.max_bound is not None:
                mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
                point_cloud = point_cloud[mask]
            
        # sampling to fixed number of points
        point_cloud = point_cloud_sampling(point_cloud=point_cloud, 
                                           num_points=self.num_points, 
                                           method=self.point_sampling_method)
        
        if not use_RGB:
            point_cloud = point_cloud[:, :3]
        return point_cloud, depth


    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        point_cloud, depth = self.get_point_cloud()
        
        obs_dict['point_cloud'] = point_cloud
        obs_dict['depth'] = depth
        
        """
        obs_dict = {
            'image': obs_pixels,  # (3, 84, 84), [0,255], uint8
            'agent_pos': obs_sensor  # (24,)
        }
        现在再加这两个
        """
        return obs_dict, reward, done, info

    def reset(self):
        obs_dict = self.env.reset()
        point_cloud, depth = self.get_point_cloud()
        obs_dict['point_cloud'] = point_cloud
        obs_dict['depth'] = depth
        return obs_dict

