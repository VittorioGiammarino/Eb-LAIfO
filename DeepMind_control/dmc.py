from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale
from DeepMind_control import pixels
from dm_env import StepType, specs

from dmc_remastered import ALL_ENVS
from dmc_remastered import DMCR_VARY

class DMC_Remastered_Env(dm_env.Environment):
    def __init__(self, 
                 task_builder,
                 visual_seed,
                 env_seed,
                 delta,
                 vary=DMCR_VARY):
        
        self._task_builder = task_builder
        self._env_seed = env_seed
        self._visual_seed = visual_seed
        
        self._env = self._task_builder(delta, dynamics_seed=0, visual_seed=0, vary=vary)
        self._vary = vary
        self._delta = delta
        
        self.make_new_env()
        
    def make_new_env(self):
        dynamics_seed = self._env_seed
        visual_seed = self._visual_seed
        self._env = self._task_builder(self._delta,
            dynamics_seed=dynamics_seed, visual_seed=visual_seed, vary=self._vary
        )
        
    def step(self, action):
        return self._env.step(action)
    
    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)   

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)

class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)

class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels', depth_flag=False, segm_flag=False):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key
        self._depth_flag = depth_flag
        self._segm_flag = segm_flag

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape

        if self._depth_flag:
            self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[1 * num_frames], pixels_shape[:2]], axis=0),
            dtype=np.uint8, minimum=0, maximum=255, name='observation')

        elif self._segm_flag:
            self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[1 * num_frames], pixels_shape[:2]], axis=0),
            dtype=np.uint8, minimum=0, maximum=255, name='observation')

        else:
            # remove batch dim
            if len(pixels_shape) == 4:
                pixels_shape = pixels_shape[1:]
            self._obs_spec = specs.BoundedArray(shape=np.concatenate(
                [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                                dtype=np.uint8,
                                                minimum=0,
                                                maximum=255,
                                                name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        obs = time_step.observation[self._pixels_key]

        # remove batch dim
        if len(obs.shape) == 4:
            obs = obs[0]

        if self._depth_flag:
            # Shift nearest values to the origin.
            obs -= obs.min()
            # Scale by 2 mean distances of near rays.
            obs /= 2*obs[obs <= 1].mean()
            # Scale to [0, 255]
            obs = 255*np.clip(obs, 0, 1).astype(np.uint8)
            obs = obs.reshape((1,)+obs.shape).copy()

        elif self._segm_flag:
            obs = obs[:, :, 0]
            # Infinity is mapped to -1
            obs = obs.astype(np.float64) + 1
            # Scale to [0, 1]
            obs = obs / obs.max()
            obs = (255*obs).astype(np.uint8)
            obs = obs.reshape((1,)+obs.shape).copy()

        else:
            obs = obs.transpose(2, 0, 1).copy()

        return obs

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
    
class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)

class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

def make(name, frame_stack, action_repeat, seed, image_height=84, image_width=84):
    domain, task = name.split('_', 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup='ball_in_cup').get(domain, domain)
    # make sure reward is not visualized
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs={'random': seed},
                         visualize_reward=False)
        pixels_key = 'pixels'
    else:
        name = f'{domain}_{task}_vision'
        env = manipulation.load(name, seed=seed)
        pixels_key = 'front_close'
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # add renderings for clasical tasks
    if (domain, task) in suite.ALL_TASKS:
        # zoom in camera for quadruped
        if domain == 'quadruped':
            camera_id = 2
        else:
            camera_id = 0
        
        render_kwargs = dict(height=image_height, width=image_width, camera_id=camera_id)
        env = pixels.Wrapper(env,
                             pixels_only=True,
                             render_kwargs=render_kwargs)
    # stack several frames
    env = FrameStackWrapper(env, frame_stack, pixels_key)
    env = ExtendedTimeStepWrapper(env)
    return env

def make_remastered(name, frame_stack, action_repeat, seed, visual_seed, vary, delta, image_height=84, image_width=84, 
                    depth_flag=False, segm_flag=False):
    domain, task = name.split('_', 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup='ball_in_cup').get(domain, domain)
    
    env = DMC_Remastered_Env(ALL_ENVS[domain][task], visual_seed, seed, delta, vary)
    pixels_key = 'pixels'
        
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    
    try:
        if domain == 'quadruped':
            camera_id = 2
        else:
            camera_id = 0

        if depth_flag:
            segm_flag=False
                
        render_kwargs = dict(height=image_height, width=image_width, camera_id=camera_id,
                             depth=depth_flag, segmentation=segm_flag)
        
        env = pixels.Wrapper(env,
                             pixels_only=True,
                             render_kwargs=render_kwargs)
    except:
        raise NotImplementedError
        
    # stack several frames
    env = FrameStackWrapper(env, frame_stack, pixels_key, depth_flag, segm_flag)
    env = ExtendedTimeStepWrapper(env)
    return env
