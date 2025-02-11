import collections
import os
import random
import xml.etree.ElementTree as ET

import numpy as np
from dm_control import mujoco
from dm_control.mujoco.wrapper import mjbindings
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import xml_tools
from lxml import etree
from scipy import ndimage

from dmc_remastered import DMCR_VARY, SUITE_DIR, register

from .generate_visuals import get_assets
from .rng import dmcr_random

enums = mjbindings.enums
mjlib = mjbindings.mjlib


def get_model(visual_seed, vary=["camera", "light"]):
    with open(os.path.join(SUITE_DIR, os.path.join("assets", "quadruped.xml")), "r") as f:
        xml = ET.fromstring(f.read())
    if visual_seed != 0:
        with dmcr_random(visual_seed):
            camera_x = random.uniform(-0.5, 0.5)
            camera_y = random.uniform(3.5, 4.5)
            camera_z = random.uniform(1.5, 2.5)

            light_x = random.uniform(-0.7, 0.7)
            light_y = random.uniform(-0.7, 0.7)
            light_z = random.uniform(3.6, 4.6)
        if "camera" in vary:
            xml[8][2][2].attrib["pos"] = f"{camera_x} {camera_y} {camera_z}"
        if "light" in vary:
            xml[8][2][4].attrib["pos"] = f"{light_x} {light_y} {light_z}"
        
    return ET.tostring(xml, encoding="utf8", method="xml")

_DEFAULT_TIME_LIMIT = 20
_CONTROL_TIMESTEP = .02

# Horizontal speeds above which the move reward is 1.
_RUN_SPEED = 5
_WALK_SPEED = 0.5

# Constants related to terrain generation.
_HEIGHTFIELD_ID = 0
_TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
_TERRAIN_BUMP_SCALE = 2  # Spatial scale of terrain bumps (in meters).

# Named model elements.
_TOES = ['toe_front_left', 'toe_back_left', 'toe_back_right', 'toe_front_right']
_WALLS = ['wall_px', 'wall_py', 'wall_nx', 'wall_ny']


@register("quadruped", "walk")
def walk(delta, time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY):  
    """Returns the Walk task."""
    model = get_model(visual_seed, vary)
    assets, _ = get_assets(visual_seed, delta, vary)
    physics = Physics.from_xml_string(model, assets)
    task = Move(desired_speed=_WALK_SPEED, random=dynamics_seed)
    return control.Environment(physics, task, time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP)


@register("quadruped", "run")
def run(delta, time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY):
    """Returns the Run task."""
    model = get_model(visual_seed, vary)
    assets, _ = get_assets(visual_seed, delta, vary)
    physics = Physics.from_xml_string(model, assets)
    task = Move(desired_speed=_RUN_SPEED, random=dynamics_seed)
    return control.Environment(physics, task, time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP)

class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Quadruped domain."""

  def _reload_from_data(self, data):
    super()._reload_from_data(data)
    # Clear cached sensor names when the physics is reloaded.
    self._sensor_types_to_names = {}
    self._hinge_names = []

  def _get_sensor_names(self, *sensor_types):
    try:
      sensor_names = self._sensor_types_to_names[sensor_types]
    except KeyError:
      [sensor_ids] = np.where(np.in1d(self.model.sensor_type, sensor_types))
      sensor_names = [self.model.id2name(s_id, 'sensor') for s_id in sensor_ids]
      self._sensor_types_to_names[sensor_types] = sensor_names
    return sensor_names

  def torso_upright(self):
    """Returns the dot-product of the torso z-axis and the global z-axis."""
    return np.asarray(self.named.data.xmat['torso', 'zz'])

  def torso_velocity(self):
    """Returns the velocity of the torso, in the local frame."""
    return self.named.data.sensordata['velocimeter'].copy()

  def egocentric_state(self):
    """Returns the state without global orientation or position."""
    if not self._hinge_names:
      [hinge_ids] = np.nonzero(self.model.jnt_type ==
                               enums.mjtJoint.mjJNT_HINGE)
      self._hinge_names = [self.model.id2name(j_id, 'joint')
                           for j_id in hinge_ids]
    return np.hstack((self.named.data.qpos[self._hinge_names],
                      self.named.data.qvel[self._hinge_names],
                      self.data.act))

  def toe_positions(self):
    """Returns toe positions in egocentric frame."""
    torso_frame = self.named.data.xmat['torso'].reshape(3, 3)
    torso_pos = self.named.data.xpos['torso']
    torso_to_toe = self.named.data.xpos[_TOES] - torso_pos
    return torso_to_toe.dot(torso_frame)

  def force_torque(self):
    """Returns scaled force/torque sensor readings at the toes."""
    force_torque_sensors = self._get_sensor_names(enums.mjtSensor.mjSENS_FORCE,
                                                  enums.mjtSensor.mjSENS_TORQUE)
    return np.arcsinh(self.named.data.sensordata[force_torque_sensors])

  def imu(self):
    """Returns IMU-like sensor readings."""
    imu_sensors = self._get_sensor_names(enums.mjtSensor.mjSENS_GYRO,
                                         enums.mjtSensor.mjSENS_ACCELEROMETER)
    return self.named.data.sensordata[imu_sensors]

  def rangefinder(self):
    """Returns scaled rangefinder sensor readings."""
    rf_sensors = self._get_sensor_names(enums.mjtSensor.mjSENS_RANGEFINDER)
    rf_readings = self.named.data.sensordata[rf_sensors]
    no_intersection = -1.0
    return np.where(rf_readings == no_intersection, 1.0, np.tanh(rf_readings))

  def origin_distance(self):
    """Returns the distance from the origin to the workspace."""
    return np.asarray(np.linalg.norm(self.named.data.site_xpos['workspace']))

  def origin(self):
    """Returns origin position in the torso frame."""
    torso_frame = self.named.data.xmat['torso'].reshape(3, 3)
    torso_pos = self.named.data.xpos['torso']
    return -torso_pos.dot(torso_frame)

  def ball_state(self):
    """Returns ball position and velocity relative to the torso frame."""
    data = self.named.data
    torso_frame = data.xmat['torso'].reshape(3, 3)
    ball_rel_pos = data.xpos['ball'] - data.xpos['torso']
    ball_rel_vel = data.qvel['ball_root'][:3] - data.qvel['root'][:3]
    ball_rot_vel = data.qvel['ball_root'][3:]
    ball_state = np.vstack((ball_rel_pos, ball_rel_vel, ball_rot_vel))
    return ball_state.dot(torso_frame).ravel()

  def target_position(self):
    """Returns target position in torso frame."""
    torso_frame = self.named.data.xmat['torso'].reshape(3, 3)
    torso_pos = self.named.data.xpos['torso']
    torso_to_target = self.named.data.site_xpos['target'] - torso_pos
    return torso_to_target.dot(torso_frame)

  def ball_to_target_distance(self):
    """Returns horizontal distance from the ball to the target."""
    ball_to_target = (self.named.data.site_xpos['target'] -
                      self.named.data.xpos['ball'])
    return np.linalg.norm(ball_to_target[:2])

  def self_to_ball_distance(self):
    """Returns horizontal distance from the quadruped workspace to the ball."""
    self_to_ball = (self.named.data.site_xpos['workspace']
                    -self.named.data.xpos['ball'])
    return np.linalg.norm(self_to_ball[:2])


def _find_non_contacting_height(physics, orientation, x_pos=0.0, y_pos=0.0):
  """Find a height with no contacts given a body orientation.

  Args:
    physics: An instance of `Physics`.
    orientation: A quaternion.
    x_pos: A float. Position along global x-axis.
    y_pos: A float. Position along global y-axis.
  Raises:
    RuntimeError: If a non-contacting configuration has not been found after
    10,000 attempts.
  """
  z_pos = 0.0  # Start embedded in the floor.
  num_contacts = 1
  num_attempts = 0
  # Move up in 1cm increments until no contacts.
  while num_contacts > 0:
    try:
      with physics.reset_context():
        physics.named.data.qpos['root'][:3] = x_pos, y_pos, z_pos
        physics.named.data.qpos['root'][3:] = orientation
    except control.PhysicsError:
      # We may encounter a PhysicsError here due to filling the contact
      # buffer, in which case we simply increment the height and continue.
      pass
    num_contacts = physics.data.ncon
    z_pos += 0.01
    num_attempts += 1
    if num_attempts > 10000:
      raise RuntimeError('Failed to find a non-contacting configuration.')


def _common_observations(physics):
  """Returns the observations common to all tasks."""
  obs = collections.OrderedDict()
  obs['egocentric_state'] = physics.egocentric_state()
  obs['torso_velocity'] = physics.torso_velocity()
  obs['torso_upright'] = physics.torso_upright()
  obs['imu'] = physics.imu()
  obs['force_torque'] = physics.force_torque()
  return obs


def _upright_reward(physics, deviation_angle=0):
  """Returns a reward proportional to how upright the torso is.

  Args:
    physics: an instance of `Physics`.
    deviation_angle: A float, in degrees. The reward is 0 when the torso is
      exactly upside-down and 1 when the torso's z-axis is less than
      `deviation_angle` away from the global z-axis.
  """
  deviation = np.cos(np.deg2rad(deviation_angle))
  return rewards.tolerance(
      physics.torso_upright(),
      bounds=(deviation, float('inf')),
      sigmoid='linear',
      margin=1 + deviation,
      value_at_margin=0)


class Move(base.Task):
  """A quadruped task solved by moving forward at a designated speed."""

  def __init__(self, desired_speed, random=None):
    """Initializes an instance of `Move`.

    Args:
      desired_speed: A float. If this value is zero, reward is given simply
        for standing upright. Otherwise this specifies the horizontal velocity
        at which the velocity-dependent reward component is maximized.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._desired_speed = desired_speed
    super().__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

    Args:
      physics: An instance of `Physics`.

    """
    # Initial configuration.
    orientation = self.random.randn(4)
    orientation /= np.linalg.norm(orientation)
    _find_non_contacting_height(physics, orientation)
    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation to the agent."""
    return _common_observations(physics)

  def get_reward(self, physics):
    """Returns a reward to the agent."""

    # Move reward term.
    move_reward = rewards.tolerance(
        physics.torso_velocity()[0],
        bounds=(self._desired_speed, float('inf')),
        margin=self._desired_speed,
        value_at_margin=0.5,
        sigmoid='linear')

    return _upright_reward(physics) * move_reward

