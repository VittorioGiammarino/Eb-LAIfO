import collections
import os
import random
import xml.etree.ElementTree as ET

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import xml_tools

from lxml import etree
import numpy as np

from dm_control.utils import io as resources

from dmc_remastered import DMCR_VARY, SUITE_DIR, register

from .generate_visuals import get_assets
from .rng import dmcr_random

def get_model(visual_seed, vary=["camera", "light"]):
    with open(os.path.join(SUITE_DIR, os.path.join("assets", "rodent.xml")), "r") as f:
        xml = ET.fromstring(f.read())
    if visual_seed != 0:
        with dmcr_random(visual_seed):
            # camera_x = random.uniform(-0.1, 0.1)
            # camera_y = random.uniform(-0.35, -0.45)
            # camera_z = random.uniform(0.05, 0.15)

            light_x = random.uniform(-0.5, 0.5)
            light_y = random.uniform(-0.5, 0.5)
            light_z = random.uniform(2.5, 3.5)
        # if "camera" in vary:
        #     xml[10][1][2].attrib["pos"] = f"{camera_x} {camera_y} {camera_z}"
        if "light" in vary:
            xml[9][1][0].attrib["pos"] = f"{light_x} {light_y} {light_z}"
        
    return ET.tostring(xml, encoding="utf8", method="xml")


_UPRIGHT_POS = (0.0, 0.0, 0.0)
_UPRIGHT_QUAT = (1., 0., 0., 0.)

_DEFAULT_TIME_LIMIT = 20
_CONTROL_TIMESTEP = .02

# Angle (in degrees) of local z from global z below which upright reward is 1.
_MAX_UPRIGHT_ANGLE = 30
_MIN_UPRIGHT_COSINE = np.cos(np.deg2rad(_MAX_UPRIGHT_ANGLE))

# Standing reward is 1 for body-over-foot height that is at least this fraction
# of the height at the default pose.
_STAND_HEIGHT_FRACTION = 0.9

# Torques which enforce joint range limits should stay below this value.
_EXCESSIVE_LIMIT_TORQUES = 60

# Horizontal speed above which Move reward is 1.
_WALK_SPEED = 1.0
_RUN_SPEED = 3.0

_HINGE_TYPE = mujoco.wrapper.mjbindings.enums.mjtJoint.mjJNT_HINGE
_LIMIT_TYPE = mujoco.wrapper.mjbindings.enums.mjtConstraint.mjCNSTR_LIMIT_JOINT

SUITE = containers.TaggedTasks()

_ASSET_DIR = os.path.join(SUITE_DIR, os.path.join("assets", "rodent_assets"))

def get_assets_rodent():
  """Returns a tuple containing the model XML string and a dict of assets."""
  assets = common.ASSETS.copy()
  _, _, filenames = next(resources.WalkResources(_ASSET_DIR))
  for filename in filenames:
    assets[filename] = resources.GetResource(os.path.join(_ASSET_DIR, filename))
  return assets

@register("rodent", "stand")
def stand(time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY):
  """Returns the Stand task."""
  model = get_model(visual_seed, vary)
  assets = get_assets_rodent()
  physics = Physics.from_xml_string(model, assets)
  task = Stand(random=dynamics_seed)
  return control.Environment(physics, task, time_limit=time_limit,
                             control_timestep=_CONTROL_TIMESTEP)


@register("rodent", "walk")
def walk(time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY):
  """Returns the Walk task."""
  move_speed = _WALK_SPEED
  model = get_model(visual_seed, vary)
  assets = get_assets_rodent()
  physics = Physics.from_xml_string(model, assets)
  task = Move(move_speed=move_speed, random=dynamics_seed)
  return control.Environment(physics, task, time_limit=time_limit,
                             control_timestep=_CONTROL_TIMESTEP)

@register("rodent", "run")
def run(time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY):
  """Returns the Run task."""
  move_speed = _RUN_SPEED
  model = get_model(visual_seed, vary)
  assets = get_assets_rodent()
  physics = Physics.from_xml_string(model, assets)
  task = Move(move_speed=move_speed, random=dynamics_seed)
  return control.Environment(physics, task, time_limit=time_limit,
                             control_timestep=_CONTROL_TIMESTEP)

class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Dog domain."""

  def torso_pelvis_height(self):
    """Returns the height of the torso."""
    return self.named.data.xpos[['torso', 'pelvis'], 'z']

  def z_projection(self):
    """Returns rotation-invariant projection of local frames to the world z."""
    return np.vstack((self.named.data.xmat['skull', ['zx', 'zy', 'zz']],
                      self.named.data.xmat['torso', ['zx', 'zy', 'zz']],
                      self.named.data.xmat['pelvis', ['zx', 'zy', 'zz']],))
                      # self.named.data.xmat['vertebra_1', ['zx', 'zy', 'zz']],
                      # self.named.data.xmat['vertebra_2', ['zx', 'zy', 'zz']],
                      # self.named.data.xmat['vertebra_3', ['zx', 'zy', 'zz']],
                      # self.named.data.xmat['vertebra_4', ['zx', 'zy', 'zz']],
                      # self.named.data.xmat['vertebra_5', ['zx', 'zy', 'zz']],
                      # self.named.data.xmat['vertebra_6', ['zx', 'zy', 'zz']],
                      # self.named.data.xmat['vertebra_C1', ['zx', 'zy', 'zz']],
                      # self.named.data.xmat['vertebra_C5', ['zx', 'zy', 'zz']],
                      # self.named.data.xmat['vertebra_C10', ['zx', 'zy', 'zz']],
                      # self.named.data.xmat['vertebra_C15', ['zx', 'zy', 'zz']],
                      # self.named.data.xmat['vertebra_C20', ['zx', 'zy', 'zz']],
                      # self.named.data.xmat['foot_L', ['zx', 'zy', 'zz']],
                      # self.named.data.xmat['toe_L', ['zx', 'zy', 'zz']],
                      # self.named.data.xmat['foot_R', ['zx', 'zy', 'zz']],
                      # self.named.data.xmat['toe_R', ['zx', 'zy', 'zz']],))

  def upright(self):
    """Returns projection from local z-axes to the z-axis of world."""
    return self.z_projection()[:, 2]

  def center_of_mass_velocity(self):
    """Returns the velocity of the center-of-mass."""
    return self.named.data.sensordata['torso_linvel']

  def torso_com_velocity(self):
    """Returns the velocity of the center-of-mass in the torso frame."""
    torso_frame = self.named.data.xmat['torso'].reshape(3, 3).copy()
    return self.center_of_mass_velocity().dot(torso_frame)

  def com_forward_velocity(self):
    """Returns the com velocity in the torso's forward direction."""
    return self.torso_com_velocity()[0]

  def joint_angles(self):
    """Returns the configuration of all hinge joints (skipping free joints)."""
    hinge_joints = self.model.jnt_type == _HINGE_TYPE
    qpos_index = self.model.jnt_qposadr[hinge_joints]
    return self.data.qpos[qpos_index].copy()

  def joint_velocities(self):
    """Returns the velocity of all hinge joints (skipping free joints)."""
    hinge_joints = self.model.jnt_type == _HINGE_TYPE
    qvel_index = self.model.jnt_dofadr[hinge_joints]
    return self.data.qvel[qvel_index].copy()

  def inertial_sensors(self):
    """Returns inertial sensor readings."""
    return self.named.data.sensordata[['accelerometer', 'velocimeter', 'gyro']]

  def touch_sensors(self):
    """Returns touch readings."""
    return self.named.data.sensordata[['palm_L', 'palm_R', 'sole_L', 'sole_R']]

class Stand(base.Task):
  """A rat stand task generating upright posture."""

  def __init__(self, random=None, observe_reward_factors=False):
    """Initializes an instance of `Stand`.

    Args:
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
      observe_reward_factors: Boolean, whether the factorised reward is a
        key in the observation dict returned to the agent.
    """
    self._observe_reward_factors = observe_reward_factors
    super().__init__(random=random)

  def initialize_episode(self, physics):
    """Randomizes initial root velocities and actuator states.

    Args:
      physics: An instance of `Physics`.

    """
    physics.reset()

    # Measure stand heights from default pose, above which stand reward is 1.
    self._stand_height = physics.torso_pelvis_height() * _STAND_HEIGHT_FRACTION

    # Measure body weight.
    body_mass = physics.named.model.body_subtreemass['torso']
    self._body_weight = -physics.model.opt.gravity[2] * body_mass

    # Randomize actuator states.
    assert physics.model.nu == physics.model.na
    for actuator_id in range(physics.model.nu):
      ctrlrange = physics.model.actuator_ctrlrange[actuator_id]
      physics.data.act[actuator_id] = self.random.uniform(*ctrlrange)

  def get_observation_components(self, physics):
    """Returns the observations for the Stand task."""
    obs = collections.OrderedDict()
    obs['joint_angles'] = physics.joint_angles()
    obs['joint_velocites'] = physics.joint_velocities()
    obs['torso_pelvis_height'] = physics.torso_pelvis_height()
    obs['z_projection'] = physics.z_projection().flatten()
    obs['torso_com_velocity'] = physics.torso_com_velocity()
    obs['inertial_sensors'] = physics.inertial_sensors()
    obs['touch_sensors'] = physics.touch_sensors()
    obs['actuator_state'] = physics.data.act.copy()
    return obs

  def get_observation(self, physics):
    """Returns the observation, possibly adding reward factors."""
    obs = self.get_observation_components(physics)
    if self._observe_reward_factors:
      obs['reward_factors'] = self.get_reward_factors(physics)
    return obs

  def get_reward_factors(self, physics):
    """Returns the factorized reward."""
    # Keep the torso  at standing height.
    torso = rewards.tolerance(physics.torso_pelvis_height()[0],
                              bounds=(self._stand_height[0], float('inf')),
                              margin=self._stand_height[0])
    # Keep the pelvis at standing height.
    pelvis = rewards.tolerance(physics.torso_pelvis_height()[1],
                               bounds=(self._stand_height[1], float('inf')),
                               margin=self._stand_height[1])
    # Keep body upright.
    upright = rewards.tolerance(physics.upright(),
                                bounds=(_MIN_UPRIGHT_COSINE, float('inf')),
                                sigmoid='linear',
                                margin=_MIN_UPRIGHT_COSINE+1,
                                value_at_margin=0)

    # Reward for foot touch forces up to bodyweight.
    touch = rewards.tolerance(physics.touch_sensors().sum(),
                              bounds=(self._body_weight, float('inf')),
                              margin=self._body_weight,
                              sigmoid='linear',
                              value_at_margin=0.9)

    return np.hstack((torso, pelvis, upright, touch))

  def get_reward(self, physics):
    """Returns the reward, product of reward factors."""
    return np.product(self.get_reward_factors(physics))


class Move(Stand):
  """A dog move task for generating locomotion."""

  def __init__(self, move_speed, random, observe_reward_factors=False):
    """Initializes an instance of `Move`.

    Args:
      move_speed: A float. Specifies a target horizontal velocity.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
      observe_reward_factors: Boolean, whether the factorised reward is a
        component of the observation dict.
    """
    self._move_speed = move_speed
    super().__init__(random, observe_reward_factors)

  def get_reward_factors(self, physics):
    """Returns the factorized reward."""
    standing = super().get_reward_factors(physics)

    speed_margin = max(1.0, self._move_speed)
    
    forward = rewards.tolerance(physics.com_forward_velocity(),
                                bounds=(self._move_speed, self._move_speed),
                                margin=speed_margin,
                                value_at_margin=0,
                                sigmoid='linear')
    forward = (4*forward + 1) / 5

    return np.hstack((standing, forward))
