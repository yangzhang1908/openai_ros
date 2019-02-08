import rospy
import numpy
from gym import spaces
from gym import envs
import sys
sys.path.append('/home/yangz/catkin_ws/src/mp500lwa4d_openai_ros')
from robot_envs import mp500lwa4d_env
from gym.envs.registration import register

timestep_limit_per_episode = 10000  # Can be any Value

register(
    id='Mp500Lwa4dWorld',
    entry_point='mp500lwa4d_openai_ros:task_envs.mp500lwa4d.mp500lwa4d_world.Mp500Lwa4dWorldEnv',
    timestep_limit=timestep_limit_per_episode,
)


print(envs.registry.all())