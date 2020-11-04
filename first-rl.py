import garage
import gym
import torch

from garage.envs import GarageEnv, normalize
from garage.torch import set_gpu_mode
from garage import wrap_experiment
from garage.experiment import deterministic, LocalRunner

from garage.torch.algos import TRPO as Pytorch_TRPO
from garage.torch.policies import GaussianMLPPolicy as Pytorch_GMP
from garage.torch.value_functions import GaussianMLPValueFunction


def trpo_garage_pytorch(env_id):
    
    env = GarageEnv(normalize(gym.make(env_id)))
    
    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)
    
    # using gaussian policy
    policy = Pytorch_GMP(env.spec,
                        hidden_sizes=[32, 32],
                        hidden_nonlinearity=torch.tanh,
                        output_nonlinearity=None)
    
    # using MLP for value approximator
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                             hidden_size=[32, 32],
                                             hidden_nonlinearity=torch.tanh,
                                             output_nonlinearity=None)
    
    # this is good
    algo = Pytorch_TRPO(env_spec=env.spec,
                       policy=policy,
                       value_function=value_function,
                       max_episode_length=100,
                       discount=0.99,
                       gae_lambda=0.97)
    
    
@wrap_experiment
def trpo_garage_pytorch_experiment(ctxt, env_id, seed):
    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)
    deterministic.set_seed(seed)
    
    runner = LocalRunner(ctxt)
    
    env = GarageEnv(normalize(gym.make(env_id)))
    
    # using gaussian policy
    policy = Pytorch_GMP(env.spec,
                        hidden_sizes=[256, 256],
                        hidden_nonlinearity=torch.tanh,
                        output_nonlinearity=None)
    
    # using MLP for value approximator
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                             hidden_size=[256, 256],
                                             hidden_nonlinearity=torch.tanh,
                                             output_nonlinearity=None)
    
    # this is good
    algo = Pytorch_TRPO(env_spec=env.spec,
                       policy=policy,
                       value_function=value_function,
                       max_episode_length=100,
                       discount=0.99,
                       gae_lambda=0.97)
    
    runner.setup(algo, env)
    runner.train(n_epochs=999,
                 batch_size=1024)
    
trpo_garage_pytorch_experiment(None, "BipedalWalker-v2", seed, 42)