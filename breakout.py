
#=== IMPORTS ===#

# core
import argparse

# openai gym
import gym

# tensorflow
from tensorflow.keras.optimizers import Adam

# local modules
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl.memory import SequentialMemory
from rl.processors import AtariProcessor
from rl.models import MinhModel
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.agent import Agent



#=== ARGUMENT PARSER ===#
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
args = parser.parse_args()



#=== SETUP ===#

# environment
env = gym.make('BreakoutDeterministic-v4')
env.seed(123)
actions = env.action_space

# model
model = MinhModel()
model.summary()

# memory
memory = SequentialMemory(limit=1000000, window_length=4)

# atari processor
processor = AtariProcessor()

# policy
# Here we select an epsilon-greedy policy, wrapped in a linear-annealed policy. This means the value for epsilon will start high and decay
# over time. For the agent, this translates into high exploration at the beginning of training. As the training progresses, the agent's
# exploration will decrease and it's actions will be those selected by the q-network.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=1000000)

# agent
# 
dqn = Agent(model=model.model, actions_n=actions.n, policy=policy, memory=memory,
               processor=processor, warmup_steps=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)

# learning rate
# 
dqn.compile(Adam(lr=.00025), metrics=['mae'])



#=== TRAIN ===#

if args.mode == 'train':
    checkpoint_weights_filename = 'weights_{step}.h5f'
    log_filename = 'dqn_log.json'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]


    dqn.fit(env, callbacks=callbacks, steps=1750000, log_interval=10000)

    # After training is done, we save the final weights
    dqn.save_weights('final_weights.h5f', overwrite=True)



#=== TEST ===#

elif args.mode == 'test':
    dqn.load_weights('trained_data/final_weights.h5f')
    dqn.test(env, episodes=10, visualize=True)
