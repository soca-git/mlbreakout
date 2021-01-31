
import warnings
from copy import deepcopy

import numpy as np



class AgentCore:

	def __init__(self, model, policy, memory, processor, actions_n, gamma=.99, batch_size=32, warmup_steps=50000,
					train_interval=4, memory_interval=1, target_model_update=10000):

		# Object parameters.
		self.actions_n = actions_n # number of actions the agent can choose from
		self.gamma = gamma # discount value
		self.batch_size = batch_size # size of recorded state, action, reward, next-state transitions used for experience replay
		self.warmup_steps = warmup_steps # number of warmup steps, warmup builds an initial exploratory experience replay (the agent performs random actions)
		self.train_interval = train_interval # specifies the step interval at which training the current network will occur
		self.memory_interval = memory_interval # specifies the step interval at which the experience replay is updated with a new transition
		self.target_model_update = target_model_update # specifies the step interval the target network weights are updated from the current network weights

		# Object vars.
		self.training = False
		self.step = 0

		# Related objects.
		self.model = model # NN model used by network
		self.policy = policy # policy used by q-network
		self.memory = memory # memory storage used for experience replay
		self.processor = processor # processor used to process atari environment states & rewards



	def fit(self, env, steps, callbacks=None, verbose=True, visualize=False, log_interval=10000):

		self.prefit_checks() # check if model has been compiled

		# callbacks setup
		callbacks, history = self.setup_callbacks('train', callbacks, env, steps, visualize, verbose, log_interval)
		callbacks.on_train_begin()

		# training vars
		self.training = True # enable training
		self.step = np.int16(0) # current step
		episode = np.int16(0) # current episode
		observation = None # current observation
		episode_step = None # current episode step
		episode_reward = None #  current episode reward
		abort = False # keyboard interrupt flag


		# keyboard interrupt try & except: safely interrupt training
		try:
			# while current step is less that total number of training steps
			while self.step < steps:
				# if new episode
				if observation is None:  
					callbacks.on_episode_begin(episode)
					episode_step = np.int16(0) # reset episode current step
					episode_reward = np.float32(0) # reset episode reward

					self.reset_states() # reset recent observation & action
					observation = env.reset() # obtain initial observation (of reset environment)
					observation = self.processor.process_state(observation) 


				### RUN A SINGLE STEP ###
				callbacks.on_step_begin(episode_step)

				### FORWARD STEP ###
				action = self.forward(observation) # choose an action
				reward = np.float32(0)
				
				callbacks.on_action_begin(action)

				observation, reward, done, info = env.step(action) # perform action and make observation, collect reward
				observation, reward, done, info = self.processor.process_step(observation, reward, done, info) # process observation & reward

				callbacks.on_action_end(action)

				### BACKWARD STEP ###
				# in training, train the agent
				metrics = self.backward(reward, terminal=done)
				episode_reward += reward

				step_logs = {
					'action': action,
					'observation': observation,
					'reward': reward,
					'metrics': metrics,
					'episode': episode,
					'info': info,
				}
				callbacks.on_step_end(episode_step, step_logs)


				episode_step += 1 # increment to next step of current episode
				self.step += 1 # increment overall current step

				if done:
					# The environment has returned 'done=True', meaning the game has terminated.
					self.forward(observation) # The agent needs to record this, so perform another forward step.
					self.backward(0., terminal=False) # Ignoring the action taken as the environment is terminated, reset

					# This episode is finished, report and reset.
					episode_logs = {
						'episode_reward': episode_reward,
						'nb_episode_steps': episode_step,
						'steps': self.step,
					}
					callbacks.on_episode_end(episode, episode_logs)

					episode += 1 # increment current episode
					observation = None # reset


		except KeyboardInterrupt:
			# We catch keyboard interrupts here so that training can be be safely aborted.
			# This is so common that we've built this right into this function, which ensures that
			# the `on_train_end` method is properly called.
			abort = True

		callbacks.on_train_end(logs={'did_abort': abort})

		return history



	def test(self, env, episodes=10, callbacks=None, visualize=True, verbose=True):

		self.prefit_checks() # check if model has been compiled
		
		# callbacks setup
		callbacks, history = self.setup_callbacks('test', callbacks, env, visualize, verbose, episodes)
		callbacks.on_train_begin()

		# test vars
		self.training = False
		self.step = 0

		# for each episode
		for episode in range(episodes):
			callbacks.on_episode_begin(episode)
			episode_step = np.int16(0)
			episode_reward = np.float32(0)

			# Obtain the initial observation by resetting the environment.
			self.reset_states()
			observation = env.reset()
			observation = self.processor.process_state(observation) 
			
			
			# run the test until the environment terminates
			done = False
			while not done:
				callbacks.on_step_begin(episode_step)

				### FORWARD STEP ###
				action = self.forward(observation) # choose an action
				reward = np.float32(0)

				callbacks.on_action_begin(action)

				observation, reward, done, info = env.step(action) # perform action and make observation, collect reward
				observation, reward, done, info = self.processor.process_step(observation, reward, done, info) # process observation & reward

				callbacks.on_action_end(action)

				### BACKWARD STEP ###
				# in test, do nothing
				self.backward(reward, terminal=done)
				episode_reward += reward

				step_logs = {
					'action': action,
					'observation': observation,
					'reward': reward,
					'episode': episode,
					'info': info,
				}
				callbacks.on_step_end(episode_step, step_logs)

				episode_step += 1 # increment to next step of current episode
				self.step += 1 # increment overall current step

			# The environment has returned 'done=True', meaning the game has terminated.
			self.forward(observation) # The agent needs to record this, so perform another forward step.
			self.backward(0., terminal=False) # Ignoring the action taken as the environment is terminated, reset

			# This episode is finished, report and reset.
			episode_logs = {
				'episode_reward': episode_reward,
				'steps': episode_step,
			}
			callbacks.on_episode_end(episode, episode_logs)


		callbacks.on_train_end()

		return history



	def forward(self, observation):
		"""
			FORWARD STEP
			Given an observation, calculate the q-values and choose an action
		"""
		state = self.memory.get_recent_state(observation) # get the most recent recorded state
		q_values = self.compute_q_values(state) # compute the q-values of this state

		action = self.policy.select_action(q_values=q_values) # select an action using the (linearly annealed) epsilon greedy policy

		# update recent observation & action
		self.recent_observation = observation
		self.recent_action = action

		return action



	def backward(self, reward, terminal):
		"""
			BACKWARD STEP (mode: train)
			Given a reward and terminal state, depending on the set intervals, perform the following:
				- store experience into agent's memory
				- train the agent by sampling the agent's memory in batches, computing the q-values and training the target network
		"""
		
		# if memory interval
		if self.step % self.memory_interval == 0:
			# record the most recent state, action, reward & terminal boolean into the agent's memory
			self.memory.append(self.recent_observation, self.recent_action, reward, terminal, training=self.training)

		metrics = [np.nan for _ in self.metrics_names] # reset metrics list

		# if not in training mode, do nothing here
		# i.e. don't perform any training
		if not self.training:
			return metrics

		# Otherwise, train the network
		# If the warmup phase is over and it is a training interval
		if self.step > self.warmup_steps and self.step % self.train_interval == 0:
			experiences = self.memory.sample(self.batch_size) # sample previous experiences from the agent's memory

			# initialize experience parameter lists
			state0_batch = []
			reward_batch = []
			action_batch = []
			terminal1_batch = []
			state1_batch = []
			# for each experience, collect the state, action, next state, reward & terminal boolean
			for e in experiences:
				state0_batch.append(e.state0) # state
				state1_batch.append(e.state1) # next state
				reward_batch.append(e.reward) # reward
				action_batch.append(e.action) # action
				terminal1_batch.append(0. if e.terminal1 else 1.) # next state terminal boolean

			# process states in batch
			state0_batch = self.processor.process_state_batch(state0_batch)
			state1_batch = self.processor.process_state_batch(state1_batch)
			# convert terminal & reward batches to array
			terminal1_batch = np.array(terminal1_batch)
			reward_batch = np.array(reward_batch)

			# Compute Q values from the experience batch
			# predict q-values using 'next_state' batch, using the target-network
			# the target-network's weights are updated less often compared to the 'live' network
			# this is governed by the target_model_update interval
			# the target-network is a more stable version of the live network
			target_q_values = self.target_model.predict_on_batch(state1_batch)
			q_batch = np.max(target_q_values, axis=1).flatten() # select the highest q-values from each experience
			
			# Compute r_t + gamma * max_a Q(s_t+1, a)
			discounted_reward_batch = self.gamma * q_batch # calculate the discounted q-values for each experience
			discounted_reward_batch *= terminal1_batch # for the terminal states, set the discounted reward to 0
			Rs = reward_batch + discounted_reward_batch # for each experience, calculate the total reward (current + discounted)

			# build targets, dummy_targets & masks arrays
			targets = np.zeros((self.batch_size, self.actions_n))
			dummy_targets = np.zeros((self.batch_size,))
			masks = np.zeros((self.batch_size, self.actions_n))
			for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
				target[action] = R  # update action with estimated accumulated reward
				dummy_targets[idx] = R
				mask[action] = 1.  # enable loss for this specific action
			targets = np.array(targets).astype('float32')
			masks = np.array(masks).astype('float32')

			# Finally, perform a single update on the entire batch. We use a dummy target since
			# the actual loss is computed in a Lambda layer that needs more complex input. However,
			# it is still useful to know the actual target to compute metrics properly.
			ins = [state0_batch] if type(self.model.input) is not list else state0_batch
			# train the model (original model updated with loss function lambda layer & output masking (see compile method of agent))
			metrics = self.trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets])
			metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]  # throw away individual losses
			metrics += self.policy.metrics
			metrics += self.processor.metrics

		# if target-model-update interval
		if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
			self.update_target_model_hard() # update the target model's weights to the current network's weights

		return metrics
