
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input
from tensorflow.keras.callbacks import History

from rl.agentcore import AgentCore
from rl.util import *

from rl.callbacks import (
    CallbackList,
    TestLogger,
    TrainEpisodeLogger,
    TrainIntervalLogger,
    Visualizer
)



class Agent(AgentCore):

	def __init__(self, delta_clip=np.inf, custom_model_objects={}, *args, **kwargs):

		super(Agent, self).__init__(*args, **kwargs)


		self.delta_clip = delta_clip
		self.custom_model_objects = custom_model_objects
		self.compiled = False
		self.reset_states()


	def prefit_checks(self):
		if not self.compiled:
			raise RuntimeError('Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')


	def pretest_checks(self):
		if not self.compiled:
			raise RuntimeError('Your tried to test your agent but it hasn\'t been compiled yet. Please call `compile()` before `test()`.')


	def setup_callbacks(self, mode, callbacks, env, steps, visualize, verbose=True, log_interval=10000, episodes=10):
		if mode == 'train':
			callbacks = [] if not callbacks else callbacks[:]
			if verbose:
				callbacks += [TrainIntervalLogger(interval=log_interval)]
			if visualize:
				callbacks += [Visualizer()]
			history = History()
			callbacks += [history]
			callbacks = CallbackList(callbacks)
			if hasattr(callbacks, 'set_model'):
				callbacks.set_model(self)
			else:
				callbacks._set_model(self)
			callbacks._set_env(env)
			params = {
				'steps': steps,
			}
			if hasattr(callbacks, 'set_params'):
				callbacks.set_params(params)
			else:
				callbacks._set_params(params)

		elif mode == 'test':
			callbacks = [] if not callbacks else callbacks[:]
			if verbose:
				callbacks += [TestLogger()]
			if visualize:
				callbacks += [Visualizer()]
			history = History()
			callbacks += [history]
			callbacks = CallbackList(callbacks)
			if hasattr(callbacks, 'set_model'):
				callbacks.set_model(self)
			else:
				callbacks._set_model(self)
			callbacks._set_env(env)
			params = {
				'episodes': episodes,
			}
			if hasattr(callbacks, 'set_params'):
				callbacks.set_params(params)
			else:
				callbacks._set_params(params)
		
		return callbacks, history


	def get_config(self):
		return {
			'model': get_object_config(self.model),
			'policy': get_object_config(self.policy),
			'actions_n': self.actions_n,
			'gamma': self.gamma,
			'batch_size': self.batch_size,
			'warmup_steps': self.warmup_steps,
			'train_interval': self.train_interval,
			'memory_interval': self.memory_interval,
			'target_model_update': self.target_model_update,
			'delta_clip': self.delta_clip,
			'memory': get_object_config(self.memory),
		}


	def compile(self, optimizer, metrics=[]):
		metrics += [mean_q]  # register default metrics

		# We never train the target model, hence we can set the optimizer and loss arbitrarily.
		self.target_model = clone_model(self.model, self.custom_model_objects)
		self.target_model.compile(optimizer='sgd', loss='mse')
		self.model.compile(optimizer='sgd', loss='mse')

		# Compile model.
		if self.target_model_update < 1.:
			# We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
			updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
			optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

		def clipped_masked_error(args):
			y_true, y_pred, mask = args
			loss = huber_loss(y_true, y_pred, self.delta_clip)
			loss *= mask  # apply element-wise mask
			return K.sum(loss, axis=-1)

		# Create trainable model. The problem is that we need to mask the output since we only
		# ever want to update the Q values for a certain action. The way we achieve this is by
		# using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
		# to mask out certain parameters by passing in multiple inputs to the Lambda layer.
		y_pred = self.model.output
		y_true = Input(name='y_true', shape=(self.actions_n,))
		mask = Input(name='mask', shape=(self.actions_n,))
		loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_true, y_pred, mask])
		ins = [self.model.input] if type(self.model.input) is not list else self.model.input
		trainable_model = Model(inputs=ins + [y_true, mask], outputs=[loss_out, y_pred])
		assert len(trainable_model.output_names) == 2
		combined_metrics = {trainable_model.output_names[1]: metrics}
		losses = [
			lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
			lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
		]
		trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
		self.trainable_model = trainable_model

		self.compiled = True


	def compute_batch_q_values(self, state_batch):
		batch = self.processor.process_state_batch(state_batch)
		q_values = self.model.predict_on_batch(batch)
		assert q_values.shape == (len(state_batch), self.actions_n)
		return q_values


	def compute_q_values(self, state):
		q_values = np.asarray(self.compute_batch_q_values([state])).flatten()
		assert q_values.shape == (self.actions_n,)
		return q_values


	def load_weights(self, filepath):
		self.model.load_weights(filepath)
		self.update_target_model_hard()


	def save_weights(self, filepath, overwrite=False):
		self.model.save_weights(filepath, overwrite=overwrite)


	def reset_states(self):
		self.recent_action = None
		self.recent_observation = None
		if self.compiled:
			self.model.reset_states()
			self.target_model.reset_states()


	def update_target_model_hard(self):
		self.target_model.set_weights(self.model.get_weights())



	@property
	def layers(self):
		return self.model.layers[:]

	@property
	def metrics_names(self):
		# Throw away individual losses and replace output name since this is hidden from the user.
		assert len(self.trainable_model.output_names) == 2
		dummy_output_name = self.trainable_model.output_names[1]
		model_metrics = [name for idx, name in enumerate(self.trainable_model.metrics_names) if idx not in (1, 2)]
		model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]

		names = model_metrics + self.policy.metrics_names[:]
		names += self.processor.metrics_names[:]
		return names

	@property
	def policy(self):
		return self.__policy

	@policy.setter
	def policy(self, policy):
		self.__policy = policy
		self.__policy._set_agent(self)
