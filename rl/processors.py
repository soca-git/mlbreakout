
from PIL import Image
import numpy as np



class AtariProcessor():

    def process_step(self, observation, reward, done, info):
        observation = self.process_state(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info

    def process_state(self, state, shape=(84, 84), convert='L'):
        img = Image.fromarray(state)
        img = img.resize(shape).convert(convert)  # resize and convert to grayscale
        state = np.array(img)
        return state.astype('uint8')  # saves storage memory

    def process_state_batch(self, batch):
        batch = np.array(batch)
        # We could perform this processing step in `process_state`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

    def process_info(self, info):
        return info

    def process_action(self, action):
        return action



    @property
    def metrics(self):
        return []

    @property
    def metrics_names(self):
        return []
