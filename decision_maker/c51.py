import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import random
from collections import deque

class C51Agent:
    def __init__(self, state_size, action_size, v_min=-10, v_max=10, num_atoms=51, discount_factor=0.99, learning_rate=1e-3, batch_size=256, epsilon=0.99, epsilon_decay=0.95, epsilon_min=0.01, target_update=50):
        self.state_size = state_size
        self.action_size = action_size
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.delta_z = (v_max - v_min) / (self.num_atoms - 1)
        self.z = tf.linspace(v_min, v_max, num_atoms)
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update = target_update
        self.replay_buffer = deque(maxlen=2000)
        self.q_network = self.build_network()
        self.target_network = self.build_network()
        self.update_target_network()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def build_network(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.action_size * self.num_atoms, activation='softmax')
        ])
        return model

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            distribution = self.q_network(state)
            distribution = tf.reshape(distribution, (self.action_size, self.num_atoms))
            q_values = tf.reduce_sum(distribution * self.z, axis=1)
            return tf.argmax(q_values).numpy()

    def put_into_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        minibatch = random.sample(self.replay_buffer, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            m = np.zeros((self.batch_size, self.num_atoms), dtype=np.float32)
            target_distribution = self.q_network(next_state)
            target_distribution = tf.reshape(target_distribution, (self.action_size, self.num_atoms))

            if done:
                m[action] = tf.reduce_sum(self.z == reward)
            else:
                next_action = tf.argmax(tf.reduce_sum(target_distribution * self.z, axis=1))
                m[action] = reward + self.discount_factor * target_distribution[next_action]

            with tf.GradientTape() as tape:
                distribution = self.q_network(state)
                loss = tf.keras.losses.KLD(m, distribution)
            grads = tape.gradient(loss, self.q_network.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
