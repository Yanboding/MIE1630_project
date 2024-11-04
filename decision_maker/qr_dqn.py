import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import random
from collections import deque

class QRDQNAgent:
    def __init__(self, state_size, action_size, num_quantiles=51, discount_factor=0.99, learning_rate=1e-3, batch_size=256, epsilon=0.99, epsilon_decay=0.95, epsilon_min=0.01, target_update=50):
        self.state_size = state_size
        self.action_size = action_size
        self.num_quantiles = num_quantiles
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
            tf.keras.layers.Dense(self.action_size * self.num_quantiles)
        ])
        return model

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            quantile_values = self.q_network(state)
            quantile_values = tf.reshape(quantile_values, (self.action_size, self.num_quantiles))
            mean_q_values = tf.reduce_mean(quantile_values, axis=1)
            return tf.argmax(mean_q_values).numpy()

    def put_into_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        minibatch = random.sample(self.replay_buffer, self.batch_size)

        tau = tf.constant((2 * np.arange(self.num_quantiles) + 1) / (2.0 * self.num_quantiles), dtype=tf.float32)

        for state, action, reward, next_state, done in minibatch:
            quantiles = self.q_network(state)
            quantiles = tf.reshape(quantiles, (self.action_size, self.num_quantiles))

            if done:
                target_quantiles = tf.constant([reward] * self.num_quantiles, dtype=tf.float32)
            else:
                next_quantiles = self.target_network(next_state)
                next_quantiles = tf.reshape(next_quantiles, (self.action_size, self.num_quantiles))
                mean_next_q_values = tf.reduce_mean(next_quantiles, axis=1)
                best_next_action = tf.argmax(mean_next_q_values)
                target_quantiles = reward + self.discount_factor * next_quantiles[best_next_action]

            quantile_loss = self.quantile_loss(quantiles[action], target_quantiles, tau)
            with tf.GradientTape() as tape:
                grads = tape.gradient(quantile_loss, self.q_network.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def quantile_loss(self, quantiles, target, tau):
        delta = target - quantiles
        loss = tf.reduce_mean(tau * tf.nn.relu(delta) + (1 - tau) * tf.nn.relu(-delta))
        return loss
