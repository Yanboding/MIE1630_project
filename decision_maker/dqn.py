import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import random
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size, discount_factor=0.99, learning_rate=1e-3, batch_size=256, epsilon=0.8, epsilon_decay=0.9, epsilon_min=0.01, target_update=50):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update = target_update
        self.replay_buffer = deque(maxlen=2000) # replay buffer of size 2000
        self.q_network = self.build_network() # policy net
        self.target_network = self.build_network() # target net Q_hat
        self.update_target_network() 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_function = tf.keras.losses.MeanSquaredError()

    def build_network(self):
        network = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear') 
        ])
        return network
    
    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    @tf.function
    def action(self, state):
        """
        Epsilon-greedy policy for selecting actions.
        """
        random_number = np.random.rand()
        if random_number <= self.epsilon:
            random_action = []
            for i in range(self.action_size):
                random_action.append(random.randrange(self.action_size))
            return np.array(random_action)
        else:
            q_values = self.q_network(state) 
            epsilon_greedy_action = tf.argmax(q_values[0], axis=-1)
            return epsilon_greedy_action 


    def put_into_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    @tf.function
    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = self.q_network.predict(state)
            if done:
                target[0][action] = reward
            else:
                q_next = np.max(self.target_network.predict(next_state)[0])
                target[0][action] = reward + self.discount_factor * q_next
            
            with tf.GradientTape() as tape:
                q_values = self.q_network(state)
                loss = self.loss_function(target, q_values)
            grads = tape.gradient(loss, self.q_network.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.q_network.load_weights(name)

    def save(self, name):
        self.q_network.save_weights(name)

    def train_dqn(self, env, episodes=500):
        for i in range(episodes):
            state, info = env.reset()
            state = np.reshape(state, [1, self.state_size])
            total_reward = 0
            for time in range(1000):
                action = self.action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                self.put_into_replay_buffer(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done or truncated:
                    print(f"Episode: {i}/{episodes}, Reward: {total_reward}, Epsilon: {self.epsilon:.2}")
                    break
                self.replay()
            if i % self.target_update == 0:
                self.update_target_network()


