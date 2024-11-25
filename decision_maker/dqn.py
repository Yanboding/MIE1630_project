import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import random
from collections import deque

tf.config.run_functions_eagerly(True)

class DQNAgent:
    def __init__(self, state_size, action_size, env, discount_factor=0.99, learning_rate=1e-3, batch_size=128, epsilon=0.25, epsilon_decay=0.995, epsilon_min=0.01, target_update=25, initial_state=[0,0,5,5]):
        self.state_size = state_size
        self.action_size = action_size 
        self.env = env
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update = target_update
        self.replay_buffer = deque(maxlen=1024)
        self.q_network = self.build_network()
        self.target_network = self.build_network()
        self.update_target_network() 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.initial_state=initial_state

    def build_network(self):
        state_input = tf.keras.layers.Input(shape=(self.state_size,))
        action_input = tf.keras.layers.Input(shape=(self.action_size,))
        time_input = tf.keras.layers.Input(shape=(1,))
        concatenated = tf.keras.layers.Concatenate()([state_input, action_input, time_input])
        
        x = tf.keras.layers.Dense(256, activation='relu')(concatenated)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        
        q_value = tf.keras.layers.Dense(1, activation='linear')(x)
        return tf.keras.Model(inputs=[state_input, action_input, time_input], outputs=q_value)
    
    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    @tf.function
    def action(self, state, t, train=True):
        state = tf.expand_dims(state, axis=0)
        time = tf.expand_dims(t, axis=0)
        
        lower_bounds = np.zeros(state[0][self.env.num_sessions:].shape, dtype=int)
        upper_bounds = np.array(state[0][self.env.num_sessions:], dtype=int) + 1 
        
        random_number = tf.random.uniform((), 0, 1)
        if random_number <= self.epsilon and train:
            action = np.random.randint(lower_bounds, upper_bounds, size=self.action_size)
        else:
            max_q_value = -float('inf')
            best_action = None
            valid_actions = self.env.valid_allocation_actions(state[0], t=t)
            for action in valid_actions:
                action_input = tf.expand_dims(action, axis=0) 
                q_value = self.q_network([state, action_input, time])[0, 0]
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_action = action
            action = best_action
        return action

    def put_into_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
            
    @tf.function
    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        minibatch = random.sample(self.replay_buffer, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = tf.expand_dims(state, axis=0)
            next_state = tf.expand_dims(next_state, axis=0)
            action_input = tf.expand_dims(action, axis=0)
            time_input = tf.expand_dims(self.env.t, axis=0)

            q_value = self.q_network([state, action_input, time_input])[0, 0]

            if done:
                target_q_value = reward
            else:
                max_q_next = -float('inf')
                
                lower_bounds = tf.zeros_like(next_state[0][self.env.num_sessions:], dtype=tf.int32)
                upper_bounds = tf.cast(next_state[0][self.env.num_sessions:], tf.int32) + 1

                feasible_action_size = np.prod(upper_bounds)
                
                for _ in range(feasible_action_size):
                    next_action = tf.stack([
                        tf.random.uniform(
                            shape=(),
                            minval=lower_bounds[i],
                            maxval=upper_bounds[i],
                            dtype=tf.int32
                        )
                        for i in range(self.action_size)
                    ])
                    next_action_input = tf.expand_dims(next_action, axis=0)
                    next_time_input = tf.expand_dims(self.env.t + 1, axis = 0)
                    q_next = self.target_network([next_state, next_action_input, next_time_input])[0, 0]
                    max_q_next = tf.maximum(max_q_next, q_next)

                target_q_value = reward + self.discount_factor * max_q_next

            with tf.GradientTape() as tape:
                q_value = self.q_network([state, action_input, time_input])[0, 0]
                loss = self.loss_function(tf.expand_dims(target_q_value, axis=0), tf.expand_dims(q_value, axis=0))

            grads = tape.gradient(loss, self.q_network.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_dqn(self, episode = 1000, init_state=[0,0,0,0], t=0, threshold=1.01, optimal_agent=None, decision_periods=5):
        print("Begin Training")
        expected_total_reward = 0
        for i in range(1, episode+1):
            state, info = self.env.reset(init_state, t)
            total_reward = 0
            for time_step in range(1, 10001):
                action = self.action(state, time_step)
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                self.put_into_replay_buffer(state, action, reward, next_state, done)
                total_reward += self.discount_factor**(time_step-1)*reward
                if done or truncated:
                    break
                state = next_state
                self.replay()
            expected_total_reward += (total_reward - expected_total_reward)/i
            print(f"Episode: {i}/{episode}, Reward: {expected_total_reward}, Epsilon: {self.epsilon:.2}")
            if i % self.target_update == 0:
                self.update_target_network()
            if ((threshold*optimal_agent <= expected_total_reward <= (threshold+0.1)*optimal_agent) and i >= 30) or (expected_total_reward >= (threshold+0.1)*optimal_agent and i >= 40) or  (expected_total_reward <= threshold*optimal_agent and i >= 40):
                break
        print("Training Ended.")

    def save_model(self, model_path):
        self.q_network.save(model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        self.q_network = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")