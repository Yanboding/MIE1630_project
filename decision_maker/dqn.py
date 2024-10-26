import tensorflow as tf
import numpy as np
import random
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size, discount_factor=0.99, learning_rate=0.001, batch_size=64, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, target_update=10):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update = target_update

        self.memory = deque(maxlen=2000)
        
        self.q_network = self.build_network()
        self.target_network = self.build_network()
        self.update_target_network() 
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_function = tf.keras.losses.MeanSquaredError()

    def build_network(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear') 
        ])
        return model
    
    def update_target_network(self):
        """Copy weights from the Q-network to the target network."""
        self.target_network.set_weights(self.q_network.get_weights())

    def act(self, state):
        """Epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train the network using experiences from the replay buffer."""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
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

def train_dqn(agent, env, episodes=500):
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        total_reward = 0
        for time in range(500): 
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                print(f"Episode: {e}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2}")
                break
            agent.replay()

        if e % agent.target_update == 0:
            agent.update_target_network()