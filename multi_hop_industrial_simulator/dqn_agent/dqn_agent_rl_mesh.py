import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from multi_hop_industrial_simulator.utils.read_inputs import read_inputs

######## Parameters ########

inputs_yaml = read_inputs('inputs.yaml')
n_actions = inputs_yaml.get('rl').get('agent').get('n_actions')  # == env.action_space.n
discount_factor = inputs_yaml.get('rl').get('agent').get('discount_factor')
learning_rate = inputs_yaml.get('rl').get('agent').get('learning_rate')

"""model_prototype = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="elu"),
    tf.keras.layers.Dense(32, activation="elu"),
    tf.keras.layers.Dense(n_actions)
])"""

optimizer = tf.keras.optimizers.Nadam(
    learning_rate=learning_rate)  # tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=1.0)
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
# loss_fn = tf.keras.losses.mean_squared_error
loss_fn = tf.keras.losses.Huber()  # tf.keras.losses.MeanSquaredError()


######## Utility functions of the RL agent ########
# Build the DNN model for the DQN agent
def get_model(input_n_actions, input_n_nodes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation="elu", input_shape=(input_n_nodes,)),
        tf.keras.layers.Dense(32, activation="elu"),
        tf.keras.layers.Dense(input_n_actions)
    ])
    return model


# Build the Dueling DNN model for the DQN agent
def get_dueling_model(input_n_actions, input_n_nodes):
    # Input Layer
    inputs = layers.Input(shape=(input_n_nodes,))

    # Shared layers (used by both the value and advantage streams)
    x = layers.Dense(32, activation="elu")(inputs)
    x = layers.Dense(32, activation="elu")(x)

    # State-value stream
    state_value = layers.Dense(1)(x)

    # Advantage stream
    advantage = layers.Dense(input_n_actions)(x)

    # Combine the state-value and advantage to get the Q-values
    q_values = state_value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=q_values)

    return model


# Choose the action based on the epsilon-greedy policy
def epsilon_greedy_policy(state, input_n_actions, model, categorical_dqn=None, input_epsilon=0):
    if np.random.rand() < input_epsilon:
        return np.random.randint(input_n_actions)  # random action
    else:
        if categorical_dqn is None:
            # state = obs[0]
            state_tensor = tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32)
            q_values = model.predict(state_tensor, verbose=0)[0]
            # q_values = model.predict(state[np.newaxis], verbose=0)[0]
            return q_values.argmax()  # optimal action according to the DQN
        else:
            state_tensor = tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32)
            q_values = model.predict(state_tensor, verbose=0)[0]
            q_values = np.sum(q_values * categorical_dqn.z, axis=1)
            return np.argmax(q_values)


# Sample experiences from the replay buffer
def sample_experiences(input_batch_size, replay_buffer):
    indices = np.random.randint(len(replay_buffer), size=input_batch_size)
    batch = [replay_buffer[index] for index in indices]

    return [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(len(replay_buffer[0]))
    ]  # [states, actions, rewards, next_states, dones]


# New function to sample experiences from the replay buffer with half of the batch size of action 1 and half of action 0
"""def sample_experiences(input_batch_size, replay_buffer):
    # Separate experiences based on action
    # random_0 = False
    # random_1 = False
    action_0_experiences = [exp for exp in replay_buffer if exp[1] == 0]
    action_1_experiences = [exp for exp in replay_buffer if exp[1] == 1]
    #print(len(action_0_experiences))
    #print(len(action_1_experiences))
    # print(action_0_experiences[-1])
    # Determine half the batch size
    half_batch_size = input_batch_size / 2

    # Sample half_batch_size experiences from each action group

    indices = np.random.randint(len(action_0_experiences), size=input_batch_size)
    batch_0 = [action_0_experiences[index] for index in indices]
    indices = np.random.randint(len(action_1_experiences), size=input_batch_size)
    batch_1 = [action_1_experiences[index] for index in indices]
    # Combine the two halves
    batch = list(batch_0) + list(batch_1)

    # Shuffle the batch to mix experiences
    np.random.shuffle(batch)

    # Create the batch in the required format
    return [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(len(replay_buffer[0]))
    ]  # [states, actions, rewards, next_states, dones]"""

"""def play_one_step(input_env, state, input_n_actions, input_epsilon, replay_buffer, model):
    action = epsilon_greedy_policy(state[0], input_n_actions, model, input_epsilon)
    next_state, output_reward, output_done, output_truncated = input_env.step(action, state[0:4], state[4])
    replay_buffer.append((state[0], action, output_reward, next_state[0], output_done, output_truncated))
    return next_state, output_reward, output_done, output_truncated, replay_buffer
"""

"""def training_step(input_batch_size, input_n_actions, input_replay_buffer, input_model):
    experiences = sample_experiences(input_batch_size, input_replay_buffer)
    states, actions, output_rewards, next_states, dones = experiences

    #new implementation
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)
    output_rewards = tf.convert_to_tensor(output_rewards, dtype=tf.float32)
    next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones, dtype=tf.float32)
    #end new implementation


    next_q_values = input_model.predict(next_states, verbose=0)
    max_next_q_values = next_q_values.max(axis=1)
    runs = 1.0 - dones  # simulation is not done
    target_q_values = output_rewards + runs * discount_factor * max_next_q_values
    #target_q_values = target_q_values.reshape(-1, 1)
    target_q_values = tf.reshape(target_q_values, (-1, 1))
    mask = tf.one_hot(actions, input_n_actions)

    with tf.GradientTape() as tape:
        all_q_values = input_model(states)
        q_values = tf.reduce_sum(all_q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_q_values, q_values))

    grads = tape.gradient(loss, input_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, input_model.trainable_variables))

    return input_model"""


############################## Double DQN ##############################

# Update the target model with the weights of the main model
def update_target_model(input_main_model, input_target_model):
    input_target_model.set_weights(input_main_model.get_weights())
    return input_target_model


"""def training_step(input_batch_size, input_n_actions, input_replay_buffer, input_model, input_target_model):
    experiences = sample_experiences(input_batch_size, input_replay_buffer)
    states, actions, rewards, next_states, dones = experiences

    # Convert experiences to tensors
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones, dtype=tf.float32)

    next_q_values = input_model.predict(next_states, verbose=0)
    next_actions = np.argmax(next_q_values, axis=1)

    target_q_values = input_target_model.predict(next_states, verbose=0)
    max_next_q_values = target_q_values[np.arange(input_batch_size), next_actions]

    runs = 1.0 - dones
    target_q_values = rewards + runs * discount_factor * max_next_q_values
    target_q_values = target_q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, input_n_actions)

    with tf.GradientTape() as tape:
        all_q_values = input_model(states)
        q_values = tf.reduce_sum(all_q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_q_values, q_values))

    grads = tape.gradient(loss, input_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, input_model.trainable_variables))"""


####################### 2 implementation DDQN ###################################

# Training step for Double DQN
def training_step(input_batch_size, input_n_actions, input_replay_buffer, input_model, input_target_model):
    # Sample experiences from the replay buffer
    experiences = sample_experiences(input_batch_size, input_replay_buffer)
    states, actions, output_rewards, next_states, dones = experiences

    # Convert experiences to tensors
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)
    output_rewards = tf.convert_to_tensor(output_rewards, dtype=tf.float32)
    next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones, dtype=tf.float32)

    # Use the main network to select actions (action selection)
    next_q_values_main = input_model.predict(next_states, verbose=0)
    best_actions = tf.argmax(next_q_values_main, axis=1)

    # Use the target network to evaluate the value of the selected actions (action evaluation)
    next_q_values_target = input_target_model.predict(next_states, verbose=0)
    max_next_q_values = tf.reduce_sum(next_q_values_target * tf.one_hot(best_actions, input_n_actions), axis=1)

    # Calculate the target Q-values using DDQN
    runs = 1.0 - dones  # If not done
    target_q_values = output_rewards + runs * discount_factor * max_next_q_values
    target_q_values = tf.reshape(target_q_values, (-1, 1))

    # Create a mask for the actions taken
    mask = tf.one_hot(actions, input_n_actions)

    with tf.GradientTape() as tape:
        # Predict Q-values for the current states using the main model
        all_q_values = input_model(states)
        q_values = tf.reduce_sum(all_q_values * mask, axis=1, keepdims=True)

        # Calculate the loss
        loss = tf.reduce_mean(loss_fn(target_q_values, q_values))

    # Compute the gradients and apply them to the main model
    grads = tape.gradient(loss, input_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, input_model.trainable_variables))

    return input_model


########################################################################################################################

# Rainbow DQN

########################################################################################################################

# Noisy Network for Rainbow DQN
class NoisyNetwork(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, stddev=0.017, activation=None):
        super(NoisyNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.stddev = stddev
        self.activation = tf.keras.activations.get(activation)

        self.mu_w = self.add_weight(
            shape=(input_dim, output_dim),
            initializer=tf.random_uniform_initializer(-stddev, stddev),
            trainable=True,
            name="mu_w"
        )
        self.mu_b = self.add_weight(
            shape=(output_dim,),
            initializer=tf.random_uniform_initializer(-stddev, stddev),
            trainable=True,
            name="mu_b"
        )

        self.sigma_w = self.add_weight(
            shape=(input_dim, output_dim),
            initializer=tf.constant_initializer(stddev),
            trainable=True,
            name="sigma_w"
        )
        self.sigma_b = self.add_weight(
            shape=(output_dim,),
            initializer=tf.constant_initializer(stddev),
            trainable=True,
            name="sigma_b"
        )

    def call(self, inputs, training=None):
        if training:
            epsilon_w = tf.random.normal(shape=(self.input_dim, self.output_dim))
            epsilon_b = tf.random.normal(shape=(self.output_dim,))
        else:
            epsilon_w = tf.zeros(shape=(self.input_dim, self.output_dim))
            epsilon_b = tf.zeros(shape=(self.output_dim,))

        weights = self.mu_w + self.sigma_w * epsilon_w
        biases = self.mu_b + self.sigma_b * epsilon_b

        output = tf.matmul(inputs, weights) + biases

        if self.activation is not None:
            output = self.activation(output)

        return output


# Function to create a noisy model for Rainbow DQN
def get_noisy_model(input_n_actions=2, input_n_nodes=4):
    inputs = tf.keras.Input(shape=(input_n_nodes,))

    x = NoisyNetwork(input_dim=input_n_nodes, output_dim=32, activation='elu')(inputs)
    x = NoisyNetwork(input_dim=32, output_dim=32, activation='elu')(x)
    outputs = NoisyNetwork(input_dim=32, output_dim=input_n_actions)(x)

    return tf.keras.Model(inputs, outputs)


# Prioritized Experience Replay
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, experience, priority):
        max_priority = max(self.priorities.max(), 1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        # Ensure priorities are clipped to avoid invalid values
        clipped_priorities = np.clip(self.priorities[:len(self.buffer)], a_min=1e-6, a_max=None)
        probs = clipped_priorities ** self.alpha
        probs /= np.sum(probs)  # Normalize probabilities

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Compute importance-sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights

        # Retrieve experiences
        experiences = [self.buffer[idx] for idx in indices]

        return experiences, indices, np.array(weights, dtype=np.float32)

    def update_last_done(self):
        """
        Update the `done` status of the last inserted experience to True.
        """
        if len(self.buffer) > 0:
            # Update the last inserted tuple
            last_index = (self.pos - 1) % self.capacity
            last_experience = self.buffer[last_index]

            # Unpack and modify the `done` status
            state, action, reward, next_state, done = last_experience
            self.buffer[last_index] = (state, action, reward, next_state, True)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


# Multi-Step Learning
def multi_step_return(rewards, gamma, n_steps):
    discounted_sum = 0.0
    for i in range(n_steps):
        discounted_sum += rewards[i] * (gamma ** i)
    return discounted_sum


# Distributional RL
class CategoricalDQN:
    def __init__(self, n_actions, n_atoms, v_min, v_max):
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.z = np.linspace(v_min, v_max, n_atoms)

    def get_distributional_model(self, input_n_nodes):
        inputs = layers.Input(shape=(input_n_nodes,))
        x = layers.Dense(32, activation="elu")(inputs)
        x = layers.Dense(32, activation="elu")(x)
        outputs = layers.Dense(self.n_actions * self.n_atoms, activation="softmax")(x)
        outputs = tf.reshape(outputs, [-1, self.n_actions, self.n_atoms])
        return tf.keras.Model(inputs, outputs)


# Rainbow DQN training step
def rainbow_training_step(input_batch_size, replay_buffer, model, target_model, beta, n_actions, n_atoms, v_min, v_max):
    delta_z = (v_max - v_min) / (n_atoms - 1)
    z = tf.linspace(v_min, v_max, n_atoms)  # Atom values

    # Sample experiences from the replay buffer
    experiences, indices, sampling_weights = replay_buffer.sample(input_batch_size, beta=beta)
    states, actions, rewards, next_states, dones = zip(*experiences)

    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones, dtype=tf.float32)

    # Predict next Q-value distributions
    next_q_dists = model.predict(next_states, verbose=0)  # Shape: [batch_size, n_actions, n_atoms]
    next_q_means = tf.reduce_sum(next_q_dists * z, axis=2)  # Compute mean Q-values
    next_actions = tf.argmax(next_q_means, axis=1)  # Select greedy actions based on mean Q-values

    # Predict target Q-value distributions
    target_q_dists = target_model.predict(next_states, verbose=0)  # Shape: [batch_size, n_actions, n_atoms]

    # Select the distribution corresponding to the next action
    batch_indices = tf.range(input_batch_size, dtype=tf.int32)[:, None]
    next_actions = tf.cast(next_actions, dtype=tf.int32)  # Cast next_actions to int32

    target_q_dists = tf.gather_nd(target_q_dists, tf.concat([batch_indices, next_actions[:, None]], axis=1))

    # Ensure z is float32
    z = tf.cast(z, dtype=tf.float32)

    # Compute projected target distribution
    z_projected = rewards[:, None] + (1.0 - dones[:, None]) * discount_factor * z
    z_projected = tf.clip_by_value(z_projected, v_min, v_max)  # Ensure within [v_min, v_max]
    b = (z_projected - v_min) / delta_z  # Map projected values to atom indices
    lower_indices = tf.floor(b)
    upper_indices = tf.math.ceil(b)

    lower_weights = upper_indices - b
    upper_weights = b - lower_indices

    # Clip indices to valid range
    lower_indices = tf.clip_by_value(lower_indices, 0, n_atoms - 1)
    upper_indices = tf.clip_by_value(upper_indices, 0, n_atoms - 1)

    # Initialize target distribution with zeros
    target_dist = tf.zeros([input_batch_size, n_atoms], dtype=tf.float32)
    # Compute weights and indices for projection
    lower_indices = tf.cast(lower_indices, tf.int32)
    upper_indices = tf.cast(upper_indices, tf.int32)

    # Create indices for scattering
    lower_indices_scatter = tf.concat([batch_indices, lower_indices], axis=1)  # [batch_size, 2]
    upper_indices_scatter = tf.concat([batch_indices, upper_indices], axis=1)  # [batch_size, 2]

    # Compute updates for lower indices
    batch_indices_expanded = tf.repeat(batch_indices, repeats=n_atoms, axis=0)  # [batch_size * n_atoms, 1]
    lower_updates = tf.reshape(lower_weights, [-1]) * tf.reshape(target_q_dists, [-1])  # Flatten weights and dists
    lower_indices_scatter = tf.concat([batch_indices_expanded, tf.reshape(lower_indices, [-1, 1])], axis=1)

    # Compute updates for upper indices
    upper_updates = tf.reshape(upper_weights, [-1]) * tf.reshape(target_q_dists, [-1])  # Flatten weights and dists
    upper_indices_scatter = tf.concat([batch_indices_expanded, tf.reshape(upper_indices, [-1, 1])], axis=1)

    # Scatter updates
    target_dist = tf.tensor_scatter_nd_add(target_dist, lower_indices_scatter, lower_updates)
    target_dist = tf.tensor_scatter_nd_add(target_dist, upper_indices_scatter, upper_updates)

    # Training step
    mask = tf.one_hot(actions, n_actions)[:, :, None]  # Mask for chosen actions
    with tf.GradientTape() as tape:
        pred_q_dists = model(states)  # [batch_size, n_actions, n_atoms]
        pred_q_dists = tf.reduce_sum(pred_q_dists * mask, axis=1)  # [batch_size, n_atoms]
        sample_losses = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)(target_dist,
                                                                                               pred_q_dists)
        loss = tf.reduce_mean(sampling_weights * sample_losses)

    # Apply gradients
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Update priorities in the replay buffer
    replay_buffer.update_priorities(indices, sample_losses.numpy())

    return model




