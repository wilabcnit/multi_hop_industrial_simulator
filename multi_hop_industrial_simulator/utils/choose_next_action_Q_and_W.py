from copy import deepcopy
import numpy as np

from multi_hop_industrial_simulator.network.ue import Ue
from multi_hop_industrial_simulator.utils.utils_for_tb_ualoha_with_dqn import compute_normalized_linear_interpolation
from multi_hop_industrial_simulator.dqn_agent.dqn_agent_rl_mesh import epsilon_greedy_policy
from multi_hop_industrial_simulator.utils.read_inputs import read_inputs

inputs = read_inputs('inputs.yaml')

number_of_tx_data_per_step = inputs.get('rl').get('agent').get('number_of_tx_data_per_step')

# Function to choose the next action for the UE with a single DNN using both Q and W values
def choose_next_action_tb_with_RL_Q_and_W(input_ue: Ue, input_enable_print: bool = False, input_n_simulation: int = 0, input_n_simulations: int = 0,
                                          input_n_actions: int = 0, input_W_min: int = None, input_W_max: int = None,
                                          input_Q_min: int = None, input_Q_max: int = None, ):
    """

    Args:
      input_ue: Ue: 
      input_enable_print: bool:  (Default value = False)
      input_n_simulation: int:  (Default value = 0)
      input_n_simulations: int:  (Default value = 0)
      input_n_actions: int:  (Default value = 0)
      input_W_min: int:  (Default value = None)
      input_W_max: int:  (Default value = None)
      input_Q_min: int:  (Default value = None)
      input_Q_max: int:  (Default value = None)

    Returns:

    """

    if input_enable_print:
        print("UE ", input_ue.get_ue_id(), " previous state: ", input_ue.Q_and_W_previous_state)
    # Copy the current state to the previous state using deepcopy
    input_ue.Q_and_W_previous_state = deepcopy(input_ue.Q_and_W_current_state)

    # build the input state for the DRL
    input_ue.Q_and_W_current_state = [round(input_ue.Q_and_W_acks_rx_per_step_counter / input_ue.Q_and_W_pcks_tx_per_step_counter, 2),
             compute_normalized_linear_interpolation(x=input_ue.Q_and_W_buffer_length, x_min=input_Q_min,
                                                     x_max=input_Q_max),
             compute_normalized_linear_interpolation(x=input_ue.Q_and_W_contention_window, x_min=input_W_min,
                                                     x_max=input_W_max),
             round(input_ue.Q_and_W_acks_rx_per_step_counter / (input_ue.Q_and_W_acks_rx_per_step_counter + input_ue.Q_and_W_dropped_pcks_per_step_counter + 1e-6), 2),
             round(input_ue.Q_and_W_dropped_pcks_per_step_counter / input_ue.Q_and_W_pcks_tx_per_step_counter, 2),
             round(np.mean(input_ue.Q_and_W_latencies) / input_ue.Q_and_W_max_latency, 2) if len(input_ue.Q_and_W_latencies) > 0 else 1,
             round(input_ue.Q_and_W_pcks_tx_per_step_counter / (3 * (input_ue.Q_and_W_buffer_length + 1)), 2)
             ]

    # Check if the new action has to be performed
    if 0 <= input_ue.Q_and_W_last_action < input_n_actions and input_ue.Q_and_W_tx_data_counter != number_of_tx_data_per_step:
        # build the input state for the DRL
        # Q_and_W_previous_state = [input_ue.Q_and_W_previous_state[0],
        #                   compute_normalized_linear_interpolation(x=input_ue.Q_and_W_previous_state[1], x_min=input_Q_min,
        #                                                           x_max=input_Q_max),
        #                   compute_normalized_linear_interpolation(x=input_ue.Q_and_W_previous_state[2], x_min=input_W_min,
        #                                                           x_max=input_W_max)
        #                   ]

        # compute the reward
        reward = input_ue.reward_computation_for_Q_and_W()
        # Populate the replay buffer with the previous state, last action, reward, current state and done flag
        input_ue.append_replay_buffer(
                        input_replay_buffer_instance=[input_ue.Q_and_W_previous_state, input_ue.Q_and_W_last_action, reward, input_ue.Q_and_W_current_state,
                                                      False])
        # Save the reward
        input_ue.append_reward(reward)

        if input_enable_print:
            print("UE ", input_ue.ue_id, " Reward: ", reward)

    # Check if the simulation is for training or testing
    if input_n_simulation < input_n_simulations - 20:
        epsilon_function = 1 - (input_n_simulation / (input_n_simulations - 20))
    else:
        epsilon_function = 0
    input_ue.set_epsilon(
        input_epsilon=max(epsilon_function, 0.01))

    if input_enable_print:
        print("UE ", input_ue.get_ue_id(), " current state: ", input_ue.Q_and_W_current_state)

    # Set the action using the epsilon-greedy policy
    input_ue.Q_and_W_last_action = epsilon_greedy_policy(state=np.array(input_ue.Q_and_W_current_state),
                                           input_n_actions=input_n_actions,
                                           model=input_ue.get_model(),
                                           input_epsilon=input_ue.get_epsilon())

    # append the action to the list of actions
    input_ue.append_action_list(input_ue.Q_and_W_last_action)

    if input_enable_print:
        print("UE ", input_ue.get_ue_id(), " has chosen action ", input_ue.Q_and_W_last_action, " with epsilon ", input_ue.get_epsilon())

    # 0: Q++; W--
    # 1: Q--; W++
    # 2: Q==; W==
    # 3: Q++; W==
    # 4: Q==; W++
    # 5: Q--; W==
    # 6: Q==; W--

    # Check if the action is forbidden
    if (input_ue.Q_and_W_last_action == 0 and (input_ue.Q_and_W_buffer_length == input_Q_max or input_ue.Q_and_W_contention_window == input_W_min)) or \
        (input_ue.Q_and_W_last_action == 1 and (input_ue.Q_and_W_buffer_length == input_Q_min or input_ue.Q_and_W_contention_window == input_W_max)) or \
        (input_ue.Q_and_W_last_action == 3 and input_ue.Q_and_W_buffer_length == input_Q_max) or \
        (input_ue.Q_and_W_last_action == 4 and input_ue.Q_and_W_contention_window == input_W_max) or \
        (input_ue.Q_and_W_last_action == 5 and input_ue.Q_and_W_buffer_length == input_Q_min) or \
        (input_ue.Q_and_W_last_action == 6 and input_ue.Q_and_W_contention_window == input_W_min):
        input_ue.Q_and_W_forbidden_action = True

    else:
        if (input_ue.Q_and_W_last_action == 0 or input_ue.Q_and_W_last_action == 3) and input_ue.Q_and_W_buffer_length < input_Q_max : # Q++
            input_ue.Q_and_W_buffer_length += 1

        elif (input_ue.Q_and_W_last_action == 1 or input_ue.Q_and_W_last_action == 5) and input_ue.Q_and_W_buffer_length > input_Q_min : # Q++
            input_ue.Q_and_W_buffer_length -= 1

        if (input_ue.Q_and_W_last_action == 1 or input_ue.Q_and_W_last_action == 4) and input_ue.Q_and_W_contention_window < input_W_max:  # Q++
            input_ue.Q_and_W_contention_window += 1

        elif (input_ue.Q_and_W_last_action == 0 or input_ue.Q_and_W_last_action == 6) and input_ue.Q_and_W_contention_window > input_W_min:  # Q++
            input_ue.Q_and_W_contention_window -= 1

    #input_ue.Q_and_W_current_state[0] = 0

    # Reset the latencies list
    input_ue.Q_and_W_latencies = list()

    # reset the counters for the next step
    input_ue.Q_and_W_acks_rx_per_step_counter = 0
    input_ue.Q_and_W_pcks_tx_per_step_counter = 0
    input_ue.Q_and_W_dropped_pcks_per_step_counter = 0

    return

# Function to choose the next action for the UE with a single DNN using only W values
def choose_next_action_tb_only_W(input_ue: Ue, input_enable_print: bool = False, input_n_simulation: int = 0, input_n_simulations: int = 0,
                                          input_n_actions: int = 0, input_W_min: int = None, input_W_max: int = None, input_goal_oriented:str = None,
                                          input_normalized_S:bool = False, input_current_tick: int = 0):
    """

    Args:
      input_ue: Ue: 
      input_enable_print: bool:  (Default value = False)
      input_n_simulation: int:  (Default value = 0)
      input_n_simulations: int:  (Default value = 0)
      input_n_actions: int:  (Default value = 0)
      input_W_min: int:  (Default value = None)
      input_W_max: int:  (Default value = None)
      input_goal_oriented:str:  (Default value = None)
      input_normalized_S:bool:  (Default value = False)
      input_current_tick: int:  (Default value = 0)

    Returns:

    """


    # Copy the current state to the previous state using deepcopy
    input_ue.W_previous_state = deepcopy(input_ue.W_current_state)

    if input_enable_print:
        print("UE ", input_ue.get_ue_id(), " previous state: ", input_ue.W_previous_state)

    # substitute the list of rtx_pcks_tx_with_success which are empty with [4]
    if len(input_ue.W_rtx_pcks_tx_with_success[-1]) == 0:
        input_ue.W_rtx_pcks_tx_with_success[-1] = [4]

    # Check the purpose of the RL agent
    # Throughput and Latency
    if input_goal_oriented == "S&L":
        if input_normalized_S:
            # Compute the normalized throughput
            input_ue.W_pcks_tx_with_success[-1] /= (input_current_tick - input_ue.W_start_tick_params_window) if input_current_tick - input_ue.W_start_tick_params_window > 0 else 1
        # # build the input state for the DRL, Only W RL
        input_ue.W_current_state = [
            round(min(
                input_ue.W_pcks_tx_with_success[-1] / np.mean(input_ue.W_pcks_tx_with_success)
                if np.mean(input_ue.W_pcks_tx_with_success) != 0 else 0, 1.0), 2),

            round(min(
                np.mean(input_ue.W_rtx_pcks_tx_with_success[-1]) / np.mean(
                    [np.mean(lst) for lst in input_ue.W_rtx_pcks_tx_with_success])
                if np.mean([np.mean(lst) for lst in input_ue.W_rtx_pcks_tx_with_success]) != 0 else 0, 1.0), 2),

            compute_normalized_linear_interpolation(x=input_ue.Q_and_W_contention_window, x_min=input_W_min,
                                                    x_max=input_W_max)
        ]

    # Only Throughput
    elif input_goal_oriented == "S":
        if input_normalized_S:
            # Compute the normalized throughput
            input_ue.W_pcks_tx_with_success[-1] /= (input_current_tick - input_ue.W_start_tick_params_window) if input_current_tick - input_ue.W_start_tick_params_window > 0 else 1
        # # build the input state for the DRL, Only W RL
        input_ue.W_current_state = [
            round(min(
                input_ue.W_pcks_tx_with_success[-1] / np.mean(input_ue.W_pcks_tx_with_success)
                if np.mean(input_ue.W_pcks_tx_with_success) != 0 else 0, 1.0), 2),

            compute_normalized_linear_interpolation(x=input_ue.Q_and_W_contention_window, x_min=input_W_min,
                                                    x_max=input_W_max)
        ]

    # Only Latency
    elif input_goal_oriented == "L":
        # # build the input state for the DRL, Only W RL
        input_ue.W_current_state = [
            round(min(
                np.mean(input_ue.W_rtx_pcks_tx_with_success[-1]) / np.mean(
                    [np.mean(lst) for lst in input_ue.W_rtx_pcks_tx_with_success])
                if np.mean([np.mean(lst) for lst in input_ue.W_rtx_pcks_tx_with_success]) != 0 else 0, 1.0), 2),

            compute_normalized_linear_interpolation(x=input_ue.Q_and_W_contention_window, x_min=input_W_min,
                                                    x_max=input_W_max)
        ]

    if input_enable_print:
        print("UE ", input_ue.get_ue_id(), " current state: ", input_ue.W_current_state)

    # Check if the new action has to be performed
    if 0 <= input_ue.W_last_action < input_n_actions and input_ue.Q_and_W_tx_data_counter != number_of_tx_data_per_step:

        # compute the reward
        reward = input_ue.reward_computation_for_only_W(input_goal_oriented=input_goal_oriented)
        # Populate the replay buffer with the previous state, last action, reward, current state and done flag
        input_ue.append_W_replay_buffer(
                        input_replay_buffer_instance=[input_ue.W_previous_state, input_ue.W_last_action, reward, input_ue.W_current_state,
                                                      False])
        # Save the reward
        input_ue.append_W_reward(reward)

        if input_enable_print:
            print("UE ", input_ue.ue_id, " Reward: ", reward)

    # Check if the simulation is for training or testing
    if input_n_simulation < input_n_simulations - 20:
        epsilon_function = 1 - (input_n_simulation / (input_n_simulations - 20))
    else:
        epsilon_function = 0
    input_ue.set_epsilon(
        input_epsilon=max(epsilon_function, 0.01))

    # Set the action using the epsilon-greedy policy
    input_ue.W_last_action = epsilon_greedy_policy(state=np.array(input_ue.W_current_state),
                                           input_n_actions=input_n_actions,
                                           model=input_ue.get_W_model(),
                                           input_epsilon=input_ue.get_epsilon())

    # append the action to the list of actions
    input_ue.append_W_action_list(input_ue.W_last_action)

    if input_enable_print:
        print("UE ", input_ue.get_ue_id(), " has chosen action ", input_ue.W_last_action, " with epsilon ", input_ue.get_epsilon())

    # 0: W--
    # 1: W++
    # 2: W==

    # Check if the action is forbidden
    if (input_ue.W_last_action == 0 and  input_ue.Q_and_W_contention_window == input_W_min) or \
        (input_ue.W_last_action == 1 and input_ue.Q_and_W_contention_window == input_W_max):
        input_ue.W_forbidden_action = True

    else:
        if (input_ue.W_last_action == 1) and input_ue.Q_and_W_contention_window < input_W_max:  # W--
            input_ue.Q_and_W_contention_window += 1

        elif (input_ue.W_last_action == 0) and input_ue.Q_and_W_contention_window > input_W_min:  # W++
            input_ue.Q_and_W_contention_window -= 1


    # Reset the latencies list
    input_ue.Q_and_W_latencies = list()

    # reset the counters for the next step
    input_ue.Q_and_W_acks_rx_per_step_counter = 0
    input_ue.Q_and_W_pcks_tx_per_step_counter = 0
    input_ue.W_dropped_pcks_per_step_counter = 0
    input_ue.W_not_added_pcks_per_step_counter = 0

    # Only W RL
    # Reset the counters for the next step
    input_ue.W_start_tick_params_window = input_current_tick + 1
    input_ue.W_pcks_tx_with_success.append(0)
    input_ue.W_rtx_pcks_tx_with_success.append([])
    input_ue.Q_and_W_saved_state_W[-1].append(input_ue.Q_and_W_contention_window)

    return

# Function to choose the next action for the UE with a single DNN using only Q values
def choose_next_action_tb_only_Q(input_ue: Ue, input_enable_print: bool = False, input_n_simulation: int = 0, input_n_simulations: int = 0,
                                          input_n_actions: int = 0, input_Q_min: int = None, input_Q_max: int = None, input_goal_oriented:str = None,
                                          input_normalized_S:bool = False, input_current_tick: int = 0):
    """

    Args:
      input_ue: Ue: 
      input_enable_print: bool:  (Default value = False)
      input_n_simulation: int:  (Default value = 0)
      input_n_simulations: int:  (Default value = 0)
      input_n_actions: int:  (Default value = 0)
      input_Q_min: int:  (Default value = None)
      input_Q_max: int:  (Default value = None)
      input_goal_oriented:str:  (Default value = None)
      input_normalized_S:bool:  (Default value = False)
      input_current_tick: int:  (Default value = 0)

    Returns:

    """


    # Copy the current state to the previous state using deepcopy
    input_ue.Q_previous_state = deepcopy(input_ue.Q_current_state)

    if input_enable_print:
        print("UE ", input_ue.get_ue_id(), " previous state: ", input_ue.Q_previous_state)

    # substitute the list of rtx_pcks_tx_with_success which are empty with [4]
    if len(input_ue.Q_rtx_pcks_tx_with_success[-1]) == 0:
        input_ue.Q_rtx_pcks_tx_with_success[-1] = [4]

    # Check the purpose of the RL agent
    # Throughput and Latency
    if input_goal_oriented == "S&L":
        if input_normalized_S:
            # Compute the normalized throughput
            input_ue.Q_pcks_tx_with_success[-1] /= (input_current_tick - input_ue.Q_start_tick_params_window) if input_current_tick - input_ue.Q_start_tick_params_window > 0 else 1
        # # build the input state for the DRL, Only W RL
        input_ue.Q_current_state = [
            round(min(
                input_ue.Q_pcks_tx_with_success[-1] / np.mean(input_ue.Q_pcks_tx_with_success)
                if np.mean(input_ue.Q_pcks_tx_with_success) != 0 else 0, 1.0), 2),

            round(min(
                np.mean(input_ue.Q_rtx_pcks_tx_with_success[-1]) / np.mean(
                    [np.mean(lst) for lst in input_ue.Q_rtx_pcks_tx_with_success])
                if np.mean([np.mean(lst) for lst in input_ue.Q_rtx_pcks_tx_with_success]) != 0 else 0, 1.0), 2),

            np.mean(input_ue.Q_buffer_utilization) / (input_ue.Q_and_W_buffer_length + 1),

            compute_normalized_linear_interpolation(x=input_ue.Q_and_W_buffer_length, x_min=input_Q_min, x_max=input_Q_max)
        ]

    # Only Throughput
    elif input_goal_oriented == "S":
        if input_normalized_S:
            # Compute the normalized throughput
            input_ue.Q_pcks_tx_with_success[-1] /= (input_current_tick - input_ue.Q_start_tick_params_window) if input_current_tick - input_ue.Q_start_tick_params_window > 0 else 1
        # # build the input state for the DRL, Only W RL
        input_ue.Q_current_state = [
            round(min(
                input_ue.Q_pcks_tx_with_success[-1] / np.mean(input_ue.Q_pcks_tx_with_success)
                if np.mean(input_ue.Q_pcks_tx_with_success) != 0 else 0, 1.0), 2),

            np.mean(input_ue.Q_buffer_utilization) / (input_ue.Q_and_W_buffer_length + 1),

            compute_normalized_linear_interpolation(x=input_ue.Q_and_W_buffer_length, x_min=input_Q_min, x_max=input_Q_max)
        ]

    # Only Latency
    elif input_goal_oriented == "L":
        # # build the input state for the DRL, Only W RL
        input_ue.Q_current_state = [
            round(min(
                np.mean(input_ue.Q_rtx_pcks_tx_with_success[-1]) / np.mean(
                    [np.mean(lst) for lst in input_ue.Q_rtx_pcks_tx_with_success])
                if np.mean([np.mean(lst) for lst in input_ue.Q_rtx_pcks_tx_with_success]) != 0 else 0, 1.0), 2),

            np.mean(input_ue.Q_buffer_utilization) / (input_ue.Q_and_W_buffer_length + 1),

            compute_normalized_linear_interpolation(x=input_ue.Q_and_W_buffer_length, x_min=input_Q_min, x_max=input_Q_max)
        ]

    if input_enable_print:
        print("UE ", input_ue.get_ue_id(), " current state: ", input_ue.Q_current_state)

    # Check if the new action has to be performed
    if 0 <= input_ue.Q_last_action < input_n_actions and input_ue.Q_and_W_tx_data_counter != number_of_tx_data_per_step:

        # compute the reward
        reward = input_ue.reward_computation_for_only_Q(input_goal_oriented=input_goal_oriented)
        # Populate the replay buffer with the previous state, last action, reward, current state and done flag
        input_ue.append_Q_replay_buffer(
                        input_replay_buffer_instance=[input_ue.Q_previous_state, input_ue.Q_last_action, reward, input_ue.Q_current_state,
                                                      False])
        # Save the reward
        input_ue.append_Q_reward(reward)

        if input_enable_print:
            print("UE ", input_ue.ue_id, " Reward: ", reward)

    # Check if the simulation is for training or testing
    if input_n_simulation < input_n_simulations - 20:
        epsilon_function = 1 - (input_n_simulation / (input_n_simulations - 20))
    else:
        epsilon_function = 0

    # Set the epsilon value
    input_ue.set_epsilon(
        input_epsilon=max(epsilon_function, 0.01))

    # Set the action using the epsilon-greedy policy
    input_ue.Q_last_action = epsilon_greedy_policy(state=np.array(input_ue.Q_current_state),
                                           input_n_actions=input_n_actions,
                                           model=input_ue.get_Q_model(),
                                           input_epsilon=input_ue.get_epsilon())

    # append the action to the list of actions
    input_ue.append_Q_action_list(input_ue.Q_last_action)

    if input_enable_print:
        print("UE ", input_ue.get_ue_id(), " has chosen action ", input_ue.Q_last_action, " with epsilon ", input_ue.get_epsilon())

    # 0: Q--
    # 1: Q++
    # 2: Q==

    # Check if the action is forbidden
    if (input_ue.Q_last_action == 0 and  input_ue.Q_and_W_buffer_length == input_Q_min) or \
        (input_ue.Q_last_action == 1 and input_ue.Q_and_W_buffer_length == input_Q_max):
        input_ue.Q_forbidden_action = True

    else:
        if (input_ue.Q_last_action == 1) and input_ue.Q_and_W_buffer_length < input_Q_max:  # Q++
            input_ue.Q_and_W_buffer_length += 1

        elif (input_ue.Q_last_action == 0) and input_ue.Q_and_W_buffer_length > input_Q_min:  # Q--
            input_ue.Q_and_W_buffer_length -= 1

    # Reset the latencies list
    input_ue.Q_and_W_latencies = list()

    # reset the counters for the next step
    input_ue.Q_and_W_acks_rx_per_step_counter = 0
    input_ue.Q_and_W_pcks_tx_per_step_counter = 0
    input_ue.Q_dropped_pcks_per_step_counter = 0
    input_ue.Q_not_added_pcks_per_step_counter = 0

    # Only W RL
    # Reset the counters for the next step
    input_ue.Q_start_tick_params_window = input_current_tick + 1
    input_ue.Q_pcks_tx_with_success.append(0)
    input_ue.Q_rtx_pcks_tx_with_success.append([])
    input_ue.Q_buffer_utilization = list()
    input_ue.Q_and_W_saved_state_Q[-1].append(input_ue.Q_and_W_buffer_length)


    return