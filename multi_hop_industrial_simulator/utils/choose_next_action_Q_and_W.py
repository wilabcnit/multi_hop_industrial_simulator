from copy import deepcopy
import numpy as np

from multi_hop_industrial_simulator.network.ue import Ue
from multi_hop_industrial_simulator.utils.utils_for_tb_ualoha_with_dqn import compute_normalized_linear_interpolation
from multi_hop_industrial_simulator.dqn_agent.dqn_agent_rl_mesh import epsilon_greedy_policy
from multi_hop_industrial_simulator.utils.read_inputs import read_inputs

inputs = read_inputs('inputs.yaml')

number_of_tx_data_per_step = inputs.get('rl').get('agent').get('number_of_tx_data_per_step')

def choose_next_action_tb_with_RL_Q_and_W(input_ue: Ue, input_enable_print: bool = False, input_n_simulation: int = 0, input_n_simulations: int = 0,
                                          input_n_actions: int = 0, input_W_min: int = None, input_W_max: int = None,
                                          input_Q_min: int = None, input_Q_max: int = None):
    """
    Choose the next action for a UE using a single DNN that considers both Q and W values.
    This function updates the UE's Q and W state, computes and stores a reward (if applicable),
    applies an epsilon-greedy policy to select the next action, enforces action constraints, and
    updates the UE's Q and W configuration accordingly.

    Args:
        input_ue (Ue): The UE instance whose next action is being selected and whose internal
                       state will be updated (replay buffer, rewards, action list, etc.).
        input_enable_print (bool): If True, prints enabled. (Default value = False)
        input_n_simulation (int): Index of the current simulation step or episode. Used to
                                  compute epsilon for exploration. (Default value = 0)
        input_n_simulations (int): Total number of simulation steps or episodes. Used to
                                   compute epsilon for exploration. (Default value = 0)
        input_n_actions (int): Number of possible discrete actions for the DNN/policy. Used
                               for bounds checking on last action. (Default value = 0)
        input_W_min (int | None): Minimum allowed value for the contention window (W). If None,
                                  normalization functions should handle it appropriately.
                                  (Default value = None)
        input_W_max (int | None): Maximum allowed value for the contention window (W). If None,
                                  normalization functions should handle it appropriately.
                                  (Default value = None)
        input_Q_min (int | None): Minimum allowed value for the buffer length (Q). If None,
                                  normalization functions should handle it appropriately.
                                  (Default value = None)
        input_Q_max (int | None): Maximum allowed value for the buffer length (Q). If None,
                                  normalization functions should handle it appropriately.
                                  (Default value = None)

    Returns:
        None

    Side effects:
        - Updates input_ue.Q_and_W_previous_state and input_ue.Q_and_W_current_state.
        - May append a transition to the UE's replay buffer.
        - Appends selected action to the UE action list and updates input_ue.Q_and_W_last_action.
        - Updates input_ue.Q_and_W_buffer_length and input_ue.Q_and_W_contention_window
          depending on the chosen action (unless action is forbidden).
        - Resets latency and per-step counters on the UE for the next step.
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

        elif (input_ue.Q_and_W_last_action == 1 or input_ue.Q_and_W_last_action == 5) and input_ue.Q_and_W_buffer_length > input_Q_min : # Q--
            input_ue.Q_and_W_buffer_length -= 1

        if (input_ue.Q_and_W_last_action == 1 or input_ue.Q_and_W_last_action == 4) and input_ue.Q_and_W_contention_window < input_W_max:  # W++
            input_ue.Q_and_W_contention_window += 1

        elif (input_ue.Q_and_W_last_action == 0 or input_ue.Q_and_W_last_action == 6) and input_ue.Q_and_W_contention_window > input_W_min:  # W--
            input_ue.Q_and_W_contention_window -= 1

    # Reset the latencies list
    input_ue.Q_and_W_latencies = list()

    # reset the counters for the next step
    input_ue.Q_and_W_acks_rx_per_step_counter = 0
    input_ue.Q_and_W_pcks_tx_per_step_counter = 0
    input_ue.Q_and_W_dropped_pcks_per_step_counter = 0

    return


def choose_next_action_tb_only_W(input_ue: Ue, input_enable_print: bool = False, input_n_simulation: int = 0,
                                 input_n_simulations: int = 0, input_n_actions: int = 0, input_W_min: int = None,
                                 input_W_max: int = None, input_goal_oriented: str = None,
                                 input_normalized_S: bool = False, input_current_tick: int = 0):
    """
    Chooses the next action for a UE using a single DNN that operates only on the contention window variable W.
    The function updates the current RL state, computes and stores rewards when applicable, selects the next action
    using an epsilon-greedy policy, enforces action constraints, and updates W accordingly.

    This function is typically used when the reinforcement learning agent's objective focuses only on
    throughput (S), latency (L), or a combination of both (S&L).

    Args:
        input_ue (Ue): The UE instance whose state and policy are being updated.
                       Must provide methods for reward computation, replay buffer
                       management, and epsilon scheduling.
        input_enable_print (bool): If True, prints enabled.
                                   (Default = False)
        input_n_simulation (int): The current simulation iteration (used to compute epsilon).
                                  (Default = 0)
        input_n_simulations (int): The total number of simulation iterations.
                                   (Default = 0)
        input_n_actions (int): The total number of possible actions the agent can take.
                               (Default = 0)
        input_W_min (int | None): Minimum contention window size allowed.
                                  (Default = None)
        input_W_max (int | None): Maximum contention window size allowed.
                                  (Default = None)
        input_goal_oriented (str | None): Defines the optimization goal.
                                          Accepted values:
                                            - "S&L" → Optimize throughput and latency
                                            - "S"   → Optimize throughput only
                                            - "L"   → Optimize latency only
                                          (Default = None)
        input_normalized_S (bool): If True, normalizes throughput over elapsed ticks
                                   to stabilize reward scaling. (Default = False)
        input_current_tick (int): The current global simulation tick, used for time-based
                                  normalization and parameter window resets. (Default = 0)

    Returns:
        None

    Side effects:
        - Updates `input_ue.W_current_state` and `input_ue.W_previous_state`
        - Computes reward and appends it to replay buffer (if applicable)
        - Selects next action using epsilon-greedy policy
        - Modifies `input_ue.Q_and_W_contention_window` according to chosen action
        - Resets several tracking counters and latency lists for the next iteration
        - Updates internal W-related historical structures (`W_pcks_tx_with_success`,
          `W_rtx_pcks_tx_with_success`, and `Q_and_W_saved_state_W`)
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
            input_ue.W_pcks_tx_with_success[-1] /= (
                (input_current_tick - input_ue.W_start_tick_params_window)
                if input_current_tick - input_ue.W_start_tick_params_window > 0
                else 1
            )
        # build the input state for the DRL, Only W RL
        input_ue.W_current_state = [
            round(
                min(
                    input_ue.W_pcks_tx_with_success[-1]
                    / np.mean(input_ue.W_pcks_tx_with_success)
                    if np.mean(input_ue.W_pcks_tx_with_success) != 0
                    else 0,
                    1.0,
                ),
                2,
            ),
            round(
                min(
                    np.mean(input_ue.W_rtx_pcks_tx_with_success[-1])
                    / np.mean(
                        [np.mean(lst) for lst in input_ue.W_rtx_pcks_tx_with_success]
                    )
                    if np.mean(
                        [np.mean(lst) for lst in input_ue.W_rtx_pcks_tx_with_success]
                    )
                    != 0
                    else 0,
                    1.0,
                ),
                2,
            ),
            compute_normalized_linear_interpolation(
                x=input_ue.Q_and_W_contention_window,
                x_min=input_W_min,
                x_max=input_W_max,
            ),
        ]

    # Only Throughput
    elif input_goal_oriented == "S":
        if input_normalized_S:
            # Compute the normalized throughput
            input_ue.W_pcks_tx_with_success[-1] /= (
                (input_current_tick - input_ue.W_start_tick_params_window)
                if input_current_tick - input_ue.W_start_tick_params_window > 0
                else 1
            )
        # build the input state for the DRL, Only W RL
        input_ue.W_current_state = [
            round(
                min(
                    input_ue.W_pcks_tx_with_success[-1]
                    / np.mean(input_ue.W_pcks_tx_with_success)
                    if np.mean(input_ue.W_pcks_tx_with_success) != 0
                    else 0,
                    1.0,
                ),
                2,
            ),
            compute_normalized_linear_interpolation(
                x=input_ue.Q_and_W_contention_window,
                x_min=input_W_min,
                x_max=input_W_max,
            ),
        ]

    # Only Latency
    elif input_goal_oriented == "L":
        # build the input state for the DRL, Only W RL
        input_ue.W_current_state = [
            round(
                min(
                    np.mean(input_ue.W_rtx_pcks_tx_with_success[-1])
                    / np.mean(
                        [np.mean(lst) for lst in input_ue.W_rtx_pcks_tx_with_success]
                    )
                    if np.mean(
                        [np.mean(lst) for lst in input_ue.W_rtx_pcks_tx_with_success]
                    )
                    != 0
                    else 0,
                    1.0,
                ),
                2,
            ),
            compute_normalized_linear_interpolation(
                x=input_ue.Q_and_W_contention_window,
                x_min=input_W_min,
                x_max=input_W_max,
            ),
        ]

    if input_enable_print:
        print("UE ", input_ue.get_ue_id(), " current state: ", input_ue.W_current_state)

    # Check if the new action has to be performed
    if (
        0 <= input_ue.W_last_action < input_n_actions
        and input_ue.Q_and_W_tx_data_counter != number_of_tx_data_per_step
    ):
        # compute the reward
        reward = input_ue.reward_computation_for_only_W(
            input_goal_oriented=input_goal_oriented
        )
        # Populate the replay buffer with the previous state, last action, reward, current state and done flag
        input_ue.append_W_replay_buffer(
            input_replay_buffer_instance=[
                input_ue.W_previous_state,
                input_ue.W_last_action,
                reward,
                input_ue.W_current_state,
                False,
            ]
        )
        # Save the reward
        input_ue.append_W_reward(reward)

        if input_enable_print:
            print("UE ", input_ue.ue_id, " Reward: ", reward)

    # Compute epsilon for epsilon-greedy
    if input_n_simulation < input_n_simulations - 20:
        epsilon_function = 1 - (input_n_simulation / (input_n_simulations - 20))
    else:
        epsilon_function = 0
    input_ue.set_epsilon(input_epsilon=max(epsilon_function, 0.01))

    # Choose next action
    input_ue.W_last_action = epsilon_greedy_policy(
        state=np.array(input_ue.W_current_state),
        input_n_actions=input_n_actions,
        model=input_ue.get_W_model(),
        input_epsilon=input_ue.get_epsilon(),
    )

    input_ue.append_W_action_list(input_ue.W_last_action)

    if input_enable_print:
        print(
            "UE ",
            input_ue.get_ue_id(),
            " has chosen action ",
            input_ue.W_last_action,
            " with epsilon ",
            input_ue.get_epsilon(),
        )

    # 0: W--
    # 1: W++
    # 2: W==

    # Check if the action is forbidden
    if (
        (input_ue.W_last_action == 0 and input_ue.Q_and_W_contention_window == input_W_min)
        or (input_ue.W_last_action == 1 and input_ue.Q_and_W_contention_window == input_W_max)
    ):
        input_ue.W_forbidden_action = True
    else:
        if (
            input_ue.W_last_action == 1
            and input_ue.Q_and_W_contention_window < input_W_max
        ):
            input_ue.Q_and_W_contention_window += 1
        elif (
            input_ue.W_last_action == 0
            and input_ue.Q_and_W_contention_window > input_W_min
        ):
            input_ue.Q_and_W_contention_window -= 1

    # Reset counters and latency tracking
    input_ue.Q_and_W_latencies = list()
    input_ue.Q_and_W_acks_rx_per_step_counter = 0
    input_ue.Q_and_W_pcks_tx_per_step_counter = 0
    input_ue.W_dropped_pcks_per_step_counter = 0
    input_ue.W_not_added_pcks_per_step_counter = 0

    # Reset W RL tracking for next iteration
    input_ue.W_start_tick_params_window = input_current_tick + 1
    input_ue.W_pcks_tx_with_success.append(0)
    input_ue.W_rtx_pcks_tx_with_success.append([])
    input_ue.Q_and_W_saved_state_W[-1].append(input_ue.Q_and_W_contention_window)

    return

def choose_next_action_tb_only_Q(input_ue: Ue, input_enable_print: bool = False, input_n_simulation: int = 0,
                                 input_n_simulations: int = 0, input_n_actions: int = 0, input_Q_min: int = None,
                                 input_Q_max: int = None, input_goal_oriented: str = None,
                                 input_normalized_S: bool = False, input_current_tick: int = 0):
    """
    Chooses the next action for a UE using a single DNN that operates only on the buffer length variable Q.
    This function updates the UE's RL state, computes and stores rewards when applicable, selects
    the next action using an epsilon-greedy policy, applies constraints to forbidden actions,
    and updates the buffer size accordingly.

    This function is designed for scenarios where the RL agent’s optimization goal focuses
    on throughput (S), latency (L), or both (S&L), using Q as the controllable parameter.

    Args:
        input_ue (Ue): The UE instance whose RL state and model are being updated.
                       Must implement reward computation, replay buffer handling,
                       and epsilon-greedy utilities.
        input_enable_print (bool): If True, prints enabled.
                                   (Default = False)
        input_n_simulation (int): The current simulation iteration (used for epsilon decay).
                                  (Default = 0)
        input_n_simulations (int): Total number of simulation iterations (used for epsilon scheduling).
                                   (Default = 0)
        input_n_actions (int): Total number of actions available to the RL agent.
                               (Default = 0)
        input_Q_min (int | None): Minimum allowable queue/buffer length value.
                                  (Default = None)
        input_Q_max (int | None): Maximum allowable queue/buffer length value.
                                  (Default = None)
        input_goal_oriented (str | None): Defines the learning objective of the RL agent.
                                          Accepted values:
                                            - "S&L": Optimize throughput and latency
                                            - "S"  : Optimize throughput only
                                            - "L"  : Optimize latency only
                                          (Default = None)
        input_normalized_S (bool): If True, normalizes throughput metrics relative to elapsed ticks
                                   to stabilize the input state. (Default = False)
        input_current_tick (int): The current simulation tick used for normalization and window updates.
                                  (Default = 0)

    Returns:
        None

    Side effects:
        - Updates UE states (`Q_current_state`, `Q_previous_state`).
        - Computes and stores rewards in the replay buffer when a valid action is performed.
        - Selects the next action via epsilon-greedy exploration/exploitation.
        - Modifies `Q_and_W_buffer_length` (increases, decreases, or maintains it).
        - Resets latency and step counters for the next simulation tick.
        - Updates Q-related tracking variables such as:
            - `Q_pcks_tx_with_success`
            - `Q_rtx_pcks_tx_with_success`
            - `Q_buffer_utilization`
            - `Q_and_W_saved_state_Q`
    """

    # Copy the current state to the previous state using deepcopy
    input_ue.Q_previous_state = deepcopy(input_ue.Q_current_state)

    if input_enable_print:
        print("UE ", input_ue.get_ue_id(), " previous state: ", input_ue.Q_previous_state)

    # Substitute the list of rtx_pcks_tx_with_success which are empty with [4]
    if len(input_ue.Q_rtx_pcks_tx_with_success[-1]) == 0:
        input_ue.Q_rtx_pcks_tx_with_success[-1] = [4]

    # Build the input state depending on the RL objective
    if input_goal_oriented == "S&L":
        if input_normalized_S:
            input_ue.Q_pcks_tx_with_success[-1] /= (
                (input_current_tick - input_ue.Q_start_tick_params_window)
                if input_current_tick - input_ue.Q_start_tick_params_window > 0
                else 1
            )
        input_ue.Q_current_state = [
            round(
                min(
                    input_ue.Q_pcks_tx_with_success[-1]
                    / np.mean(input_ue.Q_pcks_tx_with_success)
                    if np.mean(input_ue.Q_pcks_tx_with_success) != 0
                    else 0,
                    1.0,
                ),
                2,
            ),
            round(
                min(
                    np.mean(input_ue.Q_rtx_pcks_tx_with_success[-1])
                    / np.mean(
                        [np.mean(lst) for lst in input_ue.Q_rtx_pcks_tx_with_success]
                    )
                    if np.mean(
                        [np.mean(lst) for lst in input_ue.Q_rtx_pcks_tx_with_success]
                    )
                    != 0
                    else 0,
                    1.0,
                ),
                2,
            ),
            np.mean(input_ue.Q_buffer_utilization)
            / (input_ue.Q_and_W_buffer_length + 1),
            compute_normalized_linear_interpolation(
                x=input_ue.Q_and_W_buffer_length,
                x_min=input_Q_min,
                x_max=input_Q_max,
            ),
        ]

    elif input_goal_oriented == "S":
        if input_normalized_S:
            input_ue.Q_pcks_tx_with_success[-1] /= (
                (input_current_tick - input_ue.Q_start_tick_params_window)
                if input_current_tick - input_ue.Q_start_tick_params_window > 0
                else 1
            )
        input_ue.Q_current_state = [
            round(
                min(
                    input_ue.Q_pcks_tx_with_success[-1]
                    / np.mean(input_ue.Q_pcks_tx_with_success)
                    if np.mean(input_ue.Q_pcks_tx_with_success) != 0
                    else 0,
                    1.0,
                ),
                2,
            ),
            np.mean(input_ue.Q_buffer_utilization)
            / (input_ue.Q_and_W_buffer_length + 1),
            compute_normalized_linear_interpolation(
                x=input_ue.Q_and_W_buffer_length,
                x_min=input_Q_min,
                x_max=input_Q_max,
            ),
        ]

    elif input_goal_oriented == "L":
        input_ue.Q_current_state = [
            round(
                min(
                    np.mean(input_ue.Q_rtx_pcks_tx_with_success[-1])
                    / np.mean(
                        [np.mean(lst) for lst in input_ue.Q_rtx_pcks_tx_with_success]
                    )
                    if np.mean(
                        [np.mean(lst) for lst in input_ue.Q_rtx_pcks_tx_with_success]
                    )
                    != 0
                    else 0,
                    1.0,
                ),
                2,
            ),
            np.mean(input_ue.Q_buffer_utilization)
            / (input_ue.Q_and_W_buffer_length + 1),
            compute_normalized_linear_interpolation(
                x=input_ue.Q_and_W_buffer_length,
                x_min=input_Q_min,
                x_max=input_Q_max,
            ),
        ]

    if input_enable_print:
        print("UE ", input_ue.get_ue_id(), " current state: ", input_ue.Q_current_state)

    # Compute reward if applicable
    if (
        0 <= input_ue.Q_last_action < input_n_actions
        and input_ue.Q_and_W_tx_data_counter != number_of_tx_data_per_step
    ):
        reward = input_ue.reward_computation_for_only_Q(
            input_goal_oriented=input_goal_oriented
        )
        input_ue.append_Q_replay_buffer(
            input_replay_buffer_instance=[
                input_ue.Q_previous_state,
                input_ue.Q_last_action,
                reward,
                input_ue.Q_current_state,
                False,
            ]
        )
        input_ue.append_Q_reward(reward)

        if input_enable_print:
            print("UE ", input_ue.ue_id, " Reward: ", reward)

    # Compute epsilon decay for exploration
    if input_n_simulation < input_n_simulations - 20:
        epsilon_function = 1 - (input_n_simulation / (input_n_simulations - 20))
    else:
        epsilon_function = 0
    input_ue.set_epsilon(input_epsilon=max(epsilon_function, 0.01))

    # Choose next action via epsilon-greedy
    input_ue.Q_last_action = epsilon_greedy_policy(
        state=np.array(input_ue.Q_current_state),
        input_n_actions=input_n_actions,
        model=input_ue.get_Q_model(),
        input_epsilon=input_ue.get_epsilon(),
    )
    input_ue.append_Q_action_list(input_ue.Q_last_action)

    if input_enable_print:
        print(
            "UE ",
            input_ue.get_ue_id(),
            " has chosen action ",
            input_ue.Q_last_action,
            " with epsilon ",
            input_ue.get_epsilon(),
        )

    # 0: Q--,
    # 1: Q++,
    # 2: Q==
    if (
        (input_ue.Q_last_action == 0 and input_ue.Q_and_W_buffer_length == input_Q_min)
        or (input_ue.Q_last_action == 1 and input_ue.Q_and_W_buffer_length == input_Q_max)
    ):
        input_ue.Q_forbidden_action = True
    else:
        if (
            input_ue.Q_last_action == 1
            and input_ue.Q_and_W_buffer_length < input_Q_max
        ):
            input_ue.Q_and_W_buffer_length += 1
        elif (
            input_ue.Q_last_action == 0
            and input_ue.Q_and_W_buffer_length > input_Q_min
        ):
            input_ue.Q_and_W_buffer_length -= 1

    # Reset counters and tracking lists
    input_ue.Q_and_W_latencies = list()
    input_ue.Q_and_W_acks_rx_per_step_counter = 0
    input_ue.Q_and_W_pcks_tx_per_step_counter = 0
    input_ue.Q_dropped_pcks_per_step_counter = 0
    input_ue.Q_not_added_pcks_per_step_counter = 0

    # Reset per-window tracking for next step
    input_ue.Q_start_tick_params_window = input_current_tick + 1
    input_ue.Q_pcks_tx_with_success.append(0)
    input_ue.Q_rtx_pcks_tx_with_success.append([])
    input_ue.Q_buffer_utilization = list()
    input_ue.Q_and_W_saved_state_Q[-1].append(input_ue.Q_and_W_buffer_length)

    return
