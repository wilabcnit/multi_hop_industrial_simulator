from numpy import ndarray

# Function to get the minimum state duration (in ticks) among all UEs
def get_ue_min_state_duration(ue_array: ndarray):
    """

    Args:
      ue_array: ndarray: 

    Returns:

    """

    t_states_ues_tick = [ue.get_state_duration() for ue in ue_array]  # All UEs state duration

    return min(t_states_ues_tick)
