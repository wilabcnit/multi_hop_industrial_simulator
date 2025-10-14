from numpy import ndarray


def get_ue_min_state_duration(ue_array: ndarray):
    """
    Compute the minimum state duration in ticks among all UEs

    Args:
      ue_array: ndarray: array of UEs

    Returns:
        minimum state duration in ticks

    """

    t_states_ues_tick = [ue.get_state_duration() for ue in ue_array]  # All UEs state duration

    return min(t_states_ues_tick)
