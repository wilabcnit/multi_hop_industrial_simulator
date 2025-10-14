"""
    Utility scripts to compute the duration in seconds of the simulation tick
"""
import sys


def compute_simulator_tick_duration(input_params_dict: dict):
    """
    Compute the duration in seconds of the simulation tick

    Args:
      input_params_dict: dict: dictionary of input parameters

    Returns:
        simulation tick duration in seconds

    """
    if 'radio' in input_params_dict:
        ack_size_bytes = input_params_dict.get('aloha_protocol').get('ack_size_bytes')
        bit_rate_gbits = input_params_dict.get('radio').get('bit_rate_gbits') * 1e9
        t_ack = (ack_size_bytes*8)/bit_rate_gbits
        int_value = 2
        simulation_tick = t_ack/int_value
        return simulation_tick
    else:
        sys.exit("Specify the field 'radio' in params!")
