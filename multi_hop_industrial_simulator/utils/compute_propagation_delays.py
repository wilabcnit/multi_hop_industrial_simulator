"""
    Utility scripts to compute the propagation delays between all network elements
"""
import math
import numpy as np

from multi_hop_industrial_simulator.network.bs import BS
from numpy import ndarray
from scipy.constants import c

def compute_propagation_delays(ue_array: ndarray, bs: BS, input_simulator_tick_duration_s: float):
    """
    Compute the propagation delays in seconds between all network elements

    Args:
      ue_array: ndarray: array of UEs in the environment
      bs: BS: base station object
      input_simulator_tick_duration_s: float: duration of simulator tick in seconds

    Returns:
        None

    """
    for ue in ue_array:
        # Compute the distance from the BS
        distance_to_bs_m = np.sqrt((ue.x - bs.x) ** 2 + (ue.y - bs.y) ** 2 + (ue.z - bs.z) ** 2)
        # print("UE ", ue.get_ue_id(), " BS - distance: ", distance_to_bs_m)

        # Compute the propagation delay from the BS
        prop_delay_to_bs_s = distance_to_bs_m / c
        prop_delay_to_bs_tick = math.ceil(prop_delay_to_bs_s / input_simulator_tick_duration_s)
        ue.set_prop_delay_to_bs_s(input_prop_delay_to_bs_s=prop_delay_to_bs_s)
        ue.set_prop_delay_to_bs_tick(input_prop_delay_to_bs_tick=prop_delay_to_bs_tick)
        # print("UE ", ue.get_ue_id(), " - BS propagation delay: ", prop_delay_to_bs_tick)

        # Compute the distance and propagation delay from the other UEs
        for other_ue in ue_array:
            if other_ue != ue:
                distance_to_other_ue_m = np.sqrt((ue.x - other_ue.x) ** 2 + (ue.y - other_ue.y) ** 2 +
                                                 (ue.z - other_ue.z) ** 2)
                prop_delay_to_other_ue_s = distance_to_other_ue_m / c
                prop_delay_to_other_ue_tick = math.ceil(prop_delay_to_other_ue_s / input_simulator_tick_duration_s)
                ue.add_prop_delay_to_ue_s(input_ue_id=other_ue.get_ue_id(),
                                          input_prop_delay_to_ue_s=prop_delay_to_other_ue_s)
                ue.add_prop_delay_to_ue_tick(input_ue_id=other_ue.get_ue_id(),
                                             input_prop_delay_to_ue_tick=prop_delay_to_other_ue_tick)

                # print("UE ", ue.get_ue_id(), " - ", other_ue.get_ue_id(), " propagation delay: ",
                #        prop_delay_to_other_ue_tick)
                # print("UE ", ue.get_ue_id(), " - ", other_ue.get_ue_id(), " distance: ",
                #       distance_to_other_ue_m)
