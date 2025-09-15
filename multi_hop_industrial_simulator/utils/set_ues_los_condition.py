from typing import List
import numpy as np
from numpy import ndarray

from multi_hop_industrial_simulator.env.machine import Machine
from multi_hop_industrial_simulator.network.bs import BS
from multi_hop_industrial_simulator.network.ue import Ue

from multi_hop_industrial_simulator.utils.read_inputs import read_inputs

inputs = read_inputs('inputs.yaml')

ue_distribution_type = inputs.get('ue').get('ue_spatial_distribution')

# Function to determine and set the LoS/NLoS condition for a specific UE w.r.t. the BS or other UEs
def set_ues_los_condition(ue: Ue, bs: BS, machine_array: ndarray, link : str):
    """
    Determine if there is a Line-of-Sight (LoS) or Non-Line-of-Sight (NLoS) condition between UEs
    considering obstructions (machines).

    Parameters:
        ue_array : Array of UEs.
        bs : BS
        machine_array : Array of Machines.

    Returns:
        None: The function sets the 'los_nlos_info' field of each UE object with the LoS/NLoS
              information w.r.t. the BS and other UEs.

    Parameters
    ----------
    link
    machine_array
    bs
    ue
    """
    is_in_los = True

    # Pick-up gNB coordinates
    bs_coordinates = bs.get_coordinates()
    x_bs = bs_coordinates[0]
    y_bs = bs_coordinates[1]
    z_bs = bs_coordinates[2]

    # Pick-up UE coordinates
    ue_coordinates = ue.get_coordinates()
    x_ue = ue_coordinates[0]
    y_ue = ue_coordinates[1]
    z_ue = ue_coordinates[2]
    x_diff = abs(x_bs - x_ue)
    y_diff = abs(y_bs - y_ue)
    is_low_channel_condition_bool = True

    step = 0.0001  # machines[0].get_machine_size() / 2
    # If the gNB can be at the same height of UEs other cases should be inserted
    if x_diff == 0 and y_diff != 0:
                # UE and gNB have the same abscissa, move on y-axis only
        x = x_ue
        y = min(y_ue, y_bs)
        y_target = max(y_ue, y_bs)
        while y <= y_target:
            for machine in machine_array:
                if machine.x_min <= x <= machine.x_max and machine.y_min <= y <= machine.y_max:
                    z = (y - y_ue) / (y_bs - y_ue) * (z_bs - z_ue) + z_ue
                    if z <= machine.get_machine_size():
                        is_in_los = False  # Intersection, i.e NLOS
                        h_1 = ue.get_coordinates()[2]
                        h_2 = bs.get_coordinates()[2]
                        if max(h_1, h_2) >= machine.get_machine_size():
                                    # Either Tx or Rx is higher than the obstacle ==> High 3GPP model
                            is_low_channel_condition_bool = False
                        # return is_in_los
                            # break
                    # Search for the next point
            if is_in_los is False:
                y = y_target+1
            else:
                # Keep searching
                y += step

    elif x_diff != 0 and y_diff == 0:
        # UE and gNB have the same ordinate, move on x-axis only
        y = y_ue
        x = min(x_ue, x_bs)
        x_target = max(x_ue, x_bs)
        while x <= x_target:
            for machine in machine_array:
                if machine.x_min <= x <= machine.x_max and machine.y_min <= y <= machine.y_max:
                    z = (x - x_ue) / (x_bs - x_ue) * (z_bs - z_ue) + z_ue
                    if z <= machine.get_machine_size():
                        is_in_los = False  # Intersection, i.e NLOS
                        h_1 = ue.get_coordinates()[2]
                        h_2 = bs.get_coordinates()[2]
                        if max(h_1, h_2) >= machine.get_machine_size():
                            # Either Tx or Rx is higher than the obstacle ==> High 3GPP model
                            is_low_channel_condition_bool = False
                        # return is_in_los

            if is_in_los is False:
                x = x_target+1
            else:
                # Keep searching
                x += step
    elif x_diff == 0 and y_diff == 0:
        # UE and gNB have the same abscissa and ordinate:
        # just check if the UE is inside a machine, i.e move on z-axis only
        x = x_ue
        y = y_ue
        z = z_ue
        for machine in machine_array:
            if machine.x_min <= x <= machine.x_max and machine.y_min <= y <= machine.y_max:
                if z <= machine.get_machine_size():
                    is_in_los = False  # Intersection, i.e NLOS
                    h_1 = ue.get_coordinates()[2]
                    h_2 = bs.get_coordinates()[2]
                    if max(h_1, h_2) >= machine.get_machine_size():
                        # Either Tx or Rx is higher than the obstacle ==> High 3GPP model
                        is_low_channel_condition_bool = False
                    # return is_in_los
    else:
        # UE and gNB have all different coordinates
        x = min(x_ue, x_bs)
        x_target = max(x_ue, x_bs)
        angular_coefficient_projection = abs((y_bs - y_ue) / (x_bs - x_ue))
        if angular_coefficient_projection > 1:
            step = step / angular_coefficient_projection  # Reduce the increment on x such that
        # we sweep y by the original step value thus avoiding missed detections
        while x <= x_target:
            y = (x - x_ue) / (x_bs - x_ue) * (y_bs - y_ue) + y_ue
            for machine in machine_array:
                if machine.x_min <= x <= machine.x_max and machine.y_min <= y <= machine.y_max:
                    z = (y - y_ue) / (y_bs - y_ue) * (z_bs - z_ue) + z_ue
                    if z <= machine.get_machine_size():
                        is_in_los = False  # Intersection, i.e NLOS
                        h_1 = ue.get_coordinates()[2]
                        h_2 = bs.get_coordinates()[2]
                        if max(h_1, h_2) >= machine.get_machine_size():
                            is_low_channel_condition_bool = False
                            # Either Tx or Rx is higher than the obstacle ==> High 3GPP model
                        # return is_in_los

            if is_in_los is False:
                x = x_target + 1
                # No need to proceed further
                # break
            else:
                # Keep searching
                x += step

    if link == 'ue_ue':
        # ue.set_los_condition_ue_ue(is_in_los)
        ue.set_channel_condition_with_ue(is_low_channel_condition_bool=is_low_channel_condition_bool)
    if link == 'ue_bs' or link == 'bs_ue':
        ue.set_channel_condition_with_bs(is_low_channel_condition_bool=is_low_channel_condition_bool)

    return is_in_los
