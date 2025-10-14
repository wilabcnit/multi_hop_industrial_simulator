import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from multi_hop_industrial_simulator.utils.read_inputs import read_inputs
######## Parameters ########

### Insert input.yaml ####
inputs = read_inputs('inputs.yaml')
TTL = inputs.get('rl').get('router').get('TTL')

ack_bs_pr = 0.8


######## Utility functions of routing ########

def array_difference(arr1, arr2):
    """
        Compute the difference between two arrays, returning the elements
        that are present in arr1 but not in arr2.

        Args:
            arr1 (array-like): First array or list.
            arr2 (array-like): Second array or list.

        Returns:
            np.ndarray: Sorted array of elements that are in arr1 but not in arr2.
    """
    return np.setdiff1d(arr1, arr2)

# Routing Function
def get_max_index(neighbor, ack, prx, bs_seen):  # return the next_hop indices
    """
    Determine the next-hop index for routing based on neighbor connectivity,
    received ACKs, received power, and BS visibility.

    Args:
        neighbor (np.ndarray): Binary array (1 if neighbor exists, else 0).
        ack (np.ndarray): ACK reception indicators (e.g., 1 if ACK received).
        prx (np.ndarray): Received power values for each neighbor.
        bs_seen (np.ndarray): Binary flags (1 if neighbor sees BS, else 0).

    Returns:
        int or None: Index of the selected next hop.
                     -1 indicates that the BS itself is a direct neighbor.
                     None indicates no valid next hop found.
    """
    max_prx_indices = []
    max_index = None

    # CASE A: The BS is a neighbor
    if neighbor[-1] == 1:
        max_index = -1
    # CASE B: The BS is not a neighbor
    else:
        # CASE B.1: At least one ACK received and no neighbor sees the BS
        if np.sum(ack) > 0 and np.sum(bs_seen) == 0:
            max_val = np.max(ack * neighbor)
            max_index = np.argmax(ack * neighbor)

            if np.count_nonzero(ack == max_val) > 1:
                max_indeces = np.where(ack == max_val)[0]
                max_index = max_indeces[0]
                if len(max_indeces) > 1:
                    for i in max_indeces:
                        if prx[i] > prx[max_index]:
                            max_index = i
                            max_prx_indices = [i]
                        elif prx[i] == prx[max_index]:
                            max_prx_indices.append(i)

                    # Check if multiple neighbors with the same Prx
                    if len(max_prx_indices) > 1:
                        max_index = max_prx_indices[np.random.randint(0, len(max_prx_indices))]

        # CASE B.2: At least one ACK received and at least one neighbor sees the BS
        elif np.sum(ack) > 0 and np.sum(bs_seen) > 0:
            max_val = np.max(ack * neighbor * bs_seen)
            max_index = np.argmax(ack * neighbor * bs_seen)

            if np.count_nonzero((ack * bs_seen) == max_val) > 1:
                max_indeces = np.where(ack == max_val)[0]
                max_index = max_indeces[0]
                if len(max_indeces) > 1:
                    for i in max_indeces:
                        if prx[i] > prx[max_index]:
                            max_index = i
                            max_prx_indices = [i]
                        elif prx[i] == prx[max_index]:
                            max_prx_indices.append(i)

                    # Check if multiple neighbors with the same Prx
                    if len(max_prx_indices) > 1:
                        max_index = max_prx_indices[np.random.randint(0, len(max_prx_indices))]

        # CASE B.3: No ACK received
        elif np.sum(ack) == 0:
            max_indeces = np.where(neighbor == 1)[0]
            max_index = max_indeces[0]
            if len(max_indeces) > 1:
                for i in max_indeces:
                    if prx[i] > prx[max_index]:
                        max_index = i
                        max_prx_indices = [i]
                    elif prx[i] == prx[max_index]:
                        max_prx_indices.append(i)

                # Check if multiple neighbors with the same Prx
                if len(max_prx_indices) > 1:
                    max_index = max_prx_indices[np.random.randint(0, len(max_prx_indices))]

    return max_index


def select_input_DRL(input_ack, input_prx, input_nodes_number, input_DRL_type_state, input_actions_list):
    """
    Select input features for the DRL agent based on neighbor ACKs and received power.

    Args:
        input_ack (list or np.array): ACK count per neighbor (last element is BS).
        input_prx (list or np.array): Received power per neighbor.
        input_nodes_number (int): Number of nodes including BS.
        input_DRL_type_state (int): 1 or 2; defines the type of DRL state.
        input_actions_list (list): Actions taken per neighbor for normalization.

    Returns:
        np.array: Array of DRL input features.
    """

    # Check the state type
    if input_DRL_type_state == 1:
        DRL_input_list = [input_ack[-1], input_prx[-1]]

        input_ack_no_bs = cp.deepcopy(input_ack[:-1]).tolist()
        input_prx_no_bs = cp.deepcopy(input_prx[:-1]).tolist()

        for i in (range(input_nodes_number - 1)):
            # find the node with the maximum number of ACKs
            max_val = np.max(input_ack_no_bs)
            max_index = np.argmax(input_ack_no_bs)
            max_prx_indices = []
            # if there are multiple nodes with the same number of ACKs, select the one with the highest Prx
            if np.count_nonzero(input_ack_no_bs == max_val) > 1:
                max_indeces = np.where(input_ack_no_bs == max_val)[0]
                max_index = max_indeces[0]
                if len(max_indeces) > 1:
                    for i in max_indeces:
                        if input_prx_no_bs[i] > input_prx_no_bs[max_index]:
                            max_index = i
                            max_prx_indices = [i]

                        elif input_prx_no_bs[i] == input_prx_no_bs[max_index]:
                            max_prx_indices.append(i)
                    # if there are multiple nodes with the same number of ACKs and the same Prx, select one randomly
                    if len(max_prx_indices) > 1:
                        max_index = max_prx_indices[np.random.randint(0, len(max_prx_indices))]
            #Add the stats of the chosen neighbor to the DRL input list
            DRL_input_list.append(input_ack_no_bs[max_index])
            DRL_input_list.append(input_prx_no_bs[max_index])
            # Remove the node from the loop list
            input_ack_no_bs.pop(max_index)
            input_prx_no_bs.pop(max_index)

        return_list = np.array(DRL_input_list)
        return_list = np.around(return_list, 1)

        return return_list

    elif input_DRL_type_state == 2:
        DRL_input_list = [input_ack[-1] / input_actions_list[-1] if (len(input_actions_list) > 0 and input_actions_list[-1] != 0 and input_ack[-1] != 0) else 0 ]

        input_ack_no_bs = cp.deepcopy(input_ack[:-1]).tolist()
        input_ack_no_bs = [( a / b   if (b != 0 and len(input_actions_list) > 0) else 0)  for a, b in zip(input_ack_no_bs, input_actions_list[:-1])]
        input_prx_no_bs = cp.deepcopy(input_prx[:-1]).tolist()

        for i in (range(input_nodes_number - 1)):
            # find the node with the maximum number of ACKs
            max_val = np.max(input_ack_no_bs)
            max_index = np.argmax(input_ack_no_bs)
            max_prx_indices = []
            # if there are multiple nodes with the same number of ACKs, select the one with the highest Prx
            if np.count_nonzero(input_ack_no_bs == max_val) > 1:
                max_indeces = np.where(input_ack_no_bs == max_val)[0]
                max_index = max_indeces[0]
                if len(max_indeces) > 1:
                    for i in max_indeces:
                        if input_prx_no_bs[i] > input_prx_no_bs[max_index]:
                            max_index = i
                            max_prx_indices = [i]

                        elif input_prx_no_bs[i] == input_prx_no_bs[max_index]:
                            max_prx_indices.append(i)
                    # if there are multiple nodes with the same number of ACKs and the same Prx, select one randomly
                    if len(max_prx_indices) > 1:
                        max_index = max_prx_indices[np.random.randint(0, len(max_prx_indices))]
            # Add the stats of the chosen neighbor to the DRL input list
            DRL_input_list.append(input_ack_no_bs[max_index])
            # Remove the node from the loop list
            input_ack_no_bs.pop(max_index)
            input_prx_no_bs.pop(max_index)

        return_list = np.array(DRL_input_list)
        return_list = np.around(return_list, 2)

        return return_list


def ttl_reset(obs_array, address_index):  # mi serve
    """
    Reset the TTL of a specific UE.

    Args:
      obs_array: observation array.
      address_index: index of the UE to reset

    Returns:

    """
    obs_array[0][address_index] = 0
    obs_array[1][address_index] = 0
    obs_array[2][address_index] = 0
    obs_array[3][address_index] = 0
    obs_array[4][address_index] = 0

    return obs_array


def compute_normalized_linear_interpolation(x, x_min, x_max):
    """
    Compute a normalized value of x using linear interpolation between x_min and x_max.

    Args:
        x (float): Value to normalize.
        x_min (float): Minimum value of the range.
        x_max (float): Maximum value of the range.

    Returns:
        float: Normalized value between 0 and 1, rounded to 2 decimal places.
    """

    normalized_value = (x - x_min) / (x_max - x_min)

    return round(normalized_value, 2)
