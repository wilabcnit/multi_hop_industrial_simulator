import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from timessim.utils.read_inputs import read_inputs
######## Parameters ########

### Insert input.yaml ####
inputs = read_inputs('inputs.yaml')
TTL = inputs.get('rl').get('router').get('TTL')

ack_bs_pr = 0.8

##########################

######## Utility functions of routing ########

# Function to compute the difference between two arrays
def array_difference(arr1, arr2):
    return np.setdiff1d(arr1, arr2)

# Routing Function
def get_max_index(neighbor, ack, prx, bs_seen):  # return the next_hop indices
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
                    # if np.random.rand() > 0.2:
                    for i in max_indeces:
                        # if np.random.rand() > 0.2:
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
                    # if np.random.rand() > 0.2:
                    for i in max_indeces:
                        # if np.random.rand() > 0.2:
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
                # if np.random.rand() > 0.2:
                for i in max_indeces:
                    # if np.random.rand() > 0.2:
                    if prx[i] > prx[max_index]:
                        max_index = i
                        max_prx_indices = [i]
                    elif prx[i] == prx[max_index]:
                        max_prx_indices.append(i)

                # Check if multiple neighbors with the same Prx
                if len(max_prx_indices) > 1:
                    max_index = max_prx_indices[np.random.randint(0, len(max_prx_indices))]

    return max_index

# Function to select the input for the DRL agent
def select_input_DRL(input_ack, input_prx, input_nodes_number, input_DRL_type_state, input_actions_list):

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

# Worker function to handle broadcast trasmsissions
def broadcast_handling(obs_array, broadcast_response):
    old_state = obs_array[0]

    for i in range(len(broadcast_response[0])):

        if broadcast_response[0][i] == 1:
            obs_array[0][i] = 1
            obs_array[1][i] += 1
            obs_array[2][i] = broadcast_response[1][i]
            obs_array[3][i] = 0

        elif not broadcast_response[0][i] and old_state[i] == 1:
            obs_array[3][i] += 1

        if obs_array[3][i] >= TTL:
            obs_array = ttl_reset(obs_array, i)

    return obs_array

# Worker function to reset the TTL of a specific UE
def ttl_reset(obs_array, address_index):  # mi serve
    obs_array[0][address_index] = 0
    obs_array[1][address_index] = 0
    obs_array[2][address_index] = 0
    obs_array[3][address_index] = 0
    obs_array[4][address_index] = 0

    return obs_array


# Worker function to compute the normalized linear interpolation
def compute_normalized_linear_interpolation(x, x_min, x_max):

    normalized_value = (x - x_min) / (x_max - x_min)

    return round(normalized_value, 2)
