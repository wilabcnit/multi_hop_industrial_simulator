import math
import numpy as np
import random

from multi_hop_industrial_simulator.network.bs import BS
from numpy import ndarray

from multi_hop_industrial_simulator.channel_models.THz_channel import THzChannel


def check_for_neighbours(ue_array: ndarray, machine_array: ndarray, bs: BS, input_snr_threshold_db: float,
                         input_thz_channel: THzChannel, input_carrier_frequency_ghz: float, input_bandwidth_hz: float,
                         input_apply_fading: bool, input_clutter_density: float, input_shadowing_sample_index: int,
                         use_channel_measurements: bool, input_average_clutter_height_m: float,
                         antenna_gain_model: str = None):
    """
    Method to check if UEs have neighbours or are connected to the BS ->
    used to determine which UEs can directly reach the BS or not

    Args:
      ue_array: ndarray: array of UEs
      machine_array: ndarray: array of machines
      bs: BS: 
      input_snr_threshold_db: float: SNR (or SINR) threshold for the input channel
      input_thz_channel: THzChannel: channel model used
      input_carrier_frequency_ghz: float: carrier frequency (GHz)
      input_bandwidth_hz: float: bandwidth (Hz)
      input_apply_fading: bool: True if there is fading; False otherwise
      input_clutter_density: float: average clutter density
      input_shadowing_sample_index: int: index of shadowing sample
      use_channel_measurements: bool: True if Huawei measurements are used; False otherwise
      input_average_clutter_height_m: float: average clutter height (m)
      antenna_gain_model: str:  (Default value = None) type of antenna gain model

    Returns:
        None

    """

    neighbours_found = False
    n_connected = 0
    not_connected_ues = list()
    connected_ues = list()

    for ue in ue_array:
        # Compute the distance from the BS
        distance_to_bs_m = np.sqrt((ue.x - bs.x) ** 2 + (ue.y - bs.y) ** 2 + (ue.z - bs.z) ** 2)
        los_cond = 'bs_ue'

        # We assume one single shadowing sample per transmission
        # Compute the SNR
        snr_db = input_thz_channel.get_3gpp_snr_db(
            tx=ue, rx=bs,
            carrier_frequency_ghz=input_carrier_frequency_ghz,
            tx_rx_distance_m=distance_to_bs_m,
            apply_fading=input_apply_fading,
            bandwidth_hz=input_bandwidth_hz,
            clutter_density=input_clutter_density,
            input_shadowing_sample_index=input_shadowing_sample_index,
            antenna_gain_model=antenna_gain_model,
            use_channel_measurements=use_channel_measurements,
            input_average_clutter_height_m=input_average_clutter_height_m,
            los_cond=los_cond)

        # Check if the SNR meets the threshold
        if snr_db >= input_snr_threshold_db:
            n_connected += 1
            connected_ues.append(ue.get_ue_id())
        else:
            not_connected_ues.append(ue.get_ue_id())

    # Ensure approximately half of the UEs are connected to the BS
    if len(not_connected_ues) > math.floor(len(ue_array) / 2):
        # print("Need to connect MORE UEs to the BS.")
        additional_ues_connected = math.floor(len(not_connected_ues) - math.floor(len(ue_array) / 2))
        for i in range(additional_ues_connected):
            connected_to_bs = False
            changing_ue = random.choice(not_connected_ues)
            for ue in ue_array:
                if ue.get_ue_id() == changing_ue and ue.get_ue_id() in not_connected_ues:
                    not_connected_ues.remove(ue.get_ue_id())
                    machine_first = [1, 3, 5, 7]
                    while connected_to_bs is False:
                        machine_chosen = random.choice(machine_first)
                        x = (random.random() * (machine_array[machine_chosen].x_max - machine_array[machine_chosen].x_min) +
                             machine_array[machine_chosen].x_min)
                        y = (random.random() * (machine_array[machine_chosen].y_max - machine_array[machine_chosen].y_min) +
                             machine_array[machine_chosen].y_min)
                        z = random.random() * machine_array[machine_chosen].z_center * 2
                        distance_to_bs_m = np.sqrt((x - bs.x) ** 2 + (y - bs.y) ** 2 + (z - bs.z) ** 2)
                        los_cond = 'bs_ue'

                        # We assume one single shadowing sample per transmission
                        # Compute the SNR
                        snr_db = input_thz_channel.get_3gpp_snr_db(
                            tx=ue, rx=bs,
                            carrier_frequency_ghz=input_carrier_frequency_ghz,
                            tx_rx_distance_m=distance_to_bs_m,
                            apply_fading=input_apply_fading,
                            bandwidth_hz=input_bandwidth_hz,
                            clutter_density=input_clutter_density,
                            input_shadowing_sample_index=input_shadowing_sample_index,
                            antenna_gain_model=antenna_gain_model,
                            use_channel_measurements=use_channel_measurements,
                            input_average_clutter_height_m=input_average_clutter_height_m,
                            los_cond=los_cond)

                        # Check if the SNR meets the threshold
                        if snr_db >= input_snr_threshold_db:
                            connected_to_bs = True
                            ue.set_coordinates(x, y, z)
                            connected_ues.append(ue.get_ue_id())

    # Ensure approximately half of the UEs are connected to the BS
    elif len(connected_ues) > math.floor(len(ue_array) / 2):
        # print("Need to connect LESS UEs to the BS.")
        additional_ues_connected = math.floor(math.floor(len(ue_array) / 2) - len(not_connected_ues))
        for i in range(additional_ues_connected):
            connected_to_bs = True
            changing_ue = random.choice(connected_ues)
            for ue in ue_array:
                if ue.get_ue_id() == changing_ue and ue.get_ue_id() in connected_ues:
                    connected_ues.remove(ue.get_ue_id())
                    machine_second = [0, 2, 4, 6]
                    while connected_to_bs is True:
                        machine_chosen = random.choice(machine_second)
                        x = (random.random() * (
                                    machine_array[machine_chosen].x_max - machine_array[machine_chosen].x_min) +
                             machine_array[machine_chosen].x_min)
                        y = (random.random() * (
                                    machine_array[machine_chosen].y_max - machine_array[machine_chosen].y_min) +
                             machine_array[machine_chosen].y_min)
                        z = random.random() * machine_array[machine_chosen].z_center * 2
                        distance_to_bs_m = np.sqrt((x - bs.x) ** 2 + (y - bs.y) ** 2 + (z - bs.z) ** 2)
                        los_cond = 'bs_ue'

                        # We assume one single shadowing sample per transmission
                        # Compute the SNR
                        snr_db = input_thz_channel.get_3gpp_snr_db(
                            tx=ue, rx=bs,
                            carrier_frequency_ghz=input_carrier_frequency_ghz,
                            tx_rx_distance_m=distance_to_bs_m,
                            apply_fading=input_apply_fading,
                            bandwidth_hz=input_bandwidth_hz,
                            clutter_density=input_clutter_density,
                            input_shadowing_sample_index=input_shadowing_sample_index,
                            antenna_gain_model=antenna_gain_model,
                            use_channel_measurements=use_channel_measurements,
                            input_average_clutter_height_m=input_average_clutter_height_m,
                            los_cond=los_cond)

                        # Check if the SNR meets the threshold
                        if snr_db < input_snr_threshold_db:
                            connected_to_bs = False
                            ue.set_coordinates(x, y, z)
                            not_connected_ues.append(ue.get_ue_id())

    for ue in ue_array:
        neighbours_found = False
        if ue.get_ue_id() in not_connected_ues:
            not_connected_ues.remove(ue.get_ue_id())
            for other_ue in ue_array:
                if other_ue != ue and other_ue.get_ue_id() in connected_ues:
                    distance_to_other_ue_m = np.sqrt((ue.x - other_ue.x) ** 2 + (ue.y - other_ue.y) ** 2 +
                                                     (ue.z - other_ue.z) ** 2)
                    los_cond = 'ue_ue'

                    # Compute the SNR
                    snr_db = input_thz_channel.get_3gpp_snr_db(
                        tx=ue, rx=other_ue,
                        carrier_frequency_ghz=input_carrier_frequency_ghz,
                        tx_rx_distance_m=distance_to_other_ue_m,
                        apply_fading=input_apply_fading,
                        bandwidth_hz=input_bandwidth_hz,
                        clutter_density=input_clutter_density,
                        input_shadowing_sample_index=input_shadowing_sample_index,
                        antenna_gain_model=antenna_gain_model,
                        use_channel_measurements=use_channel_measurements,
                        input_average_clutter_height_m=input_average_clutter_height_m,
                        los_cond=los_cond)

                    # Check if the SNR meets the threshold
                    if snr_db >= input_snr_threshold_db:
                        neighbours_found = True

            # No neighbours found
            if neighbours_found is False:
                # Machine in the second tear of the factory ->
                machine_second = [0, 2, 4, 6]
                while neighbours_found is False:
                    machine_chosen = random.choice(machine_second)
                    x = (random.random() * (machine_array[machine_chosen].x_max - machine_array[machine_chosen].x_min) +
                         machine_array[machine_chosen].x_min)
                    y = (random.random() * (machine_array[machine_chosen].y_max - machine_array[machine_chosen].y_min) +
                         machine_array[machine_chosen].y_min)
                    z = random.random() * machine_array[machine_chosen].z_center * 2
                    for other_ue in ue_array:
                        if other_ue != ue and other_ue.get_ue_id() in connected_ues:
                            distance_to_other_ue_m = np.sqrt((x - other_ue.x) ** 2 + (y - other_ue.y) ** 2 +
                                                             (z - other_ue.z) ** 2)
                            los_cond = 'ue_ue'

                            # Compute the SNR between UEs
                            snr_db_ue_ue = input_thz_channel.get_3gpp_snr_db(
                                tx=ue, rx=other_ue,
                                carrier_frequency_ghz=input_carrier_frequency_ghz,
                                tx_rx_distance_m=distance_to_other_ue_m,
                                apply_fading=input_apply_fading,
                                bandwidth_hz=input_bandwidth_hz,
                                clutter_density=input_clutter_density,
                                input_shadowing_sample_index=input_shadowing_sample_index,
                                antenna_gain_model=antenna_gain_model,
                                use_channel_measurements=use_channel_measurements,
                                input_average_clutter_height_m=input_average_clutter_height_m,
                                los_cond=los_cond)

                            # Evaluate the SNR between UE and BS
                            distance_to_bs_m = np.sqrt((x - bs.x) ** 2 + (y - bs.y) ** 2 +
                                                       (z - bs.z) ** 2)
                            los_cond = 'bs_ue'
                            snr_db_ue_bs = input_thz_channel.get_3gpp_snr_db(
                                tx=ue, rx=bs,
                                carrier_frequency_ghz=input_carrier_frequency_ghz,
                                tx_rx_distance_m=distance_to_bs_m,
                                apply_fading=input_apply_fading,
                                bandwidth_hz=input_bandwidth_hz,
                                clutter_density=input_clutter_density,
                                input_shadowing_sample_index=input_shadowing_sample_index,
                                antenna_gain_model=antenna_gain_model,
                                use_channel_measurements=use_channel_measurements,
                                input_average_clutter_height_m=input_average_clutter_height_m,
                                los_cond=los_cond)

                            # Check if the SNRs meets the threshold
                            if snr_db_ue_ue >= input_snr_threshold_db > snr_db_ue_bs:
                                neighbours_found = True
                                ue.set_coordinates(x, y, z)
                                break
