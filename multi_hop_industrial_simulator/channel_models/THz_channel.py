import sys
from typing import List, Tuple

import numpy as np
from math import log10

import pandas as pd
from scipy import constants as ct
from multi_hop_industrial_simulator.network.ue import Ue
from multi_hop_industrial_simulator.network.bs import BS
from multi_hop_industrial_simulator.utils.get_antenna_gain_db import get_antenna_gain_db
from multi_hop_industrial_simulator.utils.read_input_file import read_input_file


class THzChannel:
    """
        Implementing the THz channel
    """

    def __init__(self, params):
        self.params = params
        self.molecular_coefficients_df = read_input_file(file_name=self.params.get('channel').get('input_file_name'),
                                                         sheet_name=self.params.get('channel').get('input_sheet_name'))
        self.molecular_coefficients_df_column_names = self.molecular_coefficients_df.columns  # Pick up the column names
        self.molecular_coefficients_step_behaviour = self.get_molecular_absorption_coefficients_df()
        self.molecular_coefficients_step_behaviour_column_names = self.molecular_coefficients_step_behaviour.columns
        self.molecular_coefficients_df = (
            self.molecular_coefficients_df.sort_values(self.molecular_coefficients_df_column_names[0]))
        self.snr_threshold_db = self.params.get('channel').get('snr_threshold_db')
        self.noise_temperature_k = self.params.get('channel').get('noise_temperature')  # Kelvin
        self.path_loss_exponent = self.params.get('channel').get('path_loss_exponent')
        self.shadowing_std_dev_db = self.params.get('channel').get('shadowing_std_dev_db')
        self.fading_param = self.params.get('channel').get('fading_param')
        self.shadowing_samples_array_db = []  # Array of shadowing samples in dB

    def get_friis_path_loss_db(self, tx_rx_distance_m: float):
        """

        Args:
          tx_rx_distance_m: float: distance between transmitter and receiver in meters

        Returns:
            Friis path loss in dB given the distance in meters

        """

        # Constants
        c = ct.speed_of_light
        pi = ct.pi

        # Computation of the Friis formula
        l_0 = 20 * log10(4 * pi / c)
        l_1 = 10 * self.path_loss_exponent * log10(tx_rx_distance_m)
        return l_0 + l_1

    def get_3gpp_path_loss_db(self, tx, rx, carrier_frequency_ghz: float, tx_rx_distance_m: float,
                              apply_fading: bool, clutter_density: float, input_shadowing_sample_index: int,
                              los_cond: str, use_huawei_measurements: bool, input_average_clutter_height_m: float):
        """

        Args:
          tx: UE or BS (transmitter)
          rx: BS or UE (receiver)
          carrier_frequency_ghz: float: frequency of carrier in GHz
          tx_rx_distance_m: float: distance between transmitter and receiver in meters
          apply_fading: bool: whether to apply fading
          clutter_density: float: density of obstacles in the reference environment
          input_shadowing_sample_index: int: index of shadowing sample in the array of values
          los_cond: str: Line of Sight or Non LoS condition
          use_huawei_measurements: bool: True if channel model experimentally derived by Huawei measurements;
                                         False if 3GPP TR 38.901 channel model
          input_average_clutter_height_m: float: average height of clutter in meters

        Returns:
            3GPP path loss in dB given the distance in meters and the carrier frequency in GHz

        """
        # Obtain shadowing sample
        xi = self.shadowing_samples_array_db[input_shadowing_sample_index]

        # Path loss in LoS
        if use_huawei_measurements:
            path_loss_db = 10 * 2.26 * log10(tx_rx_distance_m) + 32.4 + 20 * log10(carrier_frequency_ghz)
        else:
            path_loss_db = 31.84 + 21.50 * log10(tx_rx_distance_m) + 19.00 * log10(carrier_frequency_ghz)

        if los_cond == 'ue_ue':
            index = rx.ue_id
            is_los_condition = tx.is_in_los_ues[index]
            is_low_channel_condition = tx.get_channel_condition_with_ue(ue_index=index)
        elif los_cond == 'bs_ue':
            if isinstance(tx, BS):
                # Downlink
                ue = rx
            else:
                # Uplink
                ue = tx
            is_los_condition = ue.is_in_los
            is_low_channel_condition = ue.get_channel_condition_with_bs()
        else:
            sys.exit('los_cond not valid when computing the 3GPP path loss')

        if not is_los_condition:
            # NLoS
            if use_huawei_measurements:
                path_loss_db = 10 * 3.27 * log10(tx_rx_distance_m) + 32.4 + 20 * log10(carrier_frequency_ghz)
            else:
                if clutter_density < 0.4:
                    # Sparse clutter
                    if is_low_channel_condition:
                        # InF-SL
                        path_loss_sl_db = 33 + 25.5 * log10(tx_rx_distance_m) + 20 * log10(carrier_frequency_ghz)
                        path_loss_db = max(path_loss_db, path_loss_sl_db)
                    else:
                        # InF-SH
                        path_loss_sh_db = 32.4 + 23.0 * log10(tx_rx_distance_m) + 20 * log10(carrier_frequency_ghz)
                        path_loss_db = max(path_loss_db, path_loss_sh_db)
                else:
                    # Dense clutter
                    if is_low_channel_condition:
                        # InF-DL
                        path_loss_sl_db = 33 + 25.5 * log10(tx_rx_distance_m) + 20 * log10(carrier_frequency_ghz)
                        path_loss_dl_db = 18.6 + 35.7 * log10(tx_rx_distance_m) + 20 * log10(carrier_frequency_ghz)
                        path_loss_db = max(path_loss_db, path_loss_sl_db, path_loss_dl_db)
                    else:
                        # InF_DH
                        path_loss_dh_db = 33.63 + 21.9 * log10(tx_rx_distance_m) + 20 * log10(carrier_frequency_ghz)
                        path_loss_db = max(path_loss_db, path_loss_dh_db)

        # Apply fading (if desired)
        if apply_fading:
            path_loss_db = path_loss_db + xi

        return path_loss_db

    def get_molecular_absorption_coefficients_df(self):
        """
        Returns: molecular absorption coefficients in a DataFrame
        """
        # Create a step function for the molecular absorption coefficients
        # Assuming constant values between two frequencies f1 and f2, as K1 in [f1,(f1+f2)/2] and K2 in [(f1+f2)/2, f2]
        frequency_middle_point_list = []
        molecular_absorption_coefficient_list = []

        for i in range(1, len(self.molecular_coefficients_df)):
            frequency1 = self.molecular_coefficients_df[self.molecular_coefficients_df_column_names[0]].iloc[i - 1]
            frequency2 = self.molecular_coefficients_df[self.molecular_coefficients_df_column_names[0]].iloc[i]

            # Calculate midpoint and corresponding molecular absorption coefficient
            frequency_middle_point = (frequency1 + frequency2) / 2
            molecular_absorption_coefficient = (
                self.molecular_coefficients_df[self.molecular_coefficients_df_column_names[1]].iloc)[i - 1]

            frequency_middle_point_list.append(frequency_middle_point)
            molecular_absorption_coefficient_list.append(molecular_absorption_coefficient)

        # Create a new DataFrame
        return pd.DataFrame({self.molecular_coefficients_df_column_names[0]: frequency_middle_point_list,
                             self.molecular_coefficients_df_column_names[1]: molecular_absorption_coefficient_list})

    def get_molecular_absorption_coefficients(self, carrier_frequency_hz: float, bandwidth_hz: float):
        """

        Args:
          carrier_frequency_hz: float: frequency of the carrier in Hz
          bandwidth_hz: float: bandwidth of the carrier in Hz

        Returns:
            molecular absorption coefficients and the corresponding frequency intervals

        """

        # Calculate the lower and upper bounds of the frequency interval
        lower_bound = carrier_frequency_hz - bandwidth_hz / 2
        upper_bound = carrier_frequency_hz + bandwidth_hz / 2

        # Filter rows based on the frequency interval
        filtered_df = self.molecular_coefficients_step_behaviour[
            (self.molecular_coefficients_step_behaviour[self.molecular_coefficients_step_behaviour_column_names[0]] >=
             lower_bound) &
            (self.molecular_coefficients_step_behaviour[self.molecular_coefficients_step_behaviour_column_names[0]] <=
             upper_bound)].reset_index(drop=True)

        # Find the index of the frequency middle point above our upper bound
        lower_index = self.molecular_coefficients_step_behaviour[
            self.molecular_coefficients_step_behaviour[self.molecular_coefficients_step_behaviour_column_names[0]] >
            lower_bound][self.molecular_coefficients_step_behaviour_column_names[0]].idxmin()
        upper_index = self.molecular_coefficients_step_behaviour[
            self.molecular_coefficients_step_behaviour[self.molecular_coefficients_step_behaviour_column_names[0]] >
            upper_bound][self.molecular_coefficients_step_behaviour_column_names[0]].idxmin()

        # Create new rows for fc - B/2 and fc + B/2
        lower_boundary_row = {filtered_df.columns[0]: carrier_frequency_hz - bandwidth_hz / 2,
                              filtered_df.columns[1]: self.molecular_coefficients_step_behaviour[
                                  self.molecular_coefficients_step_behaviour_column_names[1]].iloc[lower_index]}
        upper_boundary_row = {filtered_df.columns[0]: carrier_frequency_hz + bandwidth_hz / 2,
                              filtered_df.columns[1]: self.molecular_coefficients_step_behaviour[
                                  self.molecular_coefficients_step_behaviour_column_names[1]].iloc[upper_index]}

        # Concatenate the new rows with the existing DataFrame
        extended_df = pd.concat([pd.DataFrame([lower_boundary_row]), filtered_df, pd.DataFrame([upper_boundary_row])],
                                ignore_index=True)

        # Loop over the extended df and return the frequency intervals of each step
        # and the corresponding molecular absorption coefficient
        frequency_intervals_list = []
        molecular_absorption_coefficient_list = []
        for i in range(1, len(extended_df)):
            frequency1 = extended_df[extended_df.columns[0]].iloc[i - 1]
            frequency2 = extended_df[extended_df.columns[0]].iloc[i]

            # Calculate the frequency interval and corresponding molecular absorption coefficient
            frequency_interval = (frequency1, frequency2)
            molecular_absorption_coefficient = extended_df[extended_df.columns[1]].iloc[i]

            # Save the above files in the lists
            frequency_intervals_list.append(frequency_interval)
            molecular_absorption_coefficient_list.append(molecular_absorption_coefficient)

        return frequency_intervals_list, molecular_absorption_coefficient_list

    def compute_integral_over_bandwidth_db(self, frequency_intervals_list_hz: List[Tuple[float, float]],
                                           molecular_absorption_coefficient_list: List[float],
                                           tx_rx_distance_m: float):
        """

        Args:
          frequency_intervals_list_hz: List[Tuple[float, float]]: list of frequency intervals in Hz [starting frequency,
          ending frequency]
          molecular_absorption_coefficient_list: List[float]: list of molecular absorption coefficients
          tx_rx_distance_m: float: distance between the transmitter and the receiver in meters

        Returns:
            integral over the bandwidth in dB

        """
        integral_over_bandwidth = 0.0
        for frequency_interval, molecular_absorption_coefficient in zip(frequency_intervals_list_hz,
                                                                        molecular_absorption_coefficient_list):
            first_term = - 1 / frequency_interval[1] + 1 / frequency_interval[0]
            second_term = np.exp(-molecular_absorption_coefficient * tx_rx_distance_m)
            integral_over_bandwidth = integral_over_bandwidth + first_term * second_term

        return 10 * log10(integral_over_bandwidth)

    def get_molecular_noise_power_dbw(self, frequency_intervals_list_hz: List[Tuple[float, float]],
                                      molecular_absorption_coefficient_list: List[float], tx_rx_distance_m: float):
        """

        Args:
          frequency_intervals_list_hz: List[Tuple[float, float]]: list of frequency intervals in Hz [starting frequency,
          ending frequency]
          molecular_absorption_coefficient_list: List[float]: list of molecular absorption coefficients
          tx_rx_distance_m: float: distance between the transmitter and the receiver in meters

        Returns:
            molecular noise power in dBW

        """
        molecular_noise_power = 0.0
        for frequency_interval, molecular_absorption_coefficient in zip(frequency_intervals_list_hz,
                                                                        molecular_absorption_coefficient_list):
            first_term = frequency_interval[1] - frequency_interval[0]
            molecular_absorption_gain = np.exp(-molecular_absorption_coefficient * tx_rx_distance_m)
            second_term = first_term * molecular_absorption_gain
            molecular_noise_power_single_term = (
                    ct.Boltzmann * self.noise_temperature_k * (first_term + second_term))
            molecular_noise_power = molecular_noise_power + molecular_noise_power_single_term

        return 10 * log10(molecular_noise_power)

    def get_3gpp_snr_db(self, tx, rx, carrier_frequency_ghz: float, tx_rx_distance_m: float,
                        apply_fading: bool, bandwidth_hz: float, clutter_density: float,
                        input_shadowing_sample_index: int,  use_huawei_measurements: bool,
                        input_average_clutter_height_m: float, los_cond: str,
                        antenna_gain_model: str = None):
        """

        Args:
          tx: UE or BS
          rx: BS or UE
          carrier_frequency_ghz: float: frequency of the carrier in GHz
          tx_rx_distance_m: float: distance between the transmitter and the receiver in meters
          apply_fading: bool: whether to apply fading
          bandwidth_hz: float: bandwidth of the carrier in Hz
          clutter_density: float: density of obstacles in the reference environment
          input_shadowing_sample_index: int: index of input shadowing sample in the array of values
          use_huawei_measurements: bool: True if channel model experimentally derived by Huawei measurements;
                                         False if 3GPP TR 38.901 channel model
          input_average_clutter_height_m: float: average height of the clutter in meters
          los_cond: str: Line of Sight or Non LoS condition
          antenna_gain_model: str: model of the antenna gain (if specified as input, Default value = None)

        Returns:
            SNR in dB according to the 3GPP model

        """

        path_loss_db = self.get_3gpp_path_loss_db(tx=tx, rx=rx, carrier_frequency_ghz=carrier_frequency_ghz,
                                                  tx_rx_distance_m=tx_rx_distance_m,
                                                  apply_fading=apply_fading, clutter_density=clutter_density,
                                                  input_shadowing_sample_index=input_shadowing_sample_index,
                                                  use_huawei_measurements=use_huawei_measurements,

                                                  los_cond= los_cond, 

                                                  input_average_clutter_height_m=input_average_clutter_height_m)

        tx_power_dbm = tx.transceiver_params.get('Transmit power')
        tx_power_dbw = tx_power_dbm - 30
        tx_antenna_efficiency = tx.transceiver_params.get('Antenna efficiency')
        tx_antenna_efficiency_db = 10 * log10(tx_antenna_efficiency)
        rx_antenna_efficiency = rx.transceiver_params.get('Antenna efficiency')
        rx_antenna_efficiency_db = 10 * log10(rx_antenna_efficiency)
        if antenna_gain_model is None or antenna_gain_model == 'input':
            # Take antenna gain from input
            tx_antenna_gain_db = tx.transceiver_params.get('Antenna gain')
            rx_antenna_gain_db = rx.transceiver_params.get('Antenna gain')
        elif antenna_gain_model == 'kumar salvia':
            # Taken from Vijay Kumar Salvia "Antenna and wave propagation" (Ed. Laxmi, 2007)
            tx_antenna_gain_db = get_antenna_gain_db(
                number_of_antennas=tx.transceiver_params.get('Number of antennas'))
            rx_antenna_gain_db = get_antenna_gain_db(number_of_antennas=rx.transceiver_params.get('Number of antennas'))
        else:
            exit('Invalid antenna_gain_model when computing the SNR according to the 3GPP model')
        rx_noise_figure_db = rx.transceiver_params.get('Noise figure')
        rx_noise_figure = 10 ** (rx_noise_figure_db / 10)
        noise_power_dbw = self.get_thermal_noise_power_dbw(input_noise_figure=rx_noise_figure,
                                                           bandwidth_hz=bandwidth_hz)

        snr_db = (tx_power_dbw + tx_antenna_efficiency_db + rx_antenna_efficiency_db + tx_antenna_gain_db +
                  rx_antenna_gain_db - path_loss_db - noise_power_dbw)
        return snr_db

    def get_3gpp_prx_db(self, tx, rx, carrier_frequency_ghz: float, tx_rx_distance_m: float,
                        apply_fading: bool, bandwidth_hz: float, clutter_density: float,
                        input_shadowing_sample_index: int,  use_huawei_measurements: bool,
                        input_average_clutter_height_m: float, los_cond: str,
                        antenna_gain_model: str = None):
        """

        Args:
          tx: UE or BS
          rx: BS or UE
          carrier_frequency_ghz: float: frequency of the carrier in GHz
          tx_rx_distance_m: float: distance between the transmitter and receiver in meters
          apply_fading: bool: whether to apply fading
          bandwidth_hz: float: bandwidth of the transmitter in GHz
          clutter_density: float: density of obstacles within the reference environment
          input_shadowing_sample_index: int: index of the shadowing sample in the array of values
          use_huawei_measurements: bool: True if channel model experimentally derived by Huawei measurements;
                                         False if 3GPP TR 38.901 channel model
          input_average_clutter_height_m: float: average clutter height in meters
          los_cond: str: Line of Sight or Non LoS condition
          antenna_gain_model: str: model of the antenna gain (if specified as input, Default value = None)

        Returns:
            received power in dB according to the 3GPP model

        """

        # Compute path loss
        path_loss_db = self.get_3gpp_path_loss_db(tx=tx, rx=rx, carrier_frequency_ghz=carrier_frequency_ghz,
                                                  tx_rx_distance_m=tx_rx_distance_m,
                                                  apply_fading=apply_fading, clutter_density=clutter_density,
                                                  input_shadowing_sample_index=input_shadowing_sample_index,
                                                  use_huawei_measurements=use_huawei_measurements,

                                                  los_cond=los_cond,

                                                  input_average_clutter_height_m=input_average_clutter_height_m)

        tx_power_dbm = tx.transceiver_params.get('Transmit power')
        tx_power_dbw = tx_power_dbm - 30
        tx_antenna_efficiency = tx.transceiver_params.get('Antenna efficiency')
        tx_antenna_efficiency_db = 10 * log10(tx_antenna_efficiency)
        rx_antenna_efficiency = rx.transceiver_params.get('Antenna efficiency')
        rx_antenna_efficiency_db = 10 * log10(rx_antenna_efficiency)
        tx_antenna_gain_db = None
        rx_antenna_gain_db = None
        if antenna_gain_model is None or antenna_gain_model == 'input':
            # Take antenna gain from input
            tx_antenna_gain_db = tx.transceiver_params.get('Antenna gain')
            rx_antenna_gain_db = rx.transceiver_params.get('Antenna gain')
        elif antenna_gain_model == 'kumar salvia':
            # Taken from Vijay Kumar Salvia "Antenna and wave propagation" (Ed. Laxmi, 2007)
            tx_antenna_gain_db = get_antenna_gain_db(
                number_of_antennas=tx.transceiver_params.get('Number of antennas'))
            rx_antenna_gain_db = get_antenna_gain_db(number_of_antennas=rx.transceiver_params.get('Number of antennas'))
        else:
            exit('Invalid antenna_gain_model when computing the SNR according to the 3GPP model')

        prx_db = (tx_power_dbw + tx_antenna_efficiency_db + rx_antenna_efficiency_db + tx_antenna_gain_db +
                  rx_antenna_gain_db - path_loss_db)
        return prx_db

    def get_3gpp_prx_lin(self, tx, rx, carrier_frequency_ghz: float, tx_rx_distance_m: float,
                        apply_fading: bool, bandwidth_hz: float, clutter_density: float,
                        input_shadowing_sample_index: int,  use_huawei_measurements: bool,
                        input_average_clutter_height_m: float, los_cond: str,
                        antenna_gain_model: str = None):
        """

        Args:
          tx: UE or BS
          rx: BS or UE
          carrier_frequency_ghz: float: frequency of the carrier in GHz
          tx_rx_distance_m: float: distance of the transmitter and the receiver in meters
                    apply_fading: bool: whether to apply fading
          apply_fading: bool: whether to apply fading
          bandwidth_hz: float: bandwidth of the transmitter in GHz
          clutter_density: float: density of obstacles within the reference environment
          input_shadowing_sample_index: int: index of the shadowing sample in the array of values
          use_huawei_measurements: bool: True if channel model experimentally derived by Huawei measurements;
                                         False if 3GPP TR 38.901 channel model
          input_average_clutter_height_m: float: average clutter height in meters
          los_cond: str: Line of Sight or Non LoS condition
          antenna_gain_model: str: model of the antenna gain (if specified as input, Default value = None)

        Returns:
            received power in [W] according to the 3GPP model

        """
        prx_db = self.get_3gpp_prx_db(tx, rx, carrier_frequency_ghz, tx_rx_distance_m,
                        apply_fading, bandwidth_hz, clutter_density,
                        input_shadowing_sample_index,  use_huawei_measurements,
                        input_average_clutter_height_m, los_cond,
                        antenna_gain_model)
        prx_lin = 10 ** (prx_db / 10)

        return prx_lin

    def get_snr_db(self, ue: Ue, bs: BS, carrier_frequency_hz: float, bandwidth_hz: float, tx_rx_distance_m: float,
                   link_direction: str, apply_fading: bool):
        """

        Args:
          ue: Ue (class)
          bs: BS (class)
          carrier_frequency_hz: float: carrier frequency in GHz
          bandwidth_hz: float: bandwidth of the transmitter in GHz
          tx_rx_distance_m: float: distance of the transmitter and the receiver in meters
          link_direction: str: direction of the communication: "Uplink" or "Downlink"
          apply_fading: bool: whether to apply fading

        Returns:
            SNR in dB

        """
        frequency_intervals_list_hz, molecular_absorption_coefficient_list = (
            self.get_molecular_absorption_coefficients(carrier_frequency_hz=carrier_frequency_hz,
                                                       bandwidth_hz=bandwidth_hz))
        received_power_dbw = self.get_received_power_dbw(
            ue=ue, bs=bs, bandwidth_hz=bandwidth_hz, tx_rx_distance_m=tx_rx_distance_m, link_direction=link_direction,
            apply_fading=apply_fading, frequency_intervals_list_hz=frequency_intervals_list_hz,
            molecular_absorption_coefficient_list=molecular_absorption_coefficient_list)
        noise_power_dbw = self.get_molecular_noise_power_dbw(
            frequency_intervals_list_hz=frequency_intervals_list_hz,
            molecular_absorption_coefficient_list=molecular_absorption_coefficient_list,
            tx_rx_distance_m=tx_rx_distance_m)
        snr_db = received_power_dbw - noise_power_dbw
        return snr_db

    def get_sinr_db(self, useful_ue: Ue, interfering_ues: List[Ue], bs: BS, carrier_frequency_hz: float,
                    bandwidth_hz: float, useful_tx_rx_distance_m: float, interfering_tx_rx_distance_m_list: List[float],
                    link_direction: str, apply_fading: bool):
        """

        Args:
          useful_ue: Ue (class)
          interfering_ues: List[Ue]: list of UEs interfering with the reference UE
          bs: BS (class)
          carrier_frequency_hz: float: carrier frequency, in Hz
          bandwidth_hz: float: bandwidth in Hz
          useful_tx_rx_distance_m: float: useful transmission distance in meters
          interfering_tx_rx_distance_m_list: List[float]: distance between interfering transmitter and receiver in meters
          link_direction: str: direction of transmission: "Uplink" or "Downlink"
          apply_fading: bool: whether to apply fading or not

        Returns:
            SINR in dB

        """
        # Compute the frequency channels and corresponding molecular absorption coefficient
        frequency_intervals_list_hz, molecular_absorption_coefficient_list = (
            self.get_molecular_absorption_coefficients(carrier_frequency_hz=carrier_frequency_hz,
                                                       bandwidth_hz=bandwidth_hz))
        # Compute the useful received power
        useful_received_power_dbw = self.get_received_power_dbw(
            ue=useful_ue, bs=bs, bandwidth_hz=bandwidth_hz, tx_rx_distance_m=useful_tx_rx_distance_m,
            link_direction=link_direction, apply_fading=apply_fading,
            frequency_intervals_list_hz=frequency_intervals_list_hz,
            molecular_absorption_coefficient_list=molecular_absorption_coefficient_list)

        # Compute the noise power
        noise_power_dbw = self.get_thermal_noise_power_dbw(bandwidth_hz=bandwidth_hz)

        # Compute the interfering power for each single interfering UEs
        interfering_received_power_dbw_list = list()
        for interfering_ue, interfering_tx_rx_distance_m in zip(interfering_ues, interfering_tx_rx_distance_m_list):
            interfering_received_power_dbw_list.append(self.get_received_power_dbw(
                ue=interfering_ue, bs=bs, bandwidth_hz=bandwidth_hz, tx_rx_distance_m=interfering_tx_rx_distance_m,
                link_direction=link_direction, apply_fading=apply_fading,
                frequency_intervals_list_hz=frequency_intervals_list_hz,
                molecular_absorption_coefficient_list=molecular_absorption_coefficient_list))

        interfering_received_power_w_list = [10 ** (interfering_received_power_dbw / 10)
                                             for interfering_received_power_dbw in interfering_received_power_dbw_list]
        interfering_received_power_dbw = 10 * log10(sum(interfering_received_power_w_list))

        sinr_db = useful_received_power_dbw - interfering_received_power_dbw - noise_power_dbw

        return sinr_db

    def get_received_power_dbw(self, ue: Ue, bs: BS, bandwidth_hz: float, tx_rx_distance_m: float, link_direction: str,
                               apply_fading: bool, frequency_intervals_list_hz: List[Tuple[float, float]],
                               molecular_absorption_coefficient_list: List[float], ):
        """

        Args:
          ue: Ue (class)
          bs: BS (class)
          bandwidth_hz: float: bandwidth in Hz
          tx_rx_distance_m: float: distance between transmitter and receiver in meters
          link_direction: str: direction of transmission: "Uplink" or "Downlink"
          apply_fading: bool: whether to apply fading or not
          frequency_intervals_list_hz: List[Tuple[float, float]]: list of frequency intervals at Hz
          molecular_absorption_coefficient_list: List[float]: list of molecular absorption coefficients

        Returns:
            received power in dBW

        """

        if link_direction == "uplink":
            tx = ue
            rx = bs
        elif link_direction == "downlink":
            tx = bs
            rx = ue
        else:
            sys.exit("The link direction is not supported")

        transmit_power_dbw = tx.transceiver_params.get('Transmit power') - 30
        bandwidth_dhz = 10 * log10(bandwidth_hz)
        transmit_psd_dbw = transmit_power_dbw - bandwidth_dhz
        transmit_gain_db = get_antenna_gain_db(
            number_of_antennas=tx.transceiver_params.get('Number of antennas'))
        tx_antenna_efficiency_db = tx.transceiver_params.get('Antenna efficiency')
        receiver_gain_db = get_antenna_gain_db(number_of_antennas=rx.transceiver_params.get('Number of antennas'))
        rx_antenna_efficiency_db = rx.transceiver_params.get('Antenna efficiency')
        path_loss_db = self.get_friis_path_loss_db(tx_rx_distance_m=tx_rx_distance_m)
        integral_over_bandwidth_db = self.compute_integral_over_bandwidth_db(
            frequency_intervals_list_hz=frequency_intervals_list_hz,
            molecular_absorption_coefficient_list=molecular_absorption_coefficient_list,
            tx_rx_distance_m=tx_rx_distance_m)
        if tx.get_los_condition(rx):
            nlos_loss_db = 0
        else:
            nlos_loss_db = self.params.get('channel').get('nlos_absorption_loss_db')
        if apply_fading:
            shadowing_sample_db = self.get_shadowing_sample_db()
            fading_sample_db = self.get_fading_sample_db()
        else:
            shadowing_sample_db = 0
            fading_sample_db = 0

        received_power_dbw = (transmit_psd_dbw + tx_antenna_efficiency_db + rx_antenna_efficiency_db +
                              transmit_gain_db + receiver_gain_db + integral_over_bandwidth_db - path_loss_db -
                              shadowing_sample_db - fading_sample_db - nlos_loss_db)

        return received_power_dbw

    def get_thermal_noise_power_dbw(self, input_noise_figure: float, bandwidth_hz: float):
        """

        Args:
          input_noise_figure: float: noise figure in dB of the receiver
          bandwidth_hz: float: bandwidth in Hz

        Returns:
            thermal noise power in dBW

        """
        thermal_noise_dbw_power = 10 * log10(ct.Boltzmann * self.noise_temperature_k * input_noise_figure * bandwidth_hz)

        return thermal_noise_dbw_power

    def is_received(self, snr_db: float):
        """

        Args:
          snr_db: float: measured SNR in dB at the receiver

        Returns:
            True, if a signal is received given the SNR in dB
            False, otherwise

        """
        if snr_db >= self.snr_threshold_db:
            is_received = True
        else:
            is_received = False

        return is_received

    def get_shadowing_std_dev_db(self):
        """
        Returns:
            shadowing standard deviation in dB
        """
        return self.shadowing_std_dev_db

    def get_fading_sample_db(self):
        """
        Returns: Rayleigh fading sample in dB
        """
        fading_sample = np.random.rayleigh(scale=self.fading_param)
        return 20 * log10(fading_sample)

    def get_fading_param(self):
        """
        Returns:
            fading parameter
        """
        return self.fading_param

    def set_shadowing_sample_db(self, input_n_shadowing_samples: int):
        """

        Args:
          input_n_shadowing_samples: int: number of shadowing samples

        """
        self.shadowing_samples_array_db = np.random.normal(loc=0, scale=self.shadowing_std_dev_db,
                                                           size=input_n_shadowing_samples)

    def get_shadowing_sample_db(self):
        """
        Returns:
            shadowing sample in dB
        """
        shadowing_normal_sample = np.random.normal(loc=0, scale=1, size=None)
        return shadowing_normal_sample * self.shadowing_std_dev_db
