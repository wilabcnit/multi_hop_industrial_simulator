import sys
from typing import List, Tuple

import numpy as np
from math import log10
from scipy import constants as ct

from timessim.utils.read_input_file import read_input_file
from timessim.utils.get_antenna_gain_db import get_antenna_gain_db
from timessim.network.ue import Ue
from timessim.network.bs import BS
from timessim.channel_models import THz_channel

# Function to compute the maximum propagation delay (in seconds) given the maximum transmission range
def compute_max_prop_delay_tx_range(ue: Ue, bs: BS, input_thz_channel: THz_channel,
                                    bandwidth_hz: float, carrier_frequency_ghz: float,
                                    snr_threshold_db: float, use_huawei_measurements: bool
                                    , enable_print: bool, antenna_gain_model: str = None):
    # inserire metodi thz_channel e eliminare transceiver_params
    tx_power_dbm = ue.transceiver_params.get('Transmit power')
    tx_power_dbw = tx_power_dbm - 30
    tx_antenna_efficiency = ue.transceiver_params.get('Antenna efficiency')
    tx_antenna_efficiency_db = 10 * log10(tx_antenna_efficiency)
    rx_antenna_efficiency = bs.transceiver_params.get('Antenna efficiency')
    rx_antenna_efficiency_db = 10 * log10(rx_antenna_efficiency)
    tx_antenna_gain_db = None
    rx_antenna_gain_db = None
    if antenna_gain_model is None or antenna_gain_model == 'input':
        # Take antenna gain from input
        tx_antenna_gain_db = ue.transceiver_params.get('Antenna gain')
        rx_antenna_gain_db = bs.transceiver_params.get('Antenna gain')
    elif antenna_gain_model == 'kumar salvia':
        # Taken from Vijay Kumar Salvia "Antenna and wave propagation" (Ed. Laxmi, 2007)
        tx_antenna_gain_db = get_antenna_gain_db(
            number_of_antennas=ue.transceiver_params.get('Number of antennas'))
        rx_antenna_gain_db = get_antenna_gain_db(number_of_antennas=bs.transceiver_params.get('Number of antennas'))
    else:
        exit('Invalid antenna_gain_model when computing the SNR according to the 3GPP model')
    rx_noise_figure_db = bs.transceiver_params.get('Noise figure')
    rx_noise_figure = 10 ** (rx_noise_figure_db / 10)
    noise_power_dbw = input_thz_channel.get_thermal_noise_power_dbw(input_noise_figure=rx_noise_figure,
                                                                    bandwidth_hz=bandwidth_hz)
    max_path_loss = (tx_power_dbw + tx_antenna_efficiency_db + rx_antenna_efficiency_db + tx_antenna_gain_db +
                     rx_antenna_gain_db - noise_power_dbw - snr_threshold_db)

    # higher distance PL

    if use_huawei_measurements:
        tx_range = 10 ** ((max_path_loss - 30.7 - 20.6 * log10(carrier_frequency_ghz)) / 22.8)
    else:
        tx_range = 10 ** ((max_path_loss - 31.84 - 19.00 * log10(carrier_frequency_ghz)) / 21.50)

    time_tx_range = tx_range / ct.speed_of_light
    if enable_print:
        print(f"Max tx range {tx_range} m")

    return time_tx_range
