import numpy as np
from scipy.special import erfc
import math


def check_collision(input_simulator_timing_structure: dict, input_t_start_rx: int, input_t_end_rx: int,
                    input_ue_id: int = None, input_tx: str = None, input_ue_id_rx: int = None,
                    ues_colliding: list = None):
    """
    Check whether a UE's transmission collides with other ongoing transmissions (DATA or ACK).

    Args:
        input_simulator_timing_structure (dict): Nested dictionary containing UE transmission timing information.
        input_t_start_rx (int): Start time of the current reception window.
        input_t_end_rx (int): End time of the current reception window.
        input_ue_id (int, optional): ID of the reference UE. Defaults to None.
        input_tx (str, optional): Identifier of the ACK transmitter (UE or BS). Used with ACK packets. Defaults to None.
        input_ue_id_rx (int, optional): ID of the UE from which the DATA packet is
        (for ACK TX not used, so set = reference UE_ID). Defaults to None.
        ues_colliding (list, optional): List to append tuples of colliding UEs and their time windows. Defaults to None.

    Returns:
        list: Updated list of tuples (ue_id, t_start_rx, t_end_rx) for all UEs colliding with the input UE.
    """

    # Loop over the timing structure to check collisions
    for ue_key_ext, ue_data_ext in input_simulator_timing_structure.items():
        if ue_key_ext == f'UE_{input_ue_id}':
            # Check collisions with DATA
            for ue_key_int, ue_value_int in ue_data_ext['DATA_RX'].items():
                if ue_key_int != f'UE_{input_ue_id_rx}':
                    min_index = np.argmin(ue_value_int[:, 1])
                    min_row = ue_value_int[min_index, :]
                    t_start_rx, t_end_rx = min_row[0], min_row[1]
                    if t_start_rx is not None and t_end_rx is not None:
                        if (input_t_start_rx <= t_start_rx < input_t_end_rx or
                                input_t_start_rx < t_end_rx <= input_t_end_rx or t_start_rx < input_t_end_rx < t_end_rx):
                            if ue_key_int not in ues_colliding:
                                ues_colliding.append((ue_key_int, t_start_rx, t_end_rx))

            # Check collisions with ACK
            for ue_key_int, ue_value_int in ue_data_ext['ACK_RX'].items():
                if ue_key_int != input_tx:
                    # I have to exclude the UE/BS from which the current UE has received an ACK
                    min_index = np.argmin(ue_value_int[:, 1])
                    min_row = ue_value_int[min_index, :]
                    t_start_rx, t_end_rx = min_row[0], min_row[1]
                    if t_start_rx is not None and t_end_rx is not None:
                        if (input_t_start_rx <= t_start_rx < input_t_end_rx or
                                input_t_start_rx < t_end_rx <= input_t_end_rx or t_start_rx < input_t_end_rx < t_end_rx):
                            if ue_key_int not in ues_colliding:
                                ues_colliding.append((ue_key_int, t_start_rx, t_end_rx))

    return ues_colliding


def check_collision_aodv(input_simulator_timing_structure: dict, input_t_start_rx: int, input_t_end_rx: int,
                         input_ue_id: int = None, input_tx: str = None, input_ue_id_rx: int = None,
                         ues_colliding: list = None):
    """
    Check whether a UE's transmission collides with others (DATA, ACK, RREQ, RREPLY) in AODV mode.

    Args:
        input_simulator_timing_structure (dict): Dictionary of UE/BS transmissions.
        input_t_start_rx (int): Start time of the current RX window.
        input_t_end_rx (int): End time of the current RX window.
        input_ue_id (int, optional): ID of the UE where there is the collision check. Defaults to None.
        input_tx (str, optional): Identifier of the ACK/RREPLY transmitter (UE or BS). Defaults to None.
        input_ue_id_rx (int, optional): ID of the UE from which the DATA / RREQ packet is addressed. Defaults to None.
        ues_colliding (list, optional): List to append colliding transmissions. Defaults to None.

    Returns:
        list: Updated list of colliding UEs and their reception time intervals.
    """

    # Loop over the timing structure to check collisions
    for ue_key_ext, ue_data_ext in input_simulator_timing_structure.items():
        if ue_key_ext == f'UE_{input_ue_id}':
            # Check collisions with DATA
            for ue_key_int, ue_value_int in ue_data_ext['DATA_RX'].items():
                if ue_key_int != f'UE_{input_ue_id_rx}':
                    min_index = np.argmin(ue_value_int[:, 1])
                    min_row = ue_value_int[min_index, :]
                    t_start_rx, t_end_rx = min_row[0], min_row[1]
                    if t_start_rx is not None and t_end_rx is not None:
                        if (input_t_start_rx <= t_start_rx < input_t_end_rx or
                                input_t_start_rx < t_end_rx <= input_t_end_rx or t_start_rx < input_t_end_rx < t_end_rx):
                            if ue_key_int not in ues_colliding:
                                ues_colliding.append((ue_key_int, t_start_rx, t_end_rx))

            # Check collisions with ACK
            for ue_key_int, ue_value_int in ue_data_ext['ACK_RX'].items():
                if ue_key_int != input_tx:
                    # Need to exclude the UE/BS from which the current UE has received an ACK
                    min_index = np.argmin(ue_value_int[:, 1])
                    min_row = ue_value_int[min_index, :]
                    t_start_rx, t_end_rx = min_row[0], min_row[1]
                    if t_start_rx is not None and t_end_rx is not None:
                        if (input_t_start_rx <= t_start_rx < input_t_end_rx or
                                input_t_start_rx < t_end_rx <= input_t_end_rx or t_start_rx < input_t_end_rx < t_end_rx):
                            mac_success = False
                            if ue_key_int not in ues_colliding:
                                ues_colliding.append((ue_key_int, t_start_rx, t_end_rx))

            # Check collisions with RREQ
            for ue_key_int, ue_value_int in ue_data_ext['RREQ'].items():
                if ue_key_int != f'UE_{input_ue_id_rx}':
                    # I have to exclude the UE/BS from which the current UE has received an ACK
                    min_index = np.argmin(ue_value_int[:, 1])
                    min_row = ue_value_int[min_index, :]
                    t_start_rx, t_end_rx = min_row[0], min_row[1]
                    if t_start_rx is not None and t_end_rx is not None:
                        if (input_t_start_rx <= t_start_rx < input_t_end_rx or
                                input_t_start_rx < t_end_rx <= input_t_end_rx or t_start_rx < input_t_end_rx < t_end_rx):
                            if ue_key_int not in ues_colliding:
                                ues_colliding.append((ue_key_int, t_start_rx, t_end_rx))

            # check collision with RREPLY
            for ue_key_int, ue_value_int in ue_data_ext['RREPLY'].items():
                if ue_key_int != input_tx or input_tx is None:  # Need to find other RREPLY colliding with the current one and sent from other UEs
                    # I have to exclude the UE/BS from which the current UE has received an ACK
                    min_index = np.argmin(ue_value_int[:, 1])
                    min_row = ue_value_int[min_index, :]
                    t_start_rx, t_end_rx = min_row[0], min_row[1]
                    if t_start_rx is not None and t_end_rx is not None:
                        if (input_t_start_rx <= t_start_rx < input_t_end_rx or
                                input_t_start_rx < t_end_rx <= input_t_end_rx or t_start_rx < input_t_end_rx < t_end_rx):
                            if ue_key_int not in ues_colliding:
                                ues_colliding.append((ue_key_int, t_start_rx, t_end_rx))

    return ues_colliding


def check_collision_bs(input_simulator_timing_structure: dict, input_t_start_rx: int, input_t_end_rx: int,
                       input_ue_id: int = None, ues_colliding: list = None):
    """
    Check whether a UE's transmission collides at the Base Station (BS) with other transmissions.

    Args:
        input_simulator_timing_structure (dict): BS timing structure containing DATA and ACK RX times.
        input_t_start_rx (int): Start time of the current reception window.
        input_t_end_rx (int): End time of the current reception window.
        input_ue_id (int, optional): ID of the transmitting UE. Defaults to None.
        ues_colliding (list, optional): List to append collisions. Defaults to None.

    Returns:
        list: Updated list of colliding UEs with their time intervals.
    """

    for ue_key_int, ue_value_int in input_simulator_timing_structure['BS']['DATA_RX'].items():
        if ue_key_int != f'UE_{input_ue_id}':
            min_index = np.argmin(ue_value_int[:, 1])
            min_row = ue_value_int[min_index, :]
            t_start_rx, t_end_rx = min_row[0], min_row[1]
            if (input_t_start_rx <= t_start_rx < input_t_end_rx or
                    input_t_start_rx < t_end_rx <= input_t_end_rx or t_start_rx < input_t_end_rx < t_end_rx):

                if ue_key_int not in ues_colliding:
                    ues_colliding.append((ue_key_int, t_start_rx, t_end_rx))

    # Check collisions with ACK
    for ue_key_int, ue_value_int in input_simulator_timing_structure['BS']['ACK_RX'].items():
        if ue_key_int != f'UE_{input_ue_id}':
            min_index = np.argmin(ue_value_int[:, 1])
            min_row = ue_value_int[min_index, :]
            t_start_rx, t_end_rx = min_row[0], min_row[1]
            if (input_t_start_rx <= t_start_rx < input_t_end_rx or
                    input_t_start_rx < t_end_rx <= input_t_end_rx or t_start_rx < input_t_end_rx <= t_end_rx):
                if ue_key_int not in ues_colliding:
                    ues_colliding.append((ue_key_int, t_start_rx, t_end_rx))

    return ues_colliding


def check_collision_bs_aodv(input_simulator_timing_structure: dict, input_t_start_rx: int, input_t_end_rx: int,
                            input_ue_id: int = None, ues_colliding: list = None):
    """
    Check whether a UE's transmission collides at the BS (including RREQ and RREPLY for AODV).

    Args:
        input_simulator_timing_structure (dict): BS timing structure with DATA, ACK, and RREQ receptions.
        input_t_start_rx (int): Start time of the RX window.
        input_t_end_rx (int): End time of the RX window.
        input_ue_id (int, optional): ID of the transmitting UE. Defaults to None.
        ues_colliding (list, optional): List to append collisions. Defaults to None.

    Returns:
        list: Updated list of colliding UEs and their transmission windows.
    """

    for ue_key_int, ue_value_int in input_simulator_timing_structure['BS']['DATA_RX'].items():
        if ue_key_int != f'UE_{input_ue_id}':
            min_index = np.argmin(ue_value_int[:, 1])
            min_row = ue_value_int[min_index, :]
            t_start_rx, t_end_rx = min_row[0], min_row[1]
            if (input_t_start_rx <= t_start_rx < input_t_end_rx or
                    input_t_start_rx < t_end_rx <= input_t_end_rx or t_start_rx < input_t_end_rx < t_end_rx):

                if ue_key_int not in ues_colliding:
                    ues_colliding.append((ue_key_int, t_start_rx, t_end_rx))

    # Check collisions with ACK
    for ue_key_int, ue_value_int in input_simulator_timing_structure['BS']['ACK_RX'].items():
        if ue_key_int != f'UE_{input_ue_id}':
            min_index = np.argmin(ue_value_int[:, 1])
            min_row = ue_value_int[min_index, :]
            t_start_rx, t_end_rx = min_row[0], min_row[1]
            if (input_t_start_rx <= t_start_rx < input_t_end_rx or
                    input_t_start_rx < t_end_rx <= input_t_end_rx or t_start_rx < input_t_end_rx <= t_end_rx):
                if ue_key_int not in ues_colliding:
                    ues_colliding.append((ue_key_int, t_start_rx, t_end_rx))

    # Check collisions with RREQ
    for ue_key_int, ue_value_int in input_simulator_timing_structure['BS']['RREQ'].items():
        if ue_key_int != f'UE_{input_ue_id}':
            min_index = np.argmin(ue_value_int[:, 1])
            min_row = ue_value_int[min_index, :]
            t_start_rx, t_end_rx = min_row[0], min_row[1]
            if (input_t_start_rx <= t_start_rx < input_t_end_rx or
                    input_t_start_rx < t_end_rx <= input_t_end_rx or t_start_rx < input_t_end_rx <= t_end_rx):
                if ue_key_int not in ues_colliding:
                    ues_colliding.append((ue_key_int, t_start_rx, t_end_rx))

    return ues_colliding


def check_success(input_snr_db: float = None, input_sir_db: float = None, input_modulation_order: int = None,
                  input_payload_bytes: int = None):
    """
    Determine whether a transmission is successful based on SNR, SIR, modulation order, and payload size.

    Args:
        input_snr_db (float, optional): Signal-to-noise ratio in dB. Defaults to None.
        input_sir_db (float, optional): Signal-to-interference ratio in dB. Defaults to None.
        input_modulation_order (int, optional): Modulation order (e.g., 4 for 16-QASK). Defaults to None.
        input_payload_bytes (int, optional): Payload size in bytes. Defaults to None.

    Returns:
        bool: True if the transmission is successful (p_success ≥ 0.9), False otherwise.
    """

    max_snr_linear = 10 ** 308  # Limit close to max float
    if (input_snr_db / 10) > 308:
        snr_linear = max_snr_linear
    else:
        snr_linear = 10 ** (input_snr_db / 10)

    # Compute the modulation order and payload in bits
    L = math.sqrt(input_modulation_order)
    input_payload_bits = input_payload_bytes * 8

    # if there are no interferers -> use the BER formula for SIR
    if input_sir_db is None:
        ber = ((L - 1) / (L * np.log2(L))) * erfc(math.sqrt(snr_linear * np.log2(L) / (L - 1) ** 2))

    else:
        if (input_sir_db / 10) > 308:
            ber = 0
        else:
            sir_linear = 10 ** (input_sir_db / 10)
            ber = ((L - 1) / (L * np.log2(L))) * erfc(math.sqrt(1 / (((L - 1) ** 2 / (snr_linear * np.log2(L))) +
                                                                     (2 * (L ** 2 - 1) / (3 * sir_linear)))))

    # Compute the probability of success
    p_succ = (1 - ber) ** input_payload_bits

    if p_succ >= 0.9:
        success = True
    else:
        success = False

    return success


def compute_p_success(input_snr_db: float = None, input_modulation_order: int = None,
                      input_payload_bytes: int = None):
    """
    Compute the probability of a successful transmission based on SNR, modulation order, and payload size.

    Args:
        input_snr_db (float, optional): Signal-to-noise ratio in dB. Defaults to None.
        input_modulation_order (int, optional): Modulation order (e.g., 4 for QPSK). Defaults to None.
        input_payload_bytes (int, optional): Payload size in bytes. Defaults to None.

    Returns:
        float: Probability of successful transmission (value between 0 and 1).
    """

    max_snr_linear = 10 ** 308  # Limit close to max float
    if (input_snr_db/10) > 308:
        snr_linear = max_snr_linear
    else:
        snr_linear = 10 ** (input_snr_db / 10)

    # Compute the modulation order and payload in bits
    L = math.sqrt(input_modulation_order)
    input_payload_bits = input_payload_bytes * 8
    # Compute the bit error rate (BER)
    ber = ((L - 1) / (L * np.log2(L))) * erfc(math.sqrt(snr_linear * np.log2(L) / (L-1)**2))
    # Compute the probability of success
    p_succ = (1 - ber) ** input_payload_bits

    return p_succ



