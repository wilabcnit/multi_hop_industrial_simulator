import numpy as np
from scipy.special import erfcinv

# Function to compute the SNR threshold in dB given the payload size in bytes and the target success probability at the physical layer
def compute_snr_threshold_db(input_payload_bytes: int, input_p_succ_phy: float):

    input_payload_bits = input_payload_bytes * 8
    L = 8  # If 4- QAM
    snr_th = (L - 1) ** 2 / np.log2(L) * (erfcinv((L / (L - 1)) * np.log2(L) * (1 - input_p_succ_phy ** (1 / input_payload_bits)))) ** 2
    snr_th_db = 10 * np.log10(snr_th)

    return snr_th_db
