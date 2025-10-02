import math
import numpy as np
from scipy import stats


# Function to compute the 95% confidence interval for the metrics
def compute_conf_int(output_dict, n_simulated_ues, n_simulations):

    conf_int_results = {key: np.zeros((n_simulated_ues, 2)) for key in
                        output_dict.keys()}
    metrics = ["p_mac", "s", "l", "e", "tx_pck", "rx_pck"]
    for metric in metrics:
        for n_ue in range(len(output_dict[metric])):
            sample_mean = np.mean(output_dict[metric][n_ue])

            # Standard error of the mean
            sem = stats.sem(output_dict[metric][n_ue])

            confidence = 0.95

            # Calculate the confidence interval
            conf_int_results[metric][n_ue] = stats.t.interval(confidence, len(output_dict[metric][n_ue]) - 1, loc=sample_mean, scale=sem)
    print(conf_int_results)

    return conf_int_results
