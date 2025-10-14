"""
    Utility scripts to create the BS object with the correct type of traffic

    @Note: simulator_tick_duration must be in seconds
"""

from multi_hop_industrial_simulator.network.bs import BS
import sys
import math

# Function to create the BS with the correct type of traffic
def instantiate_bs(input_params_dict: dict, simulator_tick_duration: float, starting_state: str, bit_rate_gbits: float):
    """

    Args:
      input_params_dict: dict: dictionary of input parameters
      simulator_tick_duration: float: duration of simulation in ticks
      starting_state: str: starting state of BS
      bit_rate_gbits: float: bit rate value in Gbit/s

    Returns:
        BS object

    """
    bs = None

    for key, values in input_params_dict.items():
        if key.startswith('traffic'):
            if key == "traffic_nrt" and values["percentage_of_ue"] != 0:
                bs = BS(params=input_params_dict, traffic_type=key, starting_state=starting_state,
                        input_full_queue=values.get("full_queue"))
                collection_duration = values["collection_on_duration"] + values["collection_standby_duration"]
                if "machine_optimization" in values.keys():  # during optimization, the node will RX data
                    collection_duration_tick = math.ceil(collection_duration/simulator_tick_duration)
                    bs.set_t_generation_optimization(collection_duration_tick)
                    scale = float(values.get('machine_optimization').get('scale'))  # in s
                    if 0.01 <= scale <= 1:
                        bs.set_exp_distribution(scale_value=scale, tick_simulator=simulator_tick_duration)
                    else:
                        sys.exit("The mean you chose for NRT for machine optimization is wrong!")
                    payload = values.get('machine_optimization').get('payload')

                    if 100 <= payload <= 300:
                        bs.set_new_packet_size(payload)
                    else:
                        sys.exit("Check the payload of NRT high and low, something is wrong!")
            else:
                bs = BS(params=input_params_dict, traffic_type=key, starting_state=starting_state,
                        input_full_queue=values.get("full_queue"))

        # Set initial bit rate of the BS
        bs.set_bit_rate_gbits(input_bit_rate_gbits=bit_rate_gbits)

    return bs
