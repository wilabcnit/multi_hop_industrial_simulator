"""
    Utility scripts to create the UE numpy array with the correct type of traffic

    @Note: simulator_tick_duration must be in seconds
"""
import numpy as np

from multi_hop_industrial_simulator.network.ue import Ue
import sys
import math

# Function to create the UEs with the correct type of traffic
def instantiate_ues(input_params_dict: dict, tot_number_of_ues: int, starting_state: str, t_state_tick: int,
                    simulator_tick_duration: float, bit_rate_gbits: float, max_n_retx_per_packet: int):
    """

    Args:
      input_params_dict: dict: 
      tot_number_of_ues: int: 
      starting_state: str: 
      t_state_tick: int: 
      simulator_tick_duration: float: 
      bit_rate_gbits: float: 
      max_n_retx_per_packet: int: 

    Returns:

    """

    percentage_of_ue_counter = 0
    ue_id = 0
    ue_array = np.empty(tot_number_of_ues, dtype=object)

    # Search all the types of input traffic and assign them to UEs
    for key, values in input_params_dict.items():
        if key.startswith('traffic'):  # Pick-up input traffic parameters
            current_percentage_of_ue = values["percentage_of_ue"]

            current_number_of_nodes = round(current_percentage_of_ue / 100 * tot_number_of_ues)
            percentage_of_ue_counter += current_percentage_of_ue  # Check that the percentage are correctly set in input

            # Instantiate UEs that will generate the current traffic type
            for i in range(current_number_of_nodes):
                # print("UE ", ue_id, " traffic information: ")
                current_ue = Ue(params=input_params_dict, ue_id=ue_id, traffic_type=key, starting_state=starting_state,
                                t_state_tick=t_state_tick, input_full_queue=values.get("full_queue"))
                current_ue.set_bit_rate_gbits(input_bit_rate_gbits=bit_rate_gbits)

                # Set information for all the UEs depending on their traffic type
                """
                    RT UE
                """
                if key == "traffic_rt":
                    # First check if the application is isochronous or not, and then check if the periodicity is correct
                    if "isochronous" in values.keys() and "period" in values.keys():
                        isochronous = values.get('isochronous')
                        period = values.get('period')  # in s
                        time_periodicity_ticks = math.ceil(period / simulator_tick_duration)  # in tick
                        if isochronous == 0:  # non - isochronous
                            if 0.002 < period <= 0.02:
                                current_ue.set_time_periodicity(time_periodicity_ticks=time_periodicity_ticks)
                                # print("NON Isochronous Application, period: ", period)
                            else:
                                sys.exit("The periodicity you chose for NON-isochronous application is wrong!")
                        else:
                            if 0.0001 <= period <= 0.002:
                                current_ue.set_time_periodicity(time_periodicity_ticks=time_periodicity_ticks)
                                # print("Isochronous Application, period: ", period)
                            else:
                                sys.exit("The periodicity you chose for Isochronous application is wrong!")
                    # Then check if the sensor is low-end or high-end and check the data size
                    if "sensor_type" in values.keys() and "payload" in values.keys():
                        sensor_type = values.get('sensor_type')
                        payload = values.get('payload')
                        if sensor_type == 0:  # low-end sensor
                            if 30 <= payload <= 100 or values.get('full_queue'):
                                # print("Low-end sensor, payload: ", payload)
                                current_ue.set_new_packet_size(payload)
                                # current_ue.set_new_packet_data_to_be_sent(bytes_per_packet)
                            else:
                                sys.exit("The payload you chose for a low-end sensor is wrong!")
                        else:
                            if 50 <= payload <= 1500:  # high-end sensor
                                # print("High-end sensor, payload: ", payload)
                                current_ue.set_new_packet_size(payload)
                                # current_ue.set_new_packet_data_to_be_sent(bytes_per_packet)
                            else:
                                sys.exit("The payload you chose for a high-end sensor is wrong!")
                """
                    CAMERA UE
                """
                if key == "traffic_cn":  # Camera Node
                    # First check if the camera is slow-motion or standard, and then check if the periodicity is correct
                    if "camera_type" in values.keys() and "period" in values.keys():
                        camera_type = values.get('camera_type')
                        period = float(values.get('period'))  # in s
                        time_periodicity_ticks = math.ceil(period / simulator_tick_duration)  # in tick
                        if camera_type == 0:  # slow-motion
                            if period == 0.0083:
                                current_ue.set_time_periodicity(time_periodicity_ticks=time_periodicity_ticks)
                                # print("Slow-motion camera, period: ", period)
                            else:
                                sys.exit("The periodicity you chose for slow-motion camera is wrong!")
                        else:
                            if period == 0.0333:
                                current_ue.set_time_periodicity(time_periodicity_ticks=time_periodicity_ticks)
                                # print("Standard camera, period: ", period)
                            else:
                                sys.exit("The periodicity you chose standard camera is wrong!")
                    # Then check if the camera is low-end, medium-end or high-end and then check the data size
                    if "resolution_type" in values.keys() and "payload" in values.keys():
                        resolution_type = values.get('resolution_type')
                        payload = values.get('payload')
                        if resolution_type == 0:  # low-end camera
                            if 2000000 <= payload < 12000000 or values.get('full_queue'):
                                # print("Low-end resolution, payload: ", payload)
                                current_ue.set_new_packet_size(payload)
                            else:
                                sys.exit("The payload you chose for a low-end resolution is wrong!")
                        elif resolution_type == 1:  # medium camera
                            if 12000000 <= payload < 45000000 or values.get('full_queue'):
                                # print("Medium resolution, payload: ", payload)
                                current_ue.set_new_packet_size(payload)
                            else:
                                sys.exit("The payload you chose for a medium resolution is wrong!")
                        elif resolution_type == 2:  # high-end camera
                            if 45000000 <= payload < 200000000 or values.get('full_queue'):
                                # print("High-end resolution, payload: ", payload)
                                current_ue.set_new_packet_size(payload)
                            else:
                                sys.exit("The payload you chose for a high-end resolution is wrong!")

                """
                    NRT UE
                """
                if key == "traffic_nrt":  # NRT Node
                    # NRT Node can behave both as NRT for "data collection" but also NRT for "machine optimization"
                    # -> instantiate parameters for data collection, since in optimization the node receives data
                    current_ue.status_nrt_change = 0  # instant of time when a nrt node has to change its status
                    current_ue.collection_on = values.get("collection_on")
                    current_ue.collection_standby = values.get("collection_standby")
                    current_ue.optimization = values.get("optimization")
                    if "data_collection" in values.keys():
                        period = float(values.get('data_collection').get('period'))  # in s
                        time_periodicity_ticks = math.ceil(period / simulator_tick_duration)  # in tick
                        if 0.001 <= period <= 0.1:
                            current_ue.set_time_periodicity(time_periodicity_ticks=time_periodicity_ticks)
                            # print("NRT for data Collection, period: ", period)
                        else:
                            sys.exit("The periodicity you chose for NRT for data Collection is wrong!")

                        payload_on = values.get('data_collection').get('payload_on')
                        payload_standby = values.get('data_collection').get('payload_standby')
                        if 1500 >= payload_on >= 100 > payload_standby >= 30 or values.get('full_queue'):
                            # print("NRT ON, payload: ", payload_on)
                            #  print("NRT STANDBY, payload: ", payload_standby)
                            current_ue.set_new_packet_size_on(payload_on)
                            current_ue.set_new_packet_size_standby(payload_standby)
                        else:
                            sys.exit("Check the payloads of NRT ON and Standby, something is wrong!")
                """
                    FQ UE (UEs have always a new data to be transmitted)
                """
                if key == "traffic_fq":
                    time_periodicity_ticks = (
                            input_params_dict.get('simulation').get('tot_simulation_time_s')
                            / simulator_tick_duration) + 10  # in tick
                    current_ue.set_time_periodicity(time_periodicity_ticks=time_periodicity_ticks)
                    payload = values.get('payload')
                    overhead = values.get('overhead_data')
                    data_size = payload + overhead
                    current_ue.set_new_packet_size(data_size)

                # Set the data duration
                current_ue.set_data_duration_s(round((current_ue.packet.packet_size * 8 * 1e-9) /
                                                     current_ue.get_bit_rate_gbits(), 11))
                current_ue.set_data_duration_tick(round(current_ue.data_duration_s / simulator_tick_duration))
                current_ue.set_max_n_retx_per_packet(input_max_n_retx_per_packet=max_n_retx_per_packet)

                # Append the UE in the list of UEs
                ue_array[ue_id] = current_ue
                ue_id += 1

    return ue_array
