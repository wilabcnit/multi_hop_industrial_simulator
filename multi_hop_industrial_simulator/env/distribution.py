import sys
import math
import random

import numpy as np
from pandas.core.frame import DataFrame
from scipy import constants

from multi_hop_industrial_simulator.network.ris import RIS
from numpy import ndarray
from multi_hop_industrial_simulator.network.bs import BS
from multi_hop_industrial_simulator.env.machine import Machine
from typing import List


class Distribution:
    """
        Distribute devices on the environment and compute LoS condition.
    """

    def __init__(self, ue_distribution_type: str, ris_distribution_type: str, machine_distribution_type: str,
                 scenario_df: DataFrame = None,
                 tot_number_of_ues: int = None, tot_number_of_ris: int = None):
        self.ue_distribution_type = ue_distribution_type
        self.ris_distribution_type = ris_distribution_type
        self.machine_distribution_type = machine_distribution_type
        self.scenario_df = scenario_df
        self.number_of_machines = len(self.scenario_df.index) - 2  # First row for bi.rex layout,
        # second row for BS coordinates
        self.number_of_ues = tot_number_of_ues
        self.number_of_ris = tot_number_of_ris

    # Create the list of machines by distributing it in the space according to the input scenario
    def distribute_machines(self, scenario_df: DataFrame = None):
        n_machines = sum(1 for idx in self.scenario_df.index if isinstance(idx, str) and idx.startswith("Machine"))
        self.number_of_machines = n_machines
        machine_array = np.empty(n_machines, dtype=object)
        machine_index = 0
        if self.machine_distribution_type != 'custom':
            for idx, (index, row) in enumerate(self.scenario_df.iterrows()):
                if str(index).startswith("Machine"):
                    machine_array[machine_index] = Machine(
                        x_center=row["X-center"],
                        y_center=row["Y-center"],
                        z_center=row["Height"],
                        machine_size=row["Machine-size"],
                        max_number_of_ues=row["Max-number-of-UEs"])
                    machine_index += 1
        else:
            sys.exit('Custom machine distribution type is not supported')

        return machine_array

    # Get the number of machines inside the environment
    def get_number_of_machines(self):
        return self.number_of_machines

    # Distribute the UEs according to their distribution type
    def distribute_ues(self, ue_array: ndarray, machine_array: ndarray, bs: BS, simulator_tick_duration: float,
                       factory_length: int = None, factory_width: int = None, factory_height: int = None):

        # Distribute UEs within the factory
        if self.ue_distribution_type == "Uniform":
            # UEs distributed uniformly within the factory
            for ue in ue_array:
                ue.set_coordinates(random.uniform(0, factory_length), random.uniform(0, factory_width),
                                   0)  # random.uniform(0, factory_height))
        elif self.ue_distribution_type == "Custom":
            # RT and NRT UEs ==> Within machines
            # CN UEs ==> On walls
            x, y, z = 0, 0, 0
            machine_index = 0
            first_tear = 0
            second_tear = 0
            machine_first = [1, 3, 5, 7]
            machine_second = [0, 2, 4, 6]

            for ue in ue_array:
                if (ue.traffic_type == 'traffic_rt' or ue.traffic_type == 'traffic_nrt' or
                        ue.traffic_type == 'traffic_fq'):

                    ###### Code for distributing half of the UEs close to the BS and half of the UEs far from the BS #########
                    if len(machine_array) <= len(ue_array):
                        machine_chosen = random.randint(0, len(machine_array) - 1)
                    else:
                        machine_chosen = random.randint(0, len(ue_array) - 1)

                    if machine_index == machine_chosen:
                        while machine_index == machine_chosen:
                            if len(machine_array) <= len(ue_array):
                                machine_chosen = random.randint(0, len(machine_array) - 1)
                            else:
                                machine_chosen = random.randint(0, len(ue_array) - 1)
                    if machine_chosen in machine_first:
                        first_tear += 1
                        ue.n_tear = 1
                    else:
                        second_tear += 1
                        ue.n_tear = 2
                    if second_tear > math.ceil(len(ue_array) / 2):
                        machine_chosen = random.choice(machine_first)
                        ue.n_tear = 1
                    elif first_tear > math.floor(len(ue_array) / 2):
                        machine_chosen = random.choice(machine_second)
                        ue.n_tear = 2

                    x = (random.random() * (machine_array[machine_chosen].x_max - machine_array[machine_chosen].x_min) +
                         machine_array[machine_chosen].x_min)
                    y = (random.random() * (machine_array[machine_chosen].y_max - machine_array[machine_chosen].y_min) +
                         machine_array[machine_chosen].y_min)
                    z = random.random() * machine_array[machine_chosen].z_center * 2

                    machine_index = machine_chosen
                    ue.id_machine = machine_chosen

                elif ue.traffic_type == 'traffic_cn':
                    panel_chosen = random.randint(1, 4)
                    if panel_chosen == 1:
                        x = random.random() * factory_length
                        y = 0

                    elif panel_chosen == 2:
                        x = random.random() * factory_length
                        y = factory_width

                    elif panel_chosen == 3:
                        y = random.random() * factory_width
                        x = 0

                    else:
                        y = random.random() * factory_width
                        x = factory_length

                    z = random.random() * factory_height
                else:
                    sys.exit('The UE traffic type is not recognized when distributing the UEs')

                # Set the coordinates of the UE
                ue.set_coordinates(x, y, z)
                # print(" UE ", ue.get_ue_id(), ": x = ", x, "; y = ", y, "; z = ", z)

                if machine_index == len(machine_array):
                    machine_index = 0

        elif self.ue_distribution_type == "Grid":
            # Distribution taken from the input file
            ue_index = 0
            for idx, (index, row) in enumerate(self.scenario_df.iterrows()):
                if ue_index < len(ue_array):
                    if str(index).startswith("UE"):
                        ue_array[ue_index].set_coordinates(x_input=row['X-center'],
                                                           y_input=row['Y-center'],
                                                           z_input=row['Height'])
                        ue_index += 1

        elif self.ue_distribution_type == "Custom":
            # RT and NRT UEs ==> Within machines
            # CN UEs ==> On walls
            x, y, z = 0, 0, 0
            machine_index = 0
            first_tear = 0
            second_tear = 0
            machine_first = [1, 3, 5, 7]
            machine_second = [0, 2, 4, 6]

            for ue in ue_array:
                if (ue.traffic_type == 'traffic_rt' or ue.traffic_type == 'traffic_nrt' or
                        ue.traffic_type == 'traffic_fq'):

                    """ Code for distributing half of the UEs close to the BS and half of the UEs far from the BS """
                    if len(machine_array) <= len(ue_array):
                        machine_chosen = random.randint(0, len(machine_array) - 1)
                    else:
                        machine_chosen = random.randint(0, len(ue_array) - 1)

                    if machine_index == machine_chosen:
                        while machine_index == machine_chosen:
                            if len(machine_array) <= len(ue_array):
                                machine_chosen = random.randint(0, len(machine_array) - 1)
                            else:
                                machine_chosen = random.randint(0, len(ue_array) - 1)
                    if machine_chosen in machine_first:
                        first_tear += 1
                        ue.n_tear = 1
                    else:
                        second_tear += 1
                        ue.n_tear = 2
                    if second_tear > math.ceil(len(ue_array) / 2):
                        machine_chosen = random.choice(machine_first)
                        ue.n_tear = 1
                    elif first_tear > math.floor(len(ue_array) / 2):
                        machine_chosen = random.choice(machine_second)
                        ue.n_tear = 2

        else:
            sys.exit('The UE distribution statistics is not recognized')

    # Distribute the BS inside the environment
    def distribute_bs(self, bs: BS):
        bs.set_coordinates(x_input=self.scenario_df.loc['BS', "X-center"],
                           y_input=self.scenario_df.loc['BS', "Y-center"],
                           z_input=self.scenario_df.loc['BS', "Height"])

    # Set the overall number of UEs within the environment
    def set_number_of_ues(self, input_n_ues: int):
        self.number_of_ues = input_n_ues

    # Get the overall number of UEs within the environment
    def get_number_of_ues(self):
        return self.number_of_ues
