import sys
import math
import random

import numpy as np
from pandas.core.frame import DataFrame

from numpy import ndarray
from multi_hop_industrial_simulator.network.bs import BS
from multi_hop_industrial_simulator.env.machine import Machine

'''
Distribution class

Distribute devices in the environment and compute LoS/NLoS condition.
'''

class Distribution:

    def __init__(self, ue_distribution_type: str, machine_distribution_type: str,
                 scenario_df: DataFrame = None,
                 tot_number_of_ues: int = None):
        self.ue_distribution_type = ue_distribution_type
        self.machine_distribution_type = machine_distribution_type
        self.scenario_df = scenario_df
        self.number_of_machines = len(self.scenario_df.index) - 2  # First row for bi.rex layout,
        # second row for BS coordinates
        self.number_of_ues = tot_number_of_ues

    def distribute_machines(self, scenario_df: DataFrame = None):
        """

        Args:
          scenario_df: DataFrame:  Select one reference scenario to be deployed(Default value = None)

        Returns:
            list of machines by distributing it in the space according to the input scenario

        """
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

    def get_number_of_machines(self):
        """
        Returns: number of machines inside the environment
        """
        return self.number_of_machines

    def distribute_ues(self, ue_array: ndarray, machine_array: ndarray, bs: BS, simulator_tick_duration: float,
                       factory_length: float = None, factory_width: float = None, factory_height: float = None):
        """

        Args:
          ue_array: ndarray: array of UEs
          machine_array: ndarray: array of machines
          bs: BS: class
          simulator_tick_duration: float: duration of simulation ticks
          factory_length: int: length of the factory in meters (Default value = None)
          factory_width: int: width of the factory in meters (Default value = None)
          factory_height: int: width of the factory in meters (Default value = None)

        Distribute the UEs according to their distribution type

        """

        # Distribute UEs within the factory
        if self.ue_distribution_type == "Uniform":
            # UEs distributed uniformly within the factory
            for ue in ue_array:
                ue.set_coordinates(random.uniform(0, factory_length), random.uniform(0, factory_width), 0)

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

        else:
            sys.exit('The UE distribution statistics is not recognized')

    def distribute_bs(self, bs: BS):
        """

        Args:
          bs: BS: class

        Distribute the BS inside the environment

        """
        bs.set_coordinates(x_input=self.scenario_df.loc['BS', "X-center"],
                           y_input=self.scenario_df.loc['BS', "Y-center"],
                           z_input=self.scenario_df.loc['BS', "Height"])

    def set_number_of_ues(self, input_n_ues: int):
        """

        Args:
          input_n_ues: int: input number of UEs

        Returns:
            overall number of UEs within the environment

        """
        self.number_of_ues = input_n_ues

    def get_number_of_ues(self):
        """
        Returns: overall number of UEs within the environment
        """
        return self.number_of_ues
