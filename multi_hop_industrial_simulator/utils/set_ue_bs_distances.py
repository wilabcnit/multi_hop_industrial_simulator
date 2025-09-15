import math
from typing import List

from multi_hop_industrial_simulator.network.bs import BS
from multi_hop_industrial_simulator.network.ue import Ue

# Function to calculate and set the distance between each UE and the BS
def set_ue_bs_distances(ue_list: List[Ue], bs: BS):

    for index, ue in enumerate(ue_list):
        x = ue.get_coordinates()[0] - bs.get_coordinates()[0]
        y = ue.get_coordinates()[1] - bs.get_coordinates()[1]
        z = ue.get_coordinates()[2] - bs.get_coordinates()[2]

        ue_bs_distance_m = math.sqrt(math.pow(x, 2) + math.pow(y, 2) + math.pow(z, 2))

        ue.set_distance_from_bs(ue_bs_distance_m=ue_bs_distance_m)
