"""
    Utility script to save the output of a simulation.
"""

import json

# Function to save the output of a simulation to a JSON file
def save_simulation_output(x: list, y: list or dict, inputs: dict, file_name: str):
    data_to_save = {"x": x, "y": y}
    data_to_save.update(inputs)
    with open(file_name, "w") as json_file:
        json.dump(data_to_save, json_file, indent=0)  # Adding indentation for readability
        # print('The simulation is finished and the output has been saved!')

