import json
from datetime import datetime

def save_all_outputs(inputs: dict, file_name: str):
    """
    Saves all the outputs into a json file

    Args:
      inputs: dict: dictionary of inputs
      file_name: str: name of output file

    Returns:
        None

    """

    current_date = datetime.now()

    year = current_date.year
    month = current_date.month
    day = current_date.day
    hour = current_date.hour
    minute = current_date.minute
    second = current_date.second
    path = "multi_hop_industrial_simulator/results/" + f'{year}_{month:02d}_{day:02d}_{hour}_{minute}_{second}_' + file_name

    for metric in inputs.keys():
        inputs[metric]= inputs[metric].tolist()

    with open(path, "w") as json_file:
        json.dump(inputs, json_file, indent=4)  # Adding indentation for readability
        print('The simulation is finished and the output has been saved!')
