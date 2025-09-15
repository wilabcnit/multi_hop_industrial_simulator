"""
    Utility script to read parameters and data from files.
"""

import yaml

# Function to read a yaml file and return a dictionary
def read_inputs(filepath):
    """
        Read yaml file and return a Python dictionary with all parameters.
    """
    with open(filepath) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    return params
