"""
    Utility script to read parameters and data from files.
"""

import yaml

def read_inputs(filepath):
    """Read yaml file and return a Python dictionary with all parameters.

    Args:
      filepath: path to yaml file

    Returns:
        dictionary of input file parameters

    """
    with open(filepath) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    return params
