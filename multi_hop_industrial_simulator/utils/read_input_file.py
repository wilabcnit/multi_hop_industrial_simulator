import os
import pandas as pd


def read_input_file(file_name: str, sheet_name: str, reset_index: bool = None):
    """

    Args:
      file_name: str: name of the input file to be read
      sheet_name: str: name of the sheet to be read
      reset_index: bool:  (Default value = None)

    Returns:
        Dataframe of the input file

    """

    file_path = os.path.join(os.path.dirname(__file__), file_name)

    if reset_index is not None and reset_index:
        input_df = pd.read_excel(file_name, sheet_name, engine='openpyxl', index_col=0)
    else:
        input_df = pd.read_excel(file_name, sheet_name, engine='openpyxl')

    return input_df
