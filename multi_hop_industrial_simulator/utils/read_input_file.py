import os
import pandas as pd

# Function to read an input Excel file and return a DataFrame
def read_input_file(file_name: str, sheet_name: str, reset_index: bool = None):

    file_path = os.path.join(os.path.dirname(__file__), file_name)

    if reset_index is not None and reset_index:
        input_df = pd.read_excel(file_name, sheet_name, engine='openpyxl', index_col=0)
    else:
        input_df = pd.read_excel(file_name, sheet_name, engine='openpyxl')

    return input_df
