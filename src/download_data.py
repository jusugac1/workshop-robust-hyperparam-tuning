# =================
# ==== IMPORTS ====
# =================

import pyreadr


# ===================
# ==== FUNCTIONS ====
# ===================

def read_rdata(file_path: str):
    """Read an RData file and return its content as a pandas DataFrame.
    
    Args:
        file_path (str): The path to the RData file.

    Returns:
        (pd.DataFrame): The content of the RData file as a pandas DataFrame.
    """
    result = pyreadr.read_r(file_path)
    # Assuming the RData file contains a single data frame
    return next(iter(result.values()))
