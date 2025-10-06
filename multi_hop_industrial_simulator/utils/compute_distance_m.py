import numpy as np

# Function to compute the distance in meters between a transmitter and a receiver
def compute_distance_m(tx, rx):
    """

    Args:
      tx: 
      rx: 

    Returns:

    """
    return np.sqrt((tx.x-rx.x)**2 + (tx.y-rx.y)**2 + (tx.z-rx.z)**2)
