
import numpy as np
import math
"""
   TrafficModel class

    This class defines the traffic generation model used in the simulation.

    It handles:
    - Configuration of input traffic characteristics
    - Support for different traffic patterns:
        - Periodic generation
        - Exponential distribution
        - Full queue mode
"""

class TrafficModel:

    def __init__(self, input_full_queue: bool):
        self.time_periodicity_ticks = 0  # Periodicity of periodic UEs in ticks
        self.t_generation = 0  # All UEs start generating the first data at the beginning of the simulation
        self.t_generation_optimization = None  # NEW
        self.scale_value = 0
        self.tick = 0
        self.period_exponential = 0
        self.full_queue = input_full_queue

    def set_time_periodicity(self, time_periodicity_ticks: int):
        """
        Set the time periodicity (in simulation ticks) for packet generation or events.

        Args:
            time_periodicity_ticks (int): The number of ticks representing the time periodicity.

        Returns:
            None
        """
        self.time_periodicity_ticks = time_periodicity_ticks

    def get_time_periodicity(self):
        """
        Get the current time periodicity (in ticks).

        Returns:
            int: The time periodicity in ticks.
        """
        return self.time_periodicity_ticks

    def set_t_generation_optimization(self, input_generation: float):
        """
        Set the optimized generation time (in ticks) for the next packet.

        Args:
            input_generation (float): The next packet generation time in ticks.

        Returns:
            None
        """
        self.t_generation_optimization = input_generation

    def set_exp_distribution(self, scale_value: float, tick_simulator: float):
        """
        Set the next packet generation time using an exponential distribution.

        Args:
            scale_value (float): The mean value (scale) of the exponential distribution.
            tick_simulator (float): The simulator tick duration used for scaling the generated time.

        Returns:
            None
        """
        self.scale_value = scale_value
        self.tick = tick_simulator
        sample_exponential = np.random.exponential(scale_value, None)
        self.t_generation_optimization += math.ceil(sample_exponential / self.tick)

    def get_exp_distribution(self):
        """
        Get the next packet generation time based on an exponential distribution.

        Returns:
            int: The generated packet time interval (in ticks) from the exponential distribution.
        """
        if self.scale_value == 0:
            self.t_generation_optimization = 0
        else:
            sample_exponential = np.random.exponential(self.scale_value, None)
            self.t_generation_optimization = math.ceil(sample_exponential / self.tick)
        return self.t_generation_optimization

    def get_full_queue_mode(self):
        """
        Get the status of the full queue mode (whether the UE has always at least one packet in the queue).

        Returns:
            bool: True if the full queue mode is active, False otherwise.
        """
        return self.full_queue
