
import numpy as np
import math

class TrafficModel:
    def __init__(self, input_full_queue: bool):
        self.time_periodicity_ticks = 0  # Periodicity of periodic UEs in ticks
        self.t_generation = 0  # All UEs start generating the first data at the beginning of the simulation
        self.t_generation_optimization = None  # NEW
        self.scale_value = 0
        self.tick = 0
        self.period_exponential = 0
        self.full_queue = input_full_queue

    # Set the time periodicity in ticks
    def set_time_periodicity(self, time_periodicity_ticks: int):
        self.time_periodicity_ticks = time_periodicity_ticks

    # Get the time periodicity in ticks
    def get_time_periodicity(self):
        return self.time_periodicity_ticks

    # Set the time of generation of the next packet in ticks
    def set_t_generation_optimization(self, input_generation: float):
        self.t_generation_optimization = input_generation

    # Set the next time of generation of the packet in ticks (according to an exponential distribution)
    def set_exp_distribution(self, scale_value: float, tick_simulator: float):
        self.scale_value = scale_value
        self.tick = tick_simulator
        sample_exponential = np.random.exponential(scale_value, None)
        self.t_generation_optimization += math.ceil(sample_exponential / self.tick)

    # Get the next time of generation of the packet in ticks (according to an exponential distribution)
    def get_exp_distribution(self):
        if self.scale_value == 0:
            self.t_generation_optimization = 0
        else:
            sample_exponential = np.random.exponential(self.scale_value, None)
            self.t_generation_optimization = math.ceil(sample_exponential / self.tick)
        return self.t_generation_optimization

    # Get the full queue mode
    def get_full_queue_mode(self):
        return self.full_queue
