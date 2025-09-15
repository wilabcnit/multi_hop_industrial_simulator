import sys

'''
Machine class

Industrial fixed machines

They are modeled as cubes of size "machine_size"  
'''


class Machine:
    def __init__(self, x_center: float, y_center: float, z_center: float, machine_size: int, max_number_of_ues: int):
        # Center_coordinates
        self.x_center = x_center
        self.y_center = y_center
        self.z_center = z_center
        self.machine_size = machine_size
        self.height = self.z_center + self.machine_size / 2
        # Coordinates of the area boundary
        self.x_max = self.x_center + self.machine_size / 2
        self.x_min = self.x_center - self.machine_size / 2
        self.y_max = self.y_center + self.machine_size / 2
        self.y_min = self.y_center - self.machine_size / 2

        self.z_max = self.z_center + self.machine_size / 2
        self.z_min = self.z_center - self.machine_size / 2

        self.max_number_of_ues = max_number_of_ues
        self.number_of_ues = 0
        self.id =-1

    # Set coordinates of the center of the machines (modeled as cubes) inside the environment and the
    # min/max coordinates along x and y axis
    def set_coordinates(self, x_input, y_input, z_input):
        self.x_center = x_input
        self.y_center = y_input
        self.z_center = z_input
        self.x_max = x_input + self.machine_size / 2
        self.x_min = x_input - self.machine_size / 2
        self.y_max = y_input + self.machine_size / 2
        self.y_min = y_input - self.machine_size / 2

    # Get the coordinates of the machines
    def get_coordinates(self):
        return self.x_center, self.y_center, self.z_center, self.x_max, self.x_min, self.y_max, self.y_min

    # Get the size of the machines within the environment
    def get_machine_size(self):
        return self.machine_size

    # Set the size of the machines within the environment
    def set_machine_size(self, machine_size: int):
        self.machine_size = machine_size

    # Set the maximum number of UEs per machine
    def set_max_number_of_ues(self, max_number_of_ues: int):
        self.max_number_of_ues = max_number_of_ues

    # Get the maximum number of UEs per machine
    def get_max_number_of_ues(self):
        return self.max_number_of_ues

    # Add a new UE inside the machine
    def add_new_ue(self):
        if self.number_of_ues + 1 <= self.max_number_of_ues:
            self.number_of_ues += 1
        else:
            return sys.exit("You are trying to add a new UE but the current machine is already full!")

    # Get the number of UEs per machine
    def get_number_of_ues(self):
        return self.number_of_ues

    # Get the machine height
    def get_machine_height(self):
        return self.height

    def move_machine(self, x, y, step, min_x, max_x, min_y, max_y):
        """
            Move the machine in a clockwise direction
        """
        if x == min_x and min_y <= y < max_y:
            y = y + step  # Move above
        elif y == max_y and min_x <= x < max_x:
            x = x + step  # Move to the right
        elif x == max_x and min_y < y <= max_y:
            y = y - step  # Move down
        elif y == min_y and min_x < x <= max_x:
            x = x - step  # Move to the left
        return x, y
