import sys

'''
Machine class

Industrial machines placed in fixed locations:
They can be modeled as cubes of fixed size or they can be parallelepiped of different sizes depending on the use case.
 
'''


class Machine:
    """ """
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

    def set_coordinates(self, x_input, y_input, z_input):
        """

        Args:
          x_input: x coordinate of the machine center
          y_input: y coordinate of the machine center
          z_input: z coordinate of the machine center

        Set coordinates of the center of the machines (modeled as cubes) inside the environment and the
        min/max coordinates along x and y axis

        """
        self.x_center = x_input
        self.y_center = y_input
        self.z_center = z_input
        self.x_max = x_input + self.machine_size / 2
        self.x_min = x_input - self.machine_size / 2
        self.y_max = y_input + self.machine_size / 2
        self.y_min = y_input - self.machine_size / 2

    def get_coordinates(self):
        """
        Returns: coordinates of the center of the machines inside the environment and the max/min
        coordinates along x and y axis
        """
        return self.x_center, self.y_center, self.z_center, self.x_max, self.x_min, self.y_max, self.y_min

    def get_machine_size(self):
        """
        Returns: size of the machine within the environment
        """
        return self.machine_size

    def set_machine_size(self, machine_size: int):
        """

        Args:
          machine_size: int: input size of the machine

        Returns:
            size of the machines within the environment

        """
        self.machine_size = machine_size

    def set_max_number_of_ues(self, max_number_of_ues: int):
        """

        Args:
          max_number_of_ues: int: input maximum number of UEs

        Returns:
            maximum number of UEs per machine

        """
        self.max_number_of_ues = max_number_of_ues

    def get_max_number_of_ues(self):
        """
        Returns: maximum number of UEs per machine
        """
        return self.max_number_of_ues

    def add_new_ue(self):
        """
        Add a new UE inside the machine
        """
        if self.number_of_ues + 1 <= self.max_number_of_ues:
            self.number_of_ues += 1
        else:
            return sys.exit("You are trying to add a new UE but the current machine is already full!")

    def get_number_of_ues(self):
        """
        Returns: number of UEs per machine
        """
        return self.number_of_ues

    def get_machine_height(self):
        """
        Returns: height of the machine
        """
        return self.height

    def move_machine(self, x, y, step, min_x, max_x, min_y, max_y):
        """Move the machine in a clockwise direction

        Args:
          x: x center of machine
          y: y center of machine
          step: step size of the movement of machines
          min_x: min x coordinate of the machine
          max_x: max x coordinate of the machine
          min_y: min y coordinate of the machine
          max_y: max y coordinate of the machine

        Returns:
            x and y center of the machine after move

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
