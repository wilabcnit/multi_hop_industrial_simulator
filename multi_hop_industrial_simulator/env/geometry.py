from pandas import DataFrame

'''
Geometry class:
Set the dimensions of the Factory in the industrial environment depending on the use case chosen
'''

class Geometry:
    """ """
    def __init__(self, scenario_df: DataFrame = None):
        self.factory_length = scenario_df.loc['Pilot line', "X-center"] * 2
        self.factory_width = scenario_df.loc['Pilot line', "Y-center"] * 2
        self.factory_height = scenario_df.loc['Pilot line', "Height"]
        self.clutter_density=0

    def set_factory_dimensions(self, input_factory_length: float, input_factory_width: float,
                               input_factory_height: float):
        """

        Args:
          input_factory_length: float: length of input factory scenario
          input_factory_width: float: width of input factory scenario
          input_factory_height: float: height of input factory scenario

        Returns:
            dimension of the factory

        """
        self.factory_length = input_factory_length
        self.factory_width = input_factory_width
        self.factory_height = input_factory_height

    def get_clutter_density(self, machines):
        """

        Args:
          machines: array of machines

        Returns:
            clutter density

        """
        area_machines=0
        for i in range(0, len(machines)):
            area_machines += machines[i].machine_size **2 #(machines[i]. x_max-machines[i]. x_min )* (machines[i]. y_max-machines[i]. y_min )

        self.clutter_density = area_machines/ (self.factory_length* self.factory_width)

    def get_factory_length(self):
        """
        Returns: length of the environment
        """
        return self.factory_length

    def get_factory_width(self):
        """
        Returns: width of the environment
        """
        return self.factory_width

    def get_factory_height(self):
        """
        Returns: height of the environment
        """
        return self.factory_height

    def get_factory_area(self):
        """
        Returns: area of the environment
        """
        return self.factory_length * self.factory_width

    def get_factory_volume(self):
        """
        Returns: volume of the environment
        """
        return self.factory_length * self.factory_width * self.factory_height
