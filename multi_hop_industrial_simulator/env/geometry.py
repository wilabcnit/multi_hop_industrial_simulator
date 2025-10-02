from pandas import DataFrame

'''
Geometry class
'''

class Geometry:
    def __init__(self, scenario_df: DataFrame = None):
        self.factory_length = scenario_df.loc['Pilot line', "X-center"] * 2
        self.factory_width = scenario_df.loc['Pilot line', "Y-center"] * 2
        self.factory_height = scenario_df.loc['Pilot line', "Height"]
        self.clutter_density=0

    # Set the dimension of the factory
    def set_factory_dimensions(self, input_factory_length: float, input_factory_width: float,
                               input_factory_height: float):
        self.factory_length = input_factory_length
        self.factory_width = input_factory_width
        self.factory_height = input_factory_height

    # Get the clutter density
    def get_clutter_density(self, machines):
        area_machines=0
        for i in range(0, len(machines)):
            area_machines += machines[i].machine_size **2 #(machines[i]. x_max-machines[i]. x_min )* (machines[i]. y_max-machines[i]. y_min )

        self.clutter_density = area_machines/ (self.factory_length* self.factory_width)

    # Get the length of the environment
    def get_factory_length(self):
        return self.factory_length

    # Get the width of the environment
    def get_factory_width(self):
        return self.factory_width

    # Get the height of the environment
    def get_factory_height(self):
        return self.factory_height

    # Get the area of the environment
    def get_factory_area(self):
        return self.factory_length * self.factory_width

    # Get the volume of the environment
    def get_factory_volume(self):
        return self.factory_length * self.factory_width * self.factory_height
