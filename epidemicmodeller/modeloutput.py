# modeloutput.py
# A file containing the model output object, which can be inherited
#  by other models to create custom model output objects.

class ModelOutput(object):
    def __init__(self, _type, params, final_size, classes, duration, simulation_time, data_log):
        self.type = _type
        self.params = params
        self.final_size = final_size
        self.classes = classes
        self.duration = duration
        self.simulation_time = simulation_time
        self.data_log = data_log
