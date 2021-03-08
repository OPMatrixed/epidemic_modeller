import time
import numpy as np
import math as maths

from epidemicmodeller import modeloutput


class CompartmentModel(object):
    name = "compartment_model"

    def __init__(self, parameters):
        self.params = parameters
        self.params["compartments"] = self.params.get("compartments", 4)

    def run_model(self):
        tic = time.perf_counter()
        t = 0
        k = 0
        compartments = [[np.random.uniform(size=self.params["N"]),
                         np.random.uniform(size=self.params["N"]),
                         np.random.uniform(-1, 1, size=self.params["N"]) * self.speed * self.timestep,
                         np.random.uniform(-1, 1, size=self.params["N"]) * self.speed * self.timestep,
                         []] for i in range(self.params["compartments"])]

        toc = time.perf_counter()

        return CompartmentModelOutput(self.params, 0, 0, round(t, 3), toc - tic, {})


class CompartmentModelOutput(modeloutput.ModelOutput):
    def __init__(self, params, final_size, classes, duration, simulation_time, data_log):
        modeloutput.ModelOutput.__init__(self, "compartment_model", params, final_size, classes, duration, simulation_time, data_log)
