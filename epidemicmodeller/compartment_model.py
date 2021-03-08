import time
import numpy as np
import matplotlib.pyplot as plt
import math as maths

from epidemicmodeller import modeloutput


class CompartmentModel(object):
    name = "compartment_model"

    def __init__(self, parameters):
        self.params = parameters
        self.params["compartments"] = self.params.get("compartments", 4)
        self.params["max_duration"] = self.params.get("max_duration", 1000)
        self.params["timesteps_per_day"] = self.params.get("timesteps_per_day", 10)
        self.params["timestep"] = 1 / self.params["timesteps_per_day"]
        self.params["speed"] = self.params.get("speed", 0.06)
        for i in range(self.params["compartments"]):
            self.params["E_"+str(i)] = parameters.get("E_"+str(i), parameters.get("E", 0))
            self.params["I_"+str(i)] = parameters.get("I_"+str(i), parameters.get("I", 0))
            self.params["R_"+str(i)] = parameters.get("R_"+str(i), parameters.get("R", 0))
            self.params["N_"+str(i)] = parameters.get("N_"+str(i), parameters.get("N", 100) // self.params["compartments"])
            self.params["S_"+str(i)] = self.params["N_"+str(i)] - self.params["E_"+str(i)] - self.params["I_"+str(i)] - self.params["R_"+str(i)]



    def run_model(self):
        tic = time.perf_counter()
        t = 0
        k = 0
        compartments = []
        for i in range(self.params["compartments"]):
            ICs = {"N": self.params.get("N_" + str(i), 100),
                   "I": self.params.get("I_" + str(i), 0), "E": self.params.get("E_" + str(i), 0),
                   "R": self.params.get("R_" + str(i), 0)}
            ICs["S"] = ICs["N"] - ICs["I"] - ICs["E"] - ICs["R"]
            compartments.append(Compartment(self.params, ICs))
        while t <= self.params["max_duration"] and any([(1 in i.state or 2 in i.state) for i in compartments]):
            for i in compartments:
                i.step(t, k)
            t += self.params["timestep"]
            k += 1
        for c in compartments:
            c.finish(k)
        toc = time.perf_counter()
        classes = {}
        classes["S"] = np.array(compartments[0].classes["S"])
        classes["E"] = np.array(compartments[0].classes["E"])
        classes["I"] = np.array(compartments[0].classes["I"])
        classes["R"] = np.array(compartments[0].classes["R"])
        for i in range(1, len(compartments)):
            classes["S"] = classes["S"] + np.array(compartments[i].classes["S"])
            classes["E"] = classes["E"] + np.array(compartments[i].classes["E"])
            classes["I"] = classes["I"] + np.array(compartments[i].classes["I"])
            classes["R"] = classes["R"] + np.array(compartments[i].classes["R"])
        classes["t"] = compartments[0].classes["t"]

        return CompartmentModelOutput(self.params, classes["R"][-1] - sum(self.params["R_"+str(i)] for i in range(self.params["compartments"])),
                                      classes, round(t, 3), toc - tic, compartments)


class CompartmentModelOutput(modeloutput.ModelOutput):
    def __init__(self, params, final_size, classes, duration, simulation_time, compartments):
        modeloutput.ModelOutput.__init__(self, "compartment_model", params, final_size, classes, duration, simulation_time, compartments)
        self.compartments = compartments


class Compartment(object):
    def __init__(self, params, ICs):
        self.params = params.copy()
        self.ICs = ICs

        self.params["b"] = self.params.get("b", 1)
        self.params["lambda"] = (1 - maths.exp(-self.params["b"])) / self.params["beta"]
        self.params["infect_distance"] = self.params["b"] / maths.sqrt(self.ICs["N"])

        self.state = np.array(
            [0] * self.ICs["S"] + [1] * self.ICs["E"] + [2] * self.ICs["I"] + [3] * self.ICs["R"])
        self.x_coords = np.random.uniform(size=self.ICs["N"])
        self.y_coords = np.random.uniform(size=self.ICs["N"])
        self.x_velocs = np.random.uniform(-1, 1, size=self.ICs["N"]) * self.params["speed"] * self.params["timestep"]
        self.y_velocs = np.random.uniform(-1, 1, size=self.ICs["N"]) * self.params["speed"] * self.params["timestep"]
        infect_rate = self.params.get("lambda", 2)
        latency_rate = 1 / self.params.get("sigma", 1 / 4)
        recovery_rate = 1 / self.params.get("gamma", 1 / 6)
        self.next_infect_function = lambda: np.random.exponential(infect_rate)
        self.latency_end_function = lambda: np.random.gamma(latency_rate)
        self.recovery_function = lambda: np.random.gamma(recovery_rate / infect_rate, infect_rate)

        self.infected_time = self.state * 0
        self.next_event = self.infected_time + np.Inf
        for i in np.where(self.state == 1)[0]:
            self.next_event[i] = self.next_infect_function()
        for i in np.where(self.state == 2)[0]:
            self.next_event[i] = self.latency_end_function()
        self.next_recovery_event = self.infected_time + np.Inf
        for i in np.where(self.state == 2)[0]:
            self.next_recovery_event[i] = self.recovery_function()

        self.state_log = np.zeros((self.params["max_duration"] * self.params["timesteps_per_day"], len(self.state)))
        self.x_coords_log = np.zeros((self.params["max_duration"] * self.params["timesteps_per_day"], self.ICs["N"]))
        self.y_coords_log = np.zeros((self.params["max_duration"] * self.params["timesteps_per_day"], self.ICs["N"]))
        self.event_log = [[] for _ in range(self.params["max_duration"] * self.params["timesteps_per_day"])]
        self.classes = {"S": [], "E": [], "I": [], "R": [], "t": []}
        self.state_log[0] = self.state
        self.x_coords_log[0] = self.x_coords
        self.y_coords_log[0] = self.y_coords
        self.classes["S"].append(self.ICs["S"])
        self.classes["E"].append(self.ICs["E"])
        self.classes["I"].append(self.ICs["I"])
        self.classes["R"].append(self.ICs["R"])
        self.classes["t"].append(0)

    def step(self, t, k):
        self.x_coords = self.x_coords + self.x_velocs
        self.y_coords = self.y_coords + self.y_velocs
        invalid = np.where(self.x_coords > 1)[0]
        for i in invalid:
            self.x_coords[i] = 2 - self.x_coords[i]
            self.x_velocs[i] *= -1
        invalid = np.where(self.x_coords < 0)[0]
        for i in invalid:
            self.x_coords[i] = -self.x_coords[i]
            self.x_velocs[i] *= -1
        invalid = np.where(self.y_coords > 1)[0]
        for i in invalid:
            self.y_coords[i] = 2 - self.y_coords[i]
            self.y_velocs[i] *= -1
        invalid = np.where(self.y_coords < 0)[0]
        for i in invalid:
            self.y_coords[i] = -self.y_coords[i]
            self.y_velocs[i] *= -1
        event_log = []

        for j in np.where(self.next_event < t)[0]:
            if self.state[j] == 1:  # Exposed -> Infectious
                self.state[j] = 2
                self.next_event[j] = t + self.next_infect_function()
                self.next_recovery_event[j] = t + self.recovery_function()
                event_log.append((j, 2))
            elif self.state[j] == 2:  # Infecting another
                self.next_event[j] = t + self.next_infect_function()
                event_log.append((j, 0))
                for l in np.where(self.state == 0)[0]:
                    if max(np.abs(self.x_coords[l] - self.x_coords[j]), np.abs(self.y_coords[l] - self.y_coords[j])) <\
                            self.params["infect_distance"]:
                        self.state[l] = 1
                        self.next_event[l] = t + self.latency_end_function()
                        event_log.append((l, 1))
                        break
        for j in np.where(self.next_recovery_event < t)[0]:  # Recovery
            self.next_recovery_event[j] = np.Inf
            self.next_event[j] = np.Inf
            self.state[j] = 3
            event_log.append((j, 3))
        self.state_log[k] = self.state
        self.x_coords_log[k] = self.x_coords
        self.y_coords_log[k] = self.y_coords
        self.classes["S"].append(sum(self.state == 0))
        self.classes["E"].append(sum(self.state == 1))
        self.classes["I"].append(sum(self.state == 2))
        self.classes["R"].append(sum(self.state == 3))
        self.classes["t"].append(t)
        self.event_log[k] = event_log

    def finish(self, k):
        self.state_log = self.state_log[:k]
        self.x_coords_log = self.x_coords_log[:k]
        self.y_coords_log = self.y_coords_log[:k]
        self.event_log = self.event_log[:k]


if __name__ == "__main__":
    num_compartments = 9
    output = CompartmentModel({"N": 700, "I": 1, "gamma": 1/6, "beta": 1/3, "b": 1, "compartments": num_compartments}).run_model()
    print(f"Compartment Model took {output.simulation_time:0.2f} seconds to run, it lasted {output.duration} days, "
          + f"and had a final size of {output.final_size}")

    row_length = maths.ceil(maths.sqrt(num_compartments))
    fig, ax = plt.subplots(row_length, row_length, figsize=(row_length*3, row_length*3), dpi=100)
    for i in range(num_compartments):
        ax[i//row_length][i%row_length].plot(output.classes["t"], output.compartments[i].classes["S"])
        ax[i//row_length][i%row_length].plot(output.classes["t"], output.compartments[i].classes["E"])
        ax[i//row_length][i%row_length].plot(output.classes["t"], output.compartments[i].classes["I"])
        ax[i//row_length][i%row_length].plot(output.classes["t"], output.compartments[i].classes["R"])
        ax[i//row_length][i%row_length].legend(["S", "E", "I", "R"])
        ax[i//row_length][i%row_length].set(xlabel="Time (days)", ylabel="Population", title="Plot of epidemic")
    plt.show()
    from epidemicmodeller import comparment_model_renderer
    comparment_model_renderer.render_compartment_model(output)
