import sys
import time
import math as maths
import numpy as np
import pygame
import matplotlib.pyplot as plt

from epidemicmodeller import modeloutput

screensize = 800
dotsize = 4


class BasicModel(object):
    name = "basicmodel"

    def __init__(self, parameters):
        self.params = parameters
        self.max_duration = self.params.get("max_duration", 1000)
        self.timesteps_per_day = self.params.get("timesteps_per_day", 10)
        self.timestep = 1 / self.timesteps_per_day
        self.speed = self.params.get("speed", 0.04)
        self.params["max_duration"] = self.max_duration
        self.params["timesteps_per_day"] = self.timesteps_per_day
        self.params["timestep"] = self.timestep
        self.params["infect_distance"] = self.params.get("infect_distance", 0.05)
        self.params["speed"] = self.speed
        self.params["E"] = parameters.get("E", 0)
        self.params["I"] = parameters.get("I", 1)
        self.params["R"] = parameters.get("R", 0)
        self.params["R"] = parameters.get("R", 0)
        if "N" in parameters.keys():
            self.params["S"] = parameters.get("S", self.params["N"] - self.params["I"])
        else:
            self.params["N"] = parameters["S"] + parameters["E"] + parameters["I"] + parameters["R"]

    def run_model(self):
        tic = time.perf_counter()

        infect_rate = self.params.get("infect_rate", 2)
        latency_rate = self.params.get("latency_rate", 4)
        recovery_rate = self.params.get("recovery_rate", 6)
        next_infect_function = lambda: np.random.exponential(infect_rate)
        latency_end_function = lambda: np.random.gamma(latency_rate)
        recovery_function = lambda: np.random.gamma(recovery_rate)

        state = np.array(
            [0] * self.params["S"] + [1] * self.params["E"] + [2] * self.params["I"] + [3] * self.params["R"])
        infected_time = state * 0
        next_event = infected_time + np.Inf
        for i in np.where(state == 1)[0]:
            next_event[i] = next_infect_function()
        for i in np.where(state == 2)[0]:
            next_event[i] = latency_end_function()
        next_recovery_event = infected_time + np.Inf
        for i in np.where(state == 2)[0]:
            next_recovery_event[i] = recovery_function()

        x_coords = np.random.uniform(size=self.params["N"])
        y_coords = np.random.uniform(size=self.params["N"])
        x_velocs = np.random.uniform(-1, 1, size=self.params["N"]) * self.speed * self.timestep
        y_velocs = np.random.uniform(-1, 1, size=self.params["N"]) * self.speed * self.timestep

        t = 0
        k = 0

        state_log = np.zeros((self.max_duration * self.timesteps_per_day, len(state)))
        x_coords_log = np.zeros((self.max_duration * self.timesteps_per_day, self.params["N"]))
        y_coords_log = np.zeros((self.max_duration * self.timesteps_per_day, self.params["N"]))
        event_log = [[] for _ in range(self.max_duration * self.timesteps_per_day)]
        classes = {"S": [], "E": [], "I": [], "R": [], "t": []}
        state_log[0] = state
        x_coords_log[0] = x_coords
        y_coords_log[0] = y_coords
        classes["S"].append(sum(state == 0))
        classes["E"].append(sum(state == 1))
        classes["I"].append(sum(state == 2))
        classes["R"].append(sum(state == 3))
        classes["t"].append(t)

        while (1 in state or 2 in state) and t <= self.max_duration:
            x_coords = x_coords + x_velocs
            y_coords = y_coords + y_velocs
            invalid = np.where(x_coords > 1)[0]
            for i in invalid:
                x_coords[i] = 2 - x_coords[i]
                x_velocs[i] *= -1
            invalid = np.where(x_coords < 0)[0]
            for i in invalid:
                x_coords[i] = -x_coords[i]
                x_velocs[i] *= -1
            invalid = np.where(y_coords > 1)[0]
            for i in invalid:
                y_coords[i] = 2 - y_coords[i]
                y_velocs[i] *= -1
            invalid = np.where(y_coords < 0)[0]
            for i in invalid:
                y_coords[i] = -y_coords[i]
                y_velocs[i] *= -1
            t += self.timestep
            k += 1

            # Event IDs: Infect attempt: 0, Got Exposed: 1, Latency ended: 2, Recovery: 3
            for j in np.where(next_event < t)[0]:
                if state[j] == 1:  # Exposed -> Infectious
                    state[j] = 2
                    next_event[j] = t + next_infect_function()
                    next_recovery_event[j] = t + recovery_function()
                    event_log[k].append((j, 2))
                elif state[j] == 2:  # Infecting another
                    next_event[j] = t + next_infect_function()
                    event_log[k].append((j, 0))
                    for l in np.where(state == 0)[0]:
                        if max(np.abs(x_coords[l] - x_coords[j]), np.abs(y_coords[l] - y_coords[j])) < self.params.get(
                                "infect_distance", 0.05):
                            state[l] = 1
                            next_event[l] = t + latency_end_function()
                            event_log[k].append((l, 1))
                            break
            for j in np.where(next_recovery_event < t)[0]:  # Recovery
                next_recovery_event[j] = np.Inf
                next_event[j] = np.Inf
                state[j] = 3
                event_log[k].append((j, 3))

            state_log[k] = state
            x_coords_log[k] = x_coords
            y_coords_log[k] = y_coords
            classes["S"].append(sum(state == 0))
            classes["E"].append(sum(state == 1))
            classes["I"].append(sum(state == 2))
            classes["R"].append(sum(state == 3))
            classes["t"].append(t)

        logindex = round(t * self.timesteps_per_day)
        state_log = state_log[:logindex]
        x_coords_log = x_coords_log[:logindex]
        y_coords_log = y_coords_log[:logindex]
        event_log = event_log[:logindex]
        toc = time.perf_counter()
        return BasicModelOutput(self.params, sum(state == 3) - self.params["R"], classes, round(t, 3), toc - tic,
                                {"state_log": state_log, "x_coords_log": x_coords_log,
                                 "y_coords_log": y_coords_log, "event_log": event_log})


class BasicModelOutput(modeloutput.ModelOutput):
    def __init__(self, params, final_size, classes, duration, simulation_time, data_log):
        modeloutput.ModelOutput.__init__(self, "basicmodel", params, final_size, classes, duration, simulation_time, data_log)


if __name__ == "__main__":
    output = BasicModel({"N": 500, "I": 1}).run_model()
    print(
        f"Model took {output.simulation_time:0.2f} seconds to run, it lasted {output.duration} days, and had a final size of {output.final_size}")
    from epidemicmodeller import basicmodelrenderer
    fig, ax = plt.subplots()
    ax.plot(output.classes["t"], output.classes["S"])
    ax.plot(output.classes["t"], output.classes["E"])
    ax.plot(output.classes["t"], output.classes["I"])
    ax.plot(output.classes["t"], output.classes["R"])
    ax.legend(["S", "E", "I", "R"])
    ax.set(xlabel="Time (days)", ylabel="Infectious", title="Infectious over time")
    plt.show()
    #basicmodelrenderer.render_basic_model(output)
