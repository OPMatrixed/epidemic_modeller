import sys
import time
import math as maths
import numpy as np
import pygame

from epidemicmodeller import modeloutput

screensize = 800
dotsize = 4
speed = 0.04


class BasicModel(object):
    name = "basicmodel"

    def __init__(self, parameters):
        self.params = parameters
        self.params["E"] = parameters.get("E", 0)
        self.params["I"] = parameters.get("I", 1)
        self.params["R"] = parameters.get("R", 0)
        self.params["R"] = parameters.get("R", 0)
        if "N" in parameters.keys():
            self.params["S"] = parameters.get("S", self.params["N"] - self.params["I"])
        else:
            self.params["N"] = parameters["S"] + parameters["E"] + parameters["I"] + parameters["R"]

    def run_model(self):
        infect_rate = self.params.get("infect_rate", 2)
        latency_rate = self.params.get("latency_rate", 4)
        recovery_rate = self.params.get("recovery_rate", 6)
        next_infect_function = lambda: np.random.exponential(infect_rate)
        latency_end_function = lambda: np.random.gamma(latency_rate)
        recovery_function = lambda: np.random.gamma(recovery_rate)

        timesteps_per_day = self.params.get("timesteps_per_day", 8)
        timestep = 1/timesteps_per_day

        state = np.array([0] * self.params["S"] + [1] * self.params["E"] + [2] * self.params["I"] + [3] * self.params["R"])
        infected_time = state*0
        next_event = infected_time+np.Inf
        for i in np.where(state == 1)[0]:
            next_event[i] = next_infect_function()
        for i in np.where(state == 2)[0]:
            next_event[i] = latency_end_function()
        next_recovery_event = infected_time + np.Inf
        for i in np.where(state == 2)[0]:
            next_recovery_event[i] = recovery_function()

        x_coords = np.random.uniform(size=self.params["N"])
        y_coords = np.random.uniform(size=self.params["N"])
        x_velocs = np.random.uniform(-1, 1, size=self.params["N"]) * speed * timestep
        y_velocs = np.random.uniform(-1, 1, size=self.params["N"]) * speed * timestep
        pygame.init()

        screen = pygame.display.set_mode((screensize, screensize))

        t = 0
        k = 0
        while (1 in state or 2 in state) and t <= self.params.get("max_duration", 1000):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    break
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
            t += timestep
            k += 1

            screen.fill((0, 0, 0))
            colours = [pygame.Color(255, 255, 255),
                       pygame.Color(0, 255, 0),
                       pygame.Color(255, 0, 0),
                       pygame.Color(100, 100, 100)]
            for i in range(len(x_coords)):
                pygame.draw.circle(screen, colours[state[i]],
                                   (round(x_coords[i]*screensize), round(y_coords[i]*screensize)), dotsize)
            if k % 16 == 0:
                print("Time", t)
            for j in np.where(next_event < t)[0]:
                if state[j] == 1:
                    state[j] = 2
                    next_event[j] = t + next_infect_function()
                    next_recovery_event[j] = t + recovery_function()
                elif state[j] == 2:
                    next_event[j] = t + next_infect_function()
                    pixel_radius = maths.ceil(self.params.get("infect_distance", 0.05)*screensize/2)
                    rect = pygame.Rect(np.round(x_coords[j] * screensize)-pixel_radius, np.round(y_coords[j] * screensize)-pixel_radius, pixel_radius*2, pixel_radius*2)
                    pygame.draw.rect(screen, pygame.Color(150, 0, 0), rect)
                    for l in np.where(state == 0)[0]:
                        if max(np.abs(x_coords[l] - x_coords[j]), np.abs(y_coords[l] - y_coords[j])) < self.params.get("infect_distance", 0.05):
                            state[l] = 1
                            next_event[l] = t + latency_end_function()
                            break
            for j in np.where(next_recovery_event < t)[0]:
                next_recovery_event[j] = np.Inf
                next_event[j] = np.Inf
                state[j] = 3
            pygame.display.flip()
            time.sleep(0.05)
        time.sleep(3)
        pygame.display.quit()
        return BasicModelOutput(self.params, sum(state == 3) - self.params["R"], t, None)


class BasicModelOutput(modeloutput.ModelOutput):
    def __init__(self, params, final_size, duration, data_log):
        modeloutput.ModelOutput.__init__(self, "basicmodel", params, final_size, duration, data_log)


if __name__ == "__main__":
    output = BasicModel({"N": 200, "I": 1}).run_model()
