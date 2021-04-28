# basic_model_renderer.py
# A file containing a function for rendering a Basic Model's realisation using pygame.

import numpy as np
import time
import math as maths
import os
# Importing pygame prints out a bunch of text, this suppresses it.
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame


def render_basic_model(model_output, screensize=800, dotsize=3, days_per_second=3):
    # Create the pygame instance.
    pygame.init()
    # Create the window/canvas to draw on.
    screen = pygame.display.set_mode((screensize, screensize))
    pygame.display.set_caption("Epidemic Modeller: Basic model renderer")
    font = pygame.font.SysFont(None, 64)
    # Calculate how many pixels wide the infection radius should be.
    pixel_radius = maths.ceil(model_output.params["infect_distance"] * screensize / 2)
    # Calculate the max frequency at which each frame should be drawn, as a delay between frames.
    delay = 1/(model_output.params["timesteps_per_day"]*days_per_second)

    event_log = model_output.data_log["event_log"]
    x_coord_log = model_output.data_log["x_coords_log"]
    y_coord_log = model_output.data_log["y_coords_log"]
    state_log = model_output.data_log["state_log"]

    for k in range(len(state_log)):
        # For each timestep,

        for event in pygame.event.get():
            # Check for pygame window events
            if event.type == pygame.QUIT:
                # If the window is told to close, exit.
                pygame.display.quit()
                return
        if k == 1:
            # On the first iteration, wait a few seconds before starting.
            time.sleep(2)
        # Fill the screen with a black background.
        screen.fill((0, 0, 0))
        # Create a colour vector to hold a colour for each SEIR class.
        colours = [pygame.Color(255, 255, 255),
                   pygame.Color(0, 255, 0),
                   pygame.Color(255, 0, 0),
                   pygame.Color(100, 100, 100)]
        for i in range(len(state_log[k])):
            # For each individual, draw them on the canvas.
            pygame.draw.circle(screen, colours[round(state_log[k, i])],
                               (round(x_coord_log[k, i] * screensize), round(y_coord_log[k, i] * screensize)), dotsize)
        for event in event_log[k]:
            if event[1] == 0:
                # Draw infect events as red squares around the infected person.
                rect = pygame.Rect(np.round(x_coord_log[k, event[0]] * screensize) - pixel_radius,
                                   np.round(y_coord_log[k, event[0]] * screensize) - pixel_radius, pixel_radius * 2,
                                   pixel_radius * 2)
                pygame.draw.rect(screen, pygame.Color(150, 0, 0), rect)
        # Draw text in the top right indicating the day of the current system state being shown.
        day_img = font.render(f"Day: {maths.floor(k/model_output.params['timesteps_per_day'])}", True, (240, 240, 240))
        screen.blit(day_img, (20, 20))
        pygame.display.flip()  # Then display this frame
        time.sleep(delay)  # And wait the delay before displaying the next frame.

    # Once we reach the end of the model history, close the window after a few seconds.
    time.sleep(2)
    pygame.display.quit()
