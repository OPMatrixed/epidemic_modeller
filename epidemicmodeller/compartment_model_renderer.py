# compartment_model_renderer.py
# A file containing a function for rendering a Compartment Model's realisation using pygame.

import numpy as np
import time
import math as maths
import os
# Importing pygame prints out a bunch of text, this suppresses it.
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame


def render_compartment_model(model_output, screensize=800, dotsize=3, days_per_second=4):
    # Create the pygame instance.
    pygame.init()
    # Create the window/canvas to draw on.
    screen = pygame.display.set_mode((screensize, screensize))
    pygame.display.set_caption("Epidemic Modeller: Compartment model renderer")
    font = pygame.font.SysFont(None, 64)

    # Calculate the max frequency at which each frame should be drawn, as a delay between frames.
    delay = 1/(model_output.params["timesteps_per_day"]*days_per_second)

    # Then calculate where each of the compartments should be placed on the canvas.
    compartments = model_output.data_log
    nc = len(compartments)
    num_rows = maths.ceil(maths.sqrt(nc))
    num_cols = num_rows
    padding = 20
    regions = []
    col_box_length = screensize//num_cols
    for i in range(num_rows):
        regions += [(col_box_length*j + padding, col_box_length*i+padding) for j in range(num_cols)]

    for k in range(len(compartments[0].state_log)):
        # For each time step,
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
        t = k*model_output.params["timestep"]
        for c_i in range(len(compartments)):
            # For each compartment,
            c = compartments[c_i]
            current_region = regions[c_i]
            # Calculate the width and height of the currently selected compartment.
            r_width = col_box_length-padding*2
            r_height = col_box_length-padding*2
            # Draw a square around the borders of the compartment.
            rect = pygame.Rect(current_region[0], current_region[1], r_width, r_height)
            pygame.draw.rect(screen, pygame.Color(200, 200, 200), rect, 1)
            # Calculate how many pixels wide the infection radius should be.
            pixel_radius = maths.ceil(c.params["infect_distance"] * col_box_length / 2)
            for i in range(len(c.state_log[k])):
                # For each individual
                if c.state_log[k, i] == -1:
                    # If this is not an individual or they are a traveller that has left, ignore them.
                    continue
                # Draw the individual as a dot on in the compartment region.
                pygame.draw.circle(screen, colours[round(c.state_log[k, i])],
                                   (round(current_region[0] + c.x_coords_log[k, i] * r_width),
                                    round(current_region[1] + c.y_coords_log[k, i] * r_height)), dotsize)
            for event in c.event_log[k]:
                if event[1] == 0:
                    # Then draw the red squares around the infectious on infect attempt events.
                    rect = pygame.Rect(np.round(current_region[0] + c.x_coords_log[k, event[0]] * r_width) - pixel_radius,
                                       np.round(current_region[1] + c.y_coords_log[k, event[0]] * r_height) - pixel_radius,
                                       pixel_radius * 2, pixel_radius * 2)
                    pygame.draw.rect(screen, pygame.Color(150, 0, 0), rect)
        for tr in model_output.travellers[k]:
            # For each current traveller,

            # Calculate the path they are taking
            current_region = regions[tr.start]
            dest_region = regions[tr.destination]
            r_width = col_box_length - padding * 2
            r_height = col_box_length - padding * 2
            start_loc_x = current_region[0] + tr.start_coords[0] * r_width
            start_loc_y = current_region[1] + tr.start_coords[1] * r_height
            end_loc_x = dest_region[0] + 0.5 * r_width
            end_loc_y = dest_region[1] + 0.5 * r_height
            # Linearly interpolate between the start and end points depending on the time spent travelling.
            interp_x = end_loc_x + (tr.end_time - t) * (start_loc_x - end_loc_x)
            interp_y = end_loc_y + (tr.end_time - t) * (start_loc_y - end_loc_y)
            # And then draw them at that location.
            pygame.draw.circle(screen, colours[tr.state], (round(interp_x), round(interp_y)), dotsize)

        # Draw text in the top right indicating the day of the current system state being shown.
        day_img = font.render(f"Day: {maths.floor(k/model_output.params['timesteps_per_day'])}", True, (240, 240, 240))
        screen.blit(day_img, (20, 20))
        pygame.display.flip()  # Then display this frame
        time.sleep(delay)  # And wait the delay before displaying the next frame.

    # Once we reach the end of the model history, close the window after a few seconds.
    time.sleep(2)
    pygame.display.quit()
