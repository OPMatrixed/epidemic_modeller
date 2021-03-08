import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
import time
import math as maths


def render_compartment_model(model_output, screensize=800, dotsize=3, days_per_second=4):
    pygame.init()
    screen = pygame.display.set_mode((screensize, screensize))
    font = pygame.font.SysFont(None, 64)

    delay = 1/(model_output.params["timesteps_per_day"]*days_per_second)

    compartments = model_output.data_log
    nc = len(compartments)
    num_rows = maths.ceil(maths.sqrt(nc))
    num_cols = num_rows
    padding = 20
    regions = []
    col_box_length = screensize//num_cols
    for i in range(num_rows):
        regions += [(col_box_length*j + padding, col_box_length*i+padding) for j in range(num_cols)]
    time.sleep(2)

    for k in range(len(compartments[0].state_log)):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                return
        screen.fill((0, 0, 0))
        colours = [pygame.Color(255, 255, 255),
                   pygame.Color(0, 255, 0),
                   pygame.Color(255, 0, 0),
                   pygame.Color(100, 100, 100)]
        for c_i in range(len(compartments)):
            c = compartments[c_i]
            current_region = regions[c_i]
            r_width = col_box_length-padding*2
            r_height = col_box_length-padding*2
            rect = pygame.Rect(current_region[0], current_region[1], r_width, r_height)
            pygame.draw.rect(screen, pygame.Color(200, 200, 200), rect, 1)
            pixel_radius = maths.ceil(c.params["infect_distance"] * col_box_length / 2)
            for i in range(len(c.state_log[k])):
                pygame.draw.circle(screen, colours[round(c.state_log[k, i])],
                                   (round(current_region[0] + c.x_coords_log[k, i] * r_width),
                                    round(current_region[1] + c.y_coords_log[k, i] * r_height)), dotsize)
            for event in c.event_log[k]:
                if event[1] == 0:
                    rect = pygame.Rect(np.round(current_region[0] + c.x_coords_log[k, event[0]] * r_width) - pixel_radius,
                                       np.round(current_region[1] + c.y_coords_log[k, event[0]] * r_height) - pixel_radius,
                                       pixel_radius * 2, pixel_radius * 2)
                    pygame.draw.rect(screen, pygame.Color(150, 0, 0), rect)
        day_img = font.render(f"Day: {maths.floor(k/model_output.params['timesteps_per_day'])}", True, (240, 240, 240))
        screen.blit(day_img, (20, 20))
        pygame.display.flip()
        time.sleep(delay)

    time.sleep(2)
    pygame.display.quit()
