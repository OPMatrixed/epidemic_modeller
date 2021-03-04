import pygame
import numpy as np
import time
import math as maths


def render_basic_model(model_output, screensize=800, dotsize=3, days_per_second=4):
    pygame.init()
    screen = pygame.display.set_mode((screensize, screensize))
    font = pygame.font.SysFont(None, 64)

    pixel_radius = maths.ceil(model_output.params["infect_distance"] * screensize / 2)
    delay = 1/(model_output.params["timesteps_per_day"]*days_per_second)
    #print("Delay:", delay)

    event_log = model_output.data_log["event_log"]
    x_coord_log = model_output.data_log["x_coords_log"]
    y_coord_log = model_output.data_log["y_coords_log"]
    state_log = model_output.data_log["state_log"]

    for k in range(len(state_log)):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                break
        screen.fill((0, 0, 0))
        colours = [pygame.Color(255, 255, 255),
                   pygame.Color(0, 255, 0),
                   pygame.Color(255, 0, 0),
                   pygame.Color(100, 100, 100)]
        for i in range(len(state_log[k])):
            pygame.draw.circle(screen, colours[round(state_log[k, i])],
                               (round(x_coord_log[k, i] * screensize), round(y_coord_log[k, i] * screensize)), dotsize)
        for event in event_log[k]:
            if event[1] == 0:
                rect = pygame.Rect(np.round(x_coord_log[k, event[0]] * screensize) - pixel_radius,
                                   np.round(y_coord_log[k, event[0]] * screensize) - pixel_radius, pixel_radius * 2,
                                   pixel_radius * 2)
                pygame.draw.rect(screen, pygame.Color(150, 0, 0), rect)
        day_img = font.render(f"Day: {maths.floor(k/model_output.params['timesteps_per_day'])}", True, (240, 240, 240))
        screen.blit(day_img, (20, 20))
        pygame.display.flip()
        time.sleep(delay)
        #if k%(model_output.params["timesteps_per_day"]*10) == 0:
        #    print(k/model_output.params["timesteps_per_day"])

    time.sleep(1)
    pygame.display.quit()
