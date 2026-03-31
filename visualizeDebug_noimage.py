import pygame
import numpy as np
from environment_noimage import pSCT_environment
pygame.init()

# ------------------------------------------------ necessary game settings ------------------------------------------------------

game_name = "pSCT sim debugger" # the name of the window that pops up
WIDTH = 1352-50 # pixels
HEIGHT = 484 # pixels
FRAME = 0
FRAME_RATE = 60 # frames / second
background_color = (0, 0, 0) # rgb color; each value ranges from 0-225 inclusive

# ----------------------------------------------------- variables ---------------------------------------------------------------

env = pSCT_environment(n_panels=2)
obs, _ = env.reset()

# ----------------------------------------------------- game logic --------------------------------------------------------------

img_size = env.telescope.img_size # number of pixels wide
pix_size = 3 # how many pixels wide each pixel in the observation is rendered as
buffer = 50
show_center = True
action = np.asarray([0, 12, 12], dtype=np.uint8)
controlling = 0
switch = False
rew = 0

def main_loop(FRAME):
    global obs, det, action, rew
    #print(action)
    observation, reward, terminated, _, det_dict = env.step(action)
    #print(observation.min())
    obs = observation[0]
    rew = reward
    #det = det_dict["detected"]
    c = env.telescope.true_centroids[0]
    #print(env.telescope._fp_to_uv(c[0], c[1]))
    #print(env.telescope.all_centroids_at_center())

def paint_loop(screen):
    global play, obs, rew

    font = pygame.font.SysFont('Times New Roman', 18)
    # text = "image"
    # text_surface = font.render(text, True, (255, 255, 255))
    # screen.blit(text_surface, (buffer + img_size * pix_size / 2 - font.size(text)[0] / 2, buffer / 2))

    # # image
    # pygame.draw.rect(screen, (255, 255, 255), (buffer - 1, buffer - 1, pix_size * img_size + 2, pix_size * img_size + 2), width = 2) # bounding box of image
    # for r in range(img_size):
    #     for c in range(img_size):
    #         col = obs[r][c]
    #         #print(col)
    #         pygame.draw.rect(screen, (col, col, col), (c * pix_size + buffer, r * pix_size + buffer, pix_size, pix_size)) # image

    text = "centroid locations"
    text_surface = font.render(text, True, (255, 255, 255))
    screen.blit(text_surface, (buffer + img_size * pix_size + buffer + img_size * pix_size / 2 - font.size(text)[0] / 2, buffer / 2))
    
    # centroids
    pygame.draw.rect(screen, (255, 255, 255), (buffer + pix_size * img_size + buffer - 1, buffer - 1, pix_size * img_size + 2, pix_size * img_size + 2), width = 2) # bounding box of detected
    for (fx, fy) in (env.telescope.true_centroids):
        u, v = env.telescope._fp_to_uv(fx, fy)
        x, y = u * pix_size + (buffer + pix_size * img_size + buffer), v * pix_size + buffer
        pygame.draw.rect(screen, (255, 255, 255), (x, y, pix_size, pix_size), width = 2) # bounding box of detected
    

    text = "Arrow Keys to move selected centroid"
    text_surface = font.render(text, True, (255, 255, 255))
    screen.blit(text_surface, (400 + buffer + img_size * pix_size + buffer + img_size * pix_size / 2 - font.size(text)[0] / 2, buffer / 2))
    text = "Space to change selected centroid"
    text_surface = font.render(text, True, (255, 255, 255))
    screen.blit(text_surface, (400 + buffer + img_size * pix_size + buffer + img_size * pix_size / 2 - font.size(text)[0] / 2, 18 + buffer / 2))
    text = "'R' to reset game"
    text_surface = font.render(text, True, (255, 255, 255))
    screen.blit(text_surface, (400 + buffer + img_size * pix_size + buffer + img_size * pix_size / 2 - font.size(text)[0] / 2, 35 + buffer / 2))
    text = f"Current score: {rew}"
    text_surface = font.render(text, True, (255, 255, 255))
    screen.blit(text_surface, (400 + buffer + img_size * pix_size + buffer + img_size * pix_size / 2 - font.size(text)[0] / 2, 35+18 + buffer / 2))
    
    
    # center
    if show_center:
        centerx, centery = env.telescope._fp_to_uv(env.telescope.center[0], env.telescope.center[1])
        xi, yi = centerx * pix_size + buffer, centery * pix_size + buffer
        xd, yd = centerx * pix_size + (buffer + pix_size * img_size + buffer), centery * pix_size + buffer
        pygame.draw.rect(screen, (100, 235, 50), (xi, yi, pix_size, pix_size), width = 2) # center of the image
        pygame.draw.rect(screen, (100, 235, 50), (xd, yd, pix_size, pix_size), width = 2) # center of the detected plane
    

def input_loop(keys, mouse, mouse_pos):
    global play, obs, action, up, switch, controlling
    if mouse[0]:
        x = (mouse_pos[0] - 50) / pix_size
        y = (mouse_pos[1] - 50) / pix_size
        fx, fy = env.telescope._uv_to_fp(x, y)
        diff = env.telescope.true_centroids - np.array([fx, fy])
        dists = np.sum(diff**2, axis=1)
        ind = np.argmin(dists)
        env.telescope.true_centroids[ind] = [fx, fy]
    
    vert = 12
    if keys[pygame.K_UP]:# and not up:
        vert += 12
    if keys[pygame.K_DOWN]:# and not up:
        vert -= 12

    horz = 12
    if keys[pygame.K_RIGHT]:# and not up:
        horz += 12
    if keys[pygame.K_LEFT]:# and not up:
        horz -= 12


    if keys[pygame.K_SPACE] and not switch:
        controlling = (controlling + 1) % env.n_panels
    
    if keys[pygame.K_SPACE]:
        switch = True
    else:
        switch = False
    
    if keys[pygame.K_r]:
        env.reset()

    action = np.asarray([controlling, horz, vert],dtype=np.uint8)


# -------------------------------------------------- background functionality -------------------------------------------------

# below makes the code function. It is not necessary to fully understand how it works,
# nor is there a need to change it
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(game_name)
clock = pygame.time.Clock()
run = True
while run:
    # fill the background with the background color
    screen.fill(background_color)

    main_loop(FRAME) # the current frame is passed to the main loop if you so choose to use it
    paint_loop(screen) # screen gets passed so that the print_loop function can draw on the screen
    input_loop(pygame.key.get_pressed(), pygame.mouse.get_pressed(), pygame.mouse.get_pos()) # a list of all inputs
    # the mouse_pressed input is a tuple containing: (left_click, middle_click, right_click)

    # increment the frame number. This is to keep track of what frame the game is on
    FRAME += 1
    clock.tick(FRAME_RATE) # keeps the game running at a max speed of 'FRAME_RATE' frames/second (might be slower if computations are heavy)
    # this makes sure everything is closed properly when the window is closed
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

pygame.quit()