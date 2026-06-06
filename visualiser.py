from neuralnetwork import *
import pygame
import numpy as np
import pickle

try:
    with open('mnist_nn_model.pkl', 'rb') as f:
        nn = pickle.load(f)
except FileNotFoundError:
    print("no model to load")
    exit()

pygame.init()
FONT = pygame.font.SysFont("monospace", 18)

CANVAS_SIZE = 280 
SIDEBAR_WIDTH = 240
WINDOW_WIDTH = CANVAS_SIZE + SIDEBAR_WIDTH
WINDOW_HEIGHT = CANVAS_SIZE

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Neural network visualiser")

canvas_surface = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
canvas_surface.fill((0, 0, 0))

drawing = False
last_pos = None
predictions = [0.0] * 10

def get_nn_prediction(high_res_surface):
    low_res_surface = pygame.transform.smoothscale(high_res_surface, (28, 28))
    rgb_pixels = pygame.surfarray.pixels3d(low_res_surface)
    gray_pixels = rgb_pixels[:, :, 0].T
    img_flattened = gray_pixels.astype(np.float32).reshape(1, 784) / 255.0
    pred_probabilities = nn.forward(img_flattened)
    return pred_probabilities[0]

running = True
while running:
    screen.fill((25, 25, 25))
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                drawing = True
                last_pos = pygame.mouse.get_pos()
            elif event.button == 3:
                canvas_surface.fill((0, 0, 0))
                predictions = [0.0] * 10
                
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
                last_pos = None
                
        elif event.type == pygame.MOUSEMOTION and drawing:
            mx, my = pygame.mouse.get_pos()
            if mx < CANVAS_SIZE and my < WINDOW_HEIGHT:
                current_pos = (mx, my)
                if last_pos is not None:
                    pygame.draw.line(canvas_surface, (255, 255, 255), last_pos, current_pos, 22)
                    pygame.draw.circle(canvas_surface, (255, 255, 255), current_pos, 11)
                last_pos = current_pos
                
                predictions = get_nn_prediction(canvas_surface)

    screen.blit(canvas_surface, (0, 0))
    
    pygame.draw.line(screen, (70, 70, 70), (CANVAS_SIZE, 0), (CANVAS_SIZE, WINDOW_HEIGHT), 3)
    
    for digit in range(10):
        prob_percentage = predictions[digit] * 100
        text_color = (200, 200, 200)
        text_suffix = ""
        
        if digit == np.argmax(predictions) and np.sum(predictions) > 0:
            text_color = (50, 255, 50)
            text_suffix = " <"
            
        text_surface = FONT.render(f"Digit {digit}: {prob_percentage:6.2f}%{text_suffix}", True, text_color)
        screen.blit(text_surface, (CANVAS_SIZE + 15, 12 + (digit * 25)))
        
    pygame.display.flip()

pygame.quit()