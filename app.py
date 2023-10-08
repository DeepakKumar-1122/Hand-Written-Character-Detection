import pygame, sys
from pygame.locals import *
import numpy as np
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt

Model = keras.models.load_model("D:\Projects\Hand Written Character Detection\model.h5")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

pygame.init()

LABELS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]

BOUNDARYINC = 5
WINDOWSIZEX = 1980
WINDOWSIZEY = 1080
DISPLAYSURFACE = pygame.display.set_mode((640, 480))
WHITE_INT = DISPLAYSURFACE.map_rgb(WHITE)
pygame.display.set_caption("Hand Writter Character Recognition")
IMAGESAVE = False
PREDICT = True
iswriting = False

number_xcord = []
number_ycord = []
img_cnt = 1

font = pygame.font.Font(None, 30)

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURFACE, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDARYINC, 0), min(
                WINDOWSIZEX, number_xcord[-1] + BOUNDARYINC
            )
            rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDARYINC, 0), min(
                WINDOWSIZEY, number_ycord[-1] + BOUNDARYINC
            )
            region_surface = DISPLAYSURFACE.subsurface(
                (
                    rect_min_x - 30,
                    rect_min_y - 30,
                    rect_max_x - rect_min_x + 60,
                    rect_max_y - rect_min_y + 60,
                )
            )
            number_xcord = []
            number_ycord = []
            img_arr = pygame.surfarray.array2d(region_surface)
            img_arr = np.flipud(img_arr)
            img_arr = np.rot90(img_arr, -1)
            img_arr1 = img_arr.astype(np.float32)
            # plt.imshow(img_arr, cmap="gray")
            # plt.show()

            if IMAGESAVE:
                cv2.imwrite("image.png")
                img_cnt += 1
            
            if PREDICT:
                # img = cv2.resize(img_arr, (28, 28))
                img = np.pad(img_arr1, (50, 50), "constant", constant_values=0)
                img = cv2.resize(img, (28, 28))
                plt.imshow(img, cmap="gray")
                plt.show()

                # Make sure LABELS is defined with your class labels before this point
                predictions = Model.predict(img.reshape(1, 28, 28, 1))

                # Print the predicted class probabilities
                # print("Predicted Probabilities:", predictions)

                # Get the predicted class label
                predicted_class_index = np.argmax(predictions)
                label = str(LABELS[predicted_class_index]).title()

                pygame.draw.rect(
                    DISPLAYSURFACE,
                    RED,
                    (
                        rect_min_x,
                        rect_min_y,
                        rect_max_x - rect_min_x,
                        rect_max_y - rect_min_y,
                    ),
                    3,
                )

                text = font.render(label, True, RED)
                text_rect = text.get_rect()
                text_rect.center = (
                    rect_min_x + (rect_max_x - rect_min_x) // 2,
                    rect_min_y - 10,
                )
                DISPLAYSURFACE.blit(text, text_rect)

        if event.type == KEYDOWN:
            if event.unicode == "n" or "U+2324":
                DISPLAYSURFACE.fill(BLACK)

        pygame.display.update()
