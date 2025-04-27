import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

# This file is not related to the mnist dataset, but is used to generate images of circles with bounding boxes.
# could be useful for training a neural network to detect circles in images.

def generate_circle_image(img_size=128, min_radius=10, max_radius=30, n_circles=1):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    boxes = []
    
    for _ in range(n_circles):
        radius = random.randint(min_radius, max_radius)
        x = random.randint(radius, img_size - radius)
        y = random.randint(radius, img_size - radius)
        color = 255  # white circle on black bg
        cv2.circle(img, (x, y), radius, color, -1)
        boxes.append((x - radius, y - radius, x + radius, y + radius))  # [x_min, y_min, x_max, y_max]

    return img, boxes

def visualize(img, boxes):
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green box
    plt.imshow(img_color)
    plt.axis('off')
    plt.show()

# Example: Generate one image and show it
image, bounding_boxes = generate_circle_image(n_circles=100, img_size=2048)
visualize(image, bounding_boxes)
