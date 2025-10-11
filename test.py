import cv2
import numpy as np


def draw_latitudes(image, center, radius, n_latitudes=9, color=(0, 0, 255), thickness=2):
    cx, cy = center
    for i in range(1, n_latitudes + 1):
        # latitude angle (avoid poles)
        phi = np.pi / 2 - (i * np.pi / (n_latitudes + 1))

        # vertical position
        y = int(cy + radius * np.sin(phi))

        # horizontal radius (ellipse width)
        rx = int(radius * np.cos(phi))

        # draw ellipse (lat line)
        cv2.ellipse(image, (cx, y), (rx, 0), 0, 0, 360, color, thickness)

    return image


# Create a blank image
size = 500
img = np.zeros((size, size, 3), dtype=np.uint8)

# Draw the sphere (white circle)
center = (size // 2, size // 2)
radius = 200
cv2.circle(img, center, radius, (255, 255, 255), -1)

# Draw latitudes
img = draw_latitudes(img, center, radius, n_latitudes=17)

cv2.imwrite("sphere_latitudes.png", img)
