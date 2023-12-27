import cv2
import numpy as np
x1 = 5
y1 = 5

x2 = 100
y2 = 50

img = np.zeros((400, 400, 3), dtype=np.uint8)

top_left = (x1, y1)
bottom_right = (x2, y2)

distance = np.sqrt(((x2 - x1) ** 2) - ((y2 - y1) ** 2))
print(f'distance: {distance}')

mid_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
print(f'mid point: {mid_point}')
colors = (0, 0, 255)
cv2.circle(img, mid_point, 2, color=colors)
# cv2.imshow('rectangle', img)
cv2.line(img, int(abs(52 - distance)), int(abs(27 - distance)), color=(0, 255, 0), thickness=2)

cv2.rectangle(img, top_left, bottom_right, color=(255, 0, 0))
cv2.imshow('rectangle', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

