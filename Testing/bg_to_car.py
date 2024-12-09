import cv2
import numpy as np

# Load images
car_image = cv2.imread('car_with_black_background.png')
background = cv2.imread('combined_image.png')
car_mask_new = cv2.imread('car_mask_new.png', cv2.IMREAD_GRAYSCALE)

# Get dimensions
car_height, car_width = car_image.shape[:2]
bg_height, bg_width = background.shape[:2]

# Calculate padding needed on sides
total_width_padding = bg_width - car_width
side_padding = total_width_padding // 2

# Calculate top padding needed
top_padding = bg_height - car_height

# Create padded images with black background
padded_image = np.zeros((bg_height, bg_width, 3), dtype=np.uint8)
padded_mask = np.zeros((bg_height, bg_width), dtype=np.uint8)

# Place car image and mask in center with padding
y_offset = top_padding
x_offset = side_padding
padded_image[y_offset:y_offset+car_height, x_offset:x_offset+car_width] = car_image
padded_mask[y_offset:y_offset+car_height, x_offset:x_offset+car_width] = car_mask_new

# Use bitwise operation to replace black background of car with the actual background
car_gray = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)
_, car_mask = cv2.threshold(car_gray, 1, 255, cv2.THRESH_BINARY)
car_mask_inv = cv2.bitwise_not(car_mask)

# Extract the region of interest from the background
roi = background[y_offset:y_offset+car_height, x_offset:x_offset+car_width]

# Black-out the area of car in ROI
bg_with_car_area = cv2.bitwise_and(roi, roi, mask=car_mask_inv)

# Take only region of car from car image
car_only = cv2.bitwise_and(car_image, car_image, mask=car_mask)

# Add the car to the background
dst = cv2.add(bg_with_car_area, car_only)
background[y_offset:y_offset+car_height, x_offset:x_offset+car_width] = dst

# Save final outputs
cv2.imwrite('car_centered_on_background.png', background)
cv2.imwrite('car_mask_centered.png', padded_mask)

# Optional: Display result
cv2.imshow('Centered Car on Background', background)
cv2.waitKey(0)
cv2.destroyAllWindows()
