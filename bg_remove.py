import cv2
import numpy as np

# Load the car image and mask
car_image = cv2.imread('1.jpeg')
car_mask = cv2.imread('1_car.png', cv2.IMREAD_GRAYSCALE)

# Ensure the mask is binary (black and white)
_, binary_mask = cv2.threshold(car_mask, 127, 255, cv2.THRESH_BINARY)

# Create an all-black background
black_background = np.zeros_like(car_image)

# Combine the car image with the black background using the mask
car_with_black_background = cv2.bitwise_and(car_image, car_image, mask=binary_mask)

# Replace the non-car area with black
inverse_mask = cv2.bitwise_not(binary_mask)
car_with_black_background += cv2.bitwise_and(black_background, black_background, mask=inverse_mask)

# Find contours in the binary mask
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assuming it's the car)
car_contour = max(contours, key=cv2.contourArea)

# Find the bottom-most point of the contour
bottom_point = np.max(car_contour[:, :, 1])

# Crop the image from top to the bottom point
cropped_image = car_with_black_background[:bottom_point+30, :]

# Save the output image
cv2.imwrite('car_with_black_background.png', cropped_image)

# Display the result (optional, if running in a local environment)
cv2.imshow("Car with Black Background", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
