import cv2
import numpy as np

# Load the wall and floor images
wall_image = cv2.imread('wall.png')
floor_image = cv2.imread('floor.png')

# Define target size
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

# Based on the reference image analysis:
# Wall takes about 65% of height (more visible wall)
# Floor takes about 35% of height (less floor visible)
# Only center ~80% width is used to match car showroom style
WALL_HEIGHT_RATIO = 0.65
FLOOR_HEIGHT_RATIO = 0.35
WIDTH_CROP_RATIO = 0.8

# Function to find white portion boundary
def find_white_boundary(image, from_bottom=False):
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    height = gray.shape[0]
    if from_bottom:
        # Scan from bottom to top
        for i in range(height-1, -1, -1):
            if not np.all(gray[i] > 250):  # Check if row is not all white
                return i + 1
    else:
        # Scan from top to bottom
        for i in range(height):
            if not np.all(gray[i] > 250):  # Check if row is not all white
                return i
    return 0

# Find where white portions end
wall_bottom_white = find_white_boundary(wall_image, from_bottom=True)
floor_top_white = find_white_boundary(floor_image, from_bottom=False)

# Crop the images
wall_cropped = wall_image[:wall_bottom_white, :]
floor_cropped = floor_image[floor_top_white:, :]

# Calculate target dimensions
target_wall_height = int(TARGET_HEIGHT * WALL_HEIGHT_RATIO)
target_floor_height = int(TARGET_HEIGHT * FLOOR_HEIGHT_RATIO)
target_crop_width = int(TARGET_WIDTH * WIDTH_CROP_RATIO)

# Resize the cropped images to match target dimensions
wall_resized = cv2.resize(wall_cropped, (target_crop_width, target_wall_height))
floor_resized = cv2.resize(floor_cropped, (target_crop_width, target_floor_height))

# Combine the resized wall and floor images
combined_image = np.vstack((wall_resized, floor_resized))

# Create a white canvas of target size
final_image = np.ones((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8) * 255

# Calculate padding for centering
x_offset = (TARGET_WIDTH - target_crop_width) // 2
y_offset = 0

# Place the combined image in the center of the white canvas
final_image[y_offset:y_offset+combined_image.shape[0], 
           x_offset:x_offset+combined_image.shape[1]] = combined_image

# Save the combined image
cv2.imwrite('combined_image.png', final_image)

# Optional: Display the result if you're working locally
cv2.imshow("Combined Image", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
