import cv2
import numpy as np

# Load the wall and floor images
wall_image = cv2.imread('wall.png')
floor_image = cv2.imread('floor.png')

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
wall_cropped = wall_image[576:wall_bottom_white, :]  # Reduce 576 from top of wall
floor_cropped = floor_image[floor_top_white:, :]

# Combine the cropped wall and floor images
combined_image = np.vstack((wall_cropped, floor_cropped))

# Resize the combined image to 1920x1080
combined_image = cv2.resize(combined_image, (1920, 1080))

# Save the combined image
cv2.imwrite('combined_image.png', combined_image)

# Optional: Display the result if you're working locally
cv2.imshow("Combined Image", combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
