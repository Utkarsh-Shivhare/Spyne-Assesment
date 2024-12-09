import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the images
car_image = cv2.imread('car_centered_on_background.png')
car_mask = cv2.imread('car_mask_centered.png', cv2.IMREAD_GRAYSCALE)
shadow_mask = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)  # Load the shadow mask
background = cv2.imread('combined_image.png')

# Normalize the shadow mask to range [0, 1] for blending
shadow_mask_normalized = shadow_mask / 255.0

# Extract the bottom boundary of the car
contours_car, _ = cv2.findContours(car_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
car_contour = max(contours_car, key=cv2.contourArea)

# Extract the top boundary of the shadow
contours_shadow, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
shadow_contour = max(contours_shadow, key=cv2.contourArea)

# Define proximity-based overlap calculation function
def calculate_overlap(car_boundary, shadow_boundary, x_shift, y_shift, max_distance=5):
    """Calculate the overlap between car and shadow boundaries after a shift."""
    shifted_shadow = shadow_boundary + np.array([x_shift, y_shift])
    car_points = car_boundary.reshape(-1, 2)
    shadow_points = shifted_shadow.reshape(-1, 2)

    distances = np.sqrt(np.sum((car_points[:, None, :] - shadow_points[None, :, :]) ** 2, axis=2))
    overlap_count = np.sum(np.any(distances <= max_distance, axis=1))

    return overlap_count

# Align for maximum overlap
best_x_shift, best_y_shift = 0, 0
max_overlap = 0
horizontal_range = range(200, car_mask.shape[1] - shadow_mask.shape[1], 10)  # Step size 10
vertical_range = range(100, car_mask.shape[0]-30, 10)

for x_shift in tqdm(horizontal_range, desc="Horizontal Shifts"):
    for y_shift in vertical_range:
        overlap = calculate_overlap(car_contour, shadow_contour, x_shift, y_shift)
        if overlap > max_overlap:
            max_overlap = overlap
            best_x_shift, best_y_shift = x_shift, y_shift
#final once again evaluation 
# horizontal_range = range(best_x_shift-10, best_x_shift+10)  # Step size 10
# vertical_range = range(best_y_shift-10, best_y_shift+10)

# for x_shift in tqdm(horizontal_range, desc="Horizontal Shifts"):
#     for y_shift in vertical_range:
#         overlap = calculate_overlap(car_contour, shadow_contour, x_shift, y_shift)
#         if overlap > max_overlap:
#             max_overlap = overlap
#             best_x_shift, best_y_shift = x_shift, y_shift
# Apply the best shift
translation_matrix = np.float32([[1, 0, best_x_shift], [0, 1, best_y_shift]])
aligned_shadow = cv2.warpAffine(shadow_mask_normalized, translation_matrix, (car_mask.shape[1], car_mask.shape[0]))

# Blend shadow with car image
aligned_shadow_3d = cv2.merge([aligned_shadow] * 3)  # Create 3-channel shadow mask
car_with_shadow = (car_image.astype(np.float32) / 255.0) * (1 - aligned_shadow_3d)  # Darken the car with shadow
car_with_shadow = (car_with_shadow * 255).astype(np.uint8)  # Convert back to 8-bit
cv2.imwrite('car_output_with_shadow_black.png', car_with_shadow)
# Place composite on background
bg_height, bg_width, _ = background.shape
car_resized = cv2.resize(car_with_shadow, (bg_width, bg_height))
final_output = cv2.addWeighted(background, 1, car_resized, 1, 0)

# Display results
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(car_image, cv2.COLOR_BGR2RGB))
plt.title('Original Car Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor((aligned_shadow_3d * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.title('Aligned Shadow Mask (Black)')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(car_with_shadow, cv2.COLOR_BGR2RGB))
plt.title('Car with Shadow Applied')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))
plt.title('Final Output with Background')
plt.axis('off')

cv2.imwrite('final_output_with_shadow_black.png', final_output)
plt.show()

cv2.destroyAllWindows()
