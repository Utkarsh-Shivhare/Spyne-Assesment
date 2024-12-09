import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Step 1: Load the Images
car_image = cv2.imread('car_centered_on_background.png')
car_mask = cv2.imread('1_car.png', cv2.IMREAD_GRAYSCALE)
shadow_mask = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)
background = cv2.imread('combined_image.png')

# Step 2: Extract Contours and Boundaries
# Convert car image to grayscale for contour detection
car_gray = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)
_, car_binary = cv2.threshold(car_gray, 1, 255, cv2.THRESH_BINARY)

# Extract the bottom boundary of the car
contours_car, _ = cv2.findContours(car_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
car_contour = max(contours_car, key=cv2.contourArea)

# Extract the top boundary of the shadow
contours_shadow, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
shadow_contour = max(contours_shadow, key=cv2.contourArea)

# Draw contours on images for visualization
car_with_contour = car_image.copy()
cv2.drawContours(car_with_contour, [car_contour], -1, (0, 255, 0), 2)
cv2.imshow('Car with Contour', car_with_contour)

shadow_with_contour = cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR)
cv2.drawContours(shadow_with_contour, [shadow_contour], -1, (0, 255, 0), 2)
cv2.imshow('Shadow with Contour', shadow_with_contour)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 3: Proximity-Based Overlap Calculation
def calculate_overlap(car_boundary, shadow_boundary, x_shift, y_shift, max_distance=5):
    """Calculate the overlap between car and shadow boundaries after a shift."""
    shifted_shadow = shadow_boundary + np.array([x_shift, y_shift])

    # Convert to NumPy arrays
    car_points = car_boundary.reshape(-1, 2)
    shadow_points = shifted_shadow.reshape(-1, 2)

    # Vectorized proximity check
    distances = np.sqrt(np.sum((car_points[:, None, :] - shadow_points[None, :, :]) ** 2, axis=2))
    overlap_count = np.sum(np.any(distances <= max_distance, axis=1))

    return overlap_count

# Step 4: Align for Maximum Overlap
best_x_shift, best_y_shift = 0, 0
max_overlap = 0

# Define iteration ranges
horizontal_range = range(0, car_mask.shape[1] - shadow_mask.shape[1], 10)  # Step size 10
vertical_range = range(0, car_mask.shape[0] - shadow_mask.shape[0], 10)

# Iterate to find the best alignment
for x_shift in tqdm(horizontal_range, desc="Horizontal Shifts"):
    for y_shift in vertical_range:
        overlap = calculate_overlap(car_contour, shadow_contour, x_shift, y_shift)
        if overlap > max_overlap:
            max_overlap = overlap
            best_x_shift, best_y_shift = x_shift, y_shift

        # Visualize each step
        translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        temp_aligned_shadow = cv2.warpAffine(shadow_mask, translation_matrix, (car_mask.shape[1], car_mask.shape[0]))
        temp_shadow_overlay = cv2.merge([temp_aligned_shadow] * 3)
        temp_visualization = cv2.addWeighted(car_image, 1, temp_shadow_overlay, 0.5, 0)

        # Add text for current and max overlap
        cv2.putText(temp_visualization, f'Max Overlap: {max_overlap}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(temp_visualization, f'Current Overlap: {overlap}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display progress
        # cv2.imshow("Alignment Progress", temp_visualization)
        # cv2.waitKey(1)

# Apply the best shift
translation_matrix = np.float32([[1, 0, best_x_shift], [0, 1, best_y_shift]])
aligned_shadow = cv2.warpAffine(shadow_mask, translation_matrix, (car_mask.shape[1], car_mask.shape[0]))

# Create gradient mask for shadow blending
gradient_height = aligned_shadow.shape[0] // 3  # Use top third for gradient
gradient = np.linspace(1, 0, gradient_height)[:, np.newaxis]
gradient = np.tile(gradient, (1, aligned_shadow.shape[1]))
full_gradient = np.ones_like(aligned_shadow, dtype=np.float32)
full_gradient[:gradient_height] = gradient
aligned_shadow_with_gradient = (aligned_shadow.astype(np.float32) * full_gradient).astype(np.uint8)

# Step 5: Visualize Contours and Alignment
visualization_image = car_image.copy()
cv2.drawContours(visualization_image, [car_contour], -1, (0, 255, 0), 2)  # Green for car
cv2.drawContours(visualization_image, [shadow_contour], -1, (0, 0, 255), 2)  # Red for shadow

# Overlay aligned shadow for debugging
shadow_overlay = cv2.merge([aligned_shadow_with_gradient] * 3)
visualization_with_shadow = cv2.addWeighted(visualization_image, 1, shadow_overlay, 0.5, 0)

# Step 6: Combine Car and Shadow
aligned_shadow_3d = cv2.merge([aligned_shadow_with_gradient] * 3)
car_with_shadow = cv2.addWeighted(car_image, 1, aligned_shadow_3d, 0.7, 0)  # Increased shadow intensity

# Step 7: Place Composite on Background
bg_height, bg_width, _ = background.shape
car_resized = cv2.resize(car_with_shadow, (bg_width, bg_height))
final_output = cv2.addWeighted(background, 1, car_resized, 1, 0)

# Step 8: Display Results
plt.figure(figsize=(15, 10))

# Display Original Contours
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(visualization_image, cv2.COLOR_BGR2RGB))
plt.title('Contours of Car and Shadow')
plt.axis('off')

# Display Alignment Progress
plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(visualization_with_shadow, cv2.COLOR_BGR2RGB))
plt.title('Aligned Shadow Overlay on Car')
plt.axis('off')

# Display Final Composite
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))
plt.title('Final Output with Background')
plt.axis('off')

# Save Final Output
cv2.imwrite('final_output_with_shadow.png', final_output)
plt.show()

# Close the visualization window
cv2.destroyAllWindows()
