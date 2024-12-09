import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# Step 1: Load the Images
car_image = cv2.imread('1.jpeg')
car_mask = cv2.imread('1_car.png', cv2.IMREAD_GRAYSCALE)
shadow_mask = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)
background = cv2.imread('combined_image.png')

# Step 2: Extract Contours and Boundaries
# Extract the bottom boundary of the car
contours_car, _ = cv2.findContours(car_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
car_contour = max(contours_car, key=cv2.contourArea)
print(car_contour)
car_bottom_boundary = np.squeeze(car_contour[np.where(car_contour[:, :, 1] == np.max(car_contour[:, :, 1]))])

# Extract the top boundary of the shadow
contours_shadow, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
shadow_contour = max(contours_shadow, key=cv2.contourArea)
shadow_top_boundary = np.squeeze(shadow_contour[np.where(shadow_contour[:, :, 1] == np.min(shadow_contour[:, :, 1]))])

# Step 3: Dynamic Alignment for Maximum Overlap
def calculate_overlap(car_boundary, shadow_boundary, x_shift, y_shift):
    """Calculate the overlap between car and shadow boundaries after a shift."""
    shifted_shadow = shadow_boundary + np.array([x_shift, y_shift])
    overlap = np.array([point for point in car_boundary if any(np.array_equal(point, s_point) for s_point in shifted_shadow)])
    return len(overlap)

# Initialize variables for maximum overlap
best_x_shift, best_y_shift = 0, 0
max_overlap = 0

# Set iteration range based on image sizes
horizontal_range = range(200, 1460-1116)  # Full horizontal coverage
vertical_range = range(200, 1095-542)    # Full vertical coverage
print(car_bottom_boundary)
print(shadow_top_boundary)
# Try different shifts to maximize overlap
for x_shift in tqdm(horizontal_range, desc="Horizontal Shifts"):  # Horizontal range to test
    for y_shift in tqdm(vertical_range, desc="Vertical Shifts", leave=False):  # Vertical range to test
        overlap = calculate_overlap(car_contour, shadow_contour, x_shift, y_shift)
        if overlap > max_overlap:
            max_overlap = overlap
            best_x_shift, best_y_shift = x_shift, y_shift
        
        # Visualize the movement of the shadow during alignment
        translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        temp_aligned_shadow = cv2.warpAffine(shadow_mask, translation_matrix, (car_mask.shape[1], car_mask.shape[0]))
        temp_shadow_overlay = cv2.merge([temp_aligned_shadow] * 3)
        temp_visualization_with_shadow = cv2.addWeighted(car_image, 1, temp_shadow_overlay, 0.5, 0)
        
        # Display the current alignment step with max and current overlap values
        cv2.putText(temp_visualization_with_shadow, f'Max Overlap: {max_overlap}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(temp_visualization_with_shadow, f'Current Overlap: {overlap}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Alignment Progress", temp_visualization_with_shadow)
        cv2.waitKey(1)  # Short delay to visualize the movement

# Apply the best shift
translation_matrix = np.float32([[1, 0, best_x_shift], [0, 1, best_y_shift]])
aligned_shadow = cv2.warpAffine(shadow_mask, translation_matrix, (car_mask.shape[1], car_mask.shape[0]))

# Step 4: Visualize Contours and Alignment
visualization_image = car_image.copy()
cv2.drawContours(visualization_image, [car_contour], -1, (0, 255, 0), 2)  # Green for car
cv2.drawContours(visualization_image, [shadow_contour], -1, (0, 0, 255), 2)  # Red for shadow

# Overlay aligned shadow for debugging
shadow_overlay = cv2.merge([aligned_shadow] * 3)
visualization_with_shadow = cv2.addWeighted(visualization_image, 1, shadow_overlay, 0.5, 0)

# Step 5: Combine Car and Shadow
aligned_shadow_3d = cv2.merge([aligned_shadow] * 3)
car_with_shadow = cv2.addWeighted(car_image, 1, aligned_shadow_3d, 0.5, 0)

# Step 6: Place the Composite on the Background
bg_height, bg_width, _ = background.shape
car_resized = cv2.resize(car_with_shadow, (bg_width, bg_height))
final_output = cv2.addWeighted(background, 1, car_resized, 1, 0)

# Step 7: Display Results
plt.figure(figsize=(15, 10))

# Display Original Image with Contours
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(visualization_image, cv2.COLOR_BGR2RGB))
plt.title('Contours of Car and Shadow')
plt.axis('off')

# Display Aligned Shadow on Car
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
