import cv2
import numpy as np
from tqdm import tqdm

    # Function to find white portion boundary
def bg_creation(wall_image,floor_image):
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
    Backgroundimage = cv2.resize(combined_image, (1920, 1080))

    # Save the combined image
    cv2.imwrite('combined_image.png', Backgroundimage)
    return Backgroundimage

def original_bg_removal(car_image,car_mask):
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
    car_mask_new=car_mask[:bottom_point+30, :]
    # Save the output image
    return cropped_image,car_mask_new
    cv2.imwrite('car_with_black_background.png', cropped_image)
    cv2.imwrite('car_mask_new.png', car_mask_new)


# Get dimension
def car_to_bg(car_image,car_mask_new,background):
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
    
    return background,padded_mask
    # Save final outputs
    cv2.imwrite('car_centered_on_background.png', background)
    cv2.imwrite('car_mask_centered.png', padded_mask)

    # Optional: Display result
    cv2.imshow('Centered Car on Background', background)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

# Normalize the shadow mask to range [0, 1] for blending
def shadow_insertion(car_image,car_mask,shadow_mask,background):
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
    # final once again evaluation 
    horizontal_range = range(best_x_shift-10, best_x_shift+10)  # Step size 10
    vertical_range = range(best_y_shift-10, best_y_shift+10)

    for x_shift in tqdm(horizontal_range, desc="Horizontal Shifts"):
        for y_shift in vertical_range:
            overlap = calculate_overlap(car_contour, shadow_contour, x_shift, y_shift)
            if overlap > max_overlap:
                max_overlap = overlap
                best_x_shift, best_y_shift = x_shift, y_shift
    # Apply the best shift
    translation_matrix = np.float32([[1, 0, best_x_shift], [0, 1, best_y_shift]])
    aligned_shadow = cv2.warpAffine(shadow_mask_normalized, translation_matrix, (car_mask.shape[1], car_mask.shape[0]))

    # Blend shadow with car image
    aligned_shadow_3d = cv2.merge([aligned_shadow] * 3)  # Create 3-channel shadow mask
    car_with_shadow = (car_image.astype(np.float32) / 255.0) * (1 - aligned_shadow_3d)  # Darken the car with shadow
    car_with_shadow = (car_with_shadow * 255).astype(np.uint8)  # Convert back to 8-bit
    cv2.imwrite('car_output_with_shadow_black.png', car_with_shadow)
    # Place composite on background
    return car_with_shadow

for i in range(1,7):
    car_image = cv2.imread(f'assignment\images\{i}.jpeg')
    car_mask = cv2.imread(f'assignment\car_masks\{i}.png', cv2.IMREAD_GRAYSCALE)
    shadow_mask = cv2.imread(f'assignment\shadow_masks\{i}.png', cv2.IMREAD_GRAYSCALE) 
    floor=cv2.imread(r'assignment\floor.png')
    wall=cv2.imread(r'assignment\wall.png')
    
    # Get original dimensions
    original_height, original_width = car_image.shape[:2]
    
    # Calculate resize ratios
    height_ratio = 1095 / original_height
    width_ratio = 1460 / original_width
    
    # Resize car image and masks
    car_image = cv2.resize(car_image, (1460, 1095))
    car_mask = cv2.resize(car_mask, (1460, 1095))
    
    # Get shadow mask dimensions and resize with same ratios
    shadow_height, shadow_width = shadow_mask.shape[:2]
    new_shadow_width = int(shadow_width * width_ratio)
    new_shadow_height = int(shadow_height * height_ratio)
    shadow_mask = cv2.resize(shadow_mask, (new_shadow_width, new_shadow_height))
    background = bg_creation(wall,floor)
    car_image,car_mask_new=original_bg_removal(car_image,car_mask)
    background,padded_mask=car_to_bg(car_image,car_mask_new,background)
    car_with_shadow=shadow_insertion(background,padded_mask,shadow_mask,background)
    cv2.imwrite(f'assignment\{i}.png', car_with_shadow)
