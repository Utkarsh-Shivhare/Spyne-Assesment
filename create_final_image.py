import cv2
import numpy as np

def create_background(wall_path, floor_path):
    # Load images
    wall_image = cv2.imread(wall_path)
    floor_image = cv2.imread(floor_path)
    
    def find_white_boundary(image, from_bottom=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        height = gray.shape[0]
        
        for i in range(height-1, -1, -1) if from_bottom else range(height):
            if not np.all(gray[i] > 250):
                return i + 1 if from_bottom else i
        return 0

    # Crop and combine background
    wall_bottom_white = find_white_boundary(wall_image, from_bottom=True)
    floor_top_white = find_white_boundary(floor_image, from_bottom=False)
    
    wall_cropped = wall_image[576:wall_bottom_white, :]
    floor_cropped = floor_image[floor_top_white:, :]
    
    background = np.vstack((wall_cropped, floor_cropped))
    return cv2.resize(background, (1920, 1080))

def process_car_and_shadow(car_path, car_mask_path, shadow_mask_path, background):
    # Load images
    car_image = cv2.imread(car_path)
    car_mask = cv2.imread(car_mask_path, cv2.IMREAD_GRAYSCALE)
    shadow_mask = cv2.imread(shadow_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Find car contour and bottom point
    contours, _ = cv2.findContours(car_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    car_contour = max(contours, key=cv2.contourArea)
    bottom_point = np.max(car_contour[:, :, 1])
    
    # Crop car image with padding
    crop_bottom = min(bottom_point + 30, car_image.shape[0])
    car_cropped = car_image[:crop_bottom, :]
    car_mask_cropped = car_mask[:crop_bottom, :]
    
    # Resize car and mask to match background width while maintaining aspect ratio
    aspect_ratio = car_cropped.shape[1] / car_cropped.shape[0]
    new_height = int(background.shape[1] / aspect_ratio)
    car_resized = cv2.resize(car_cropped, (background.shape[1], new_height))
    car_mask_resized = cv2.resize(car_mask_cropped, (background.shape[1], new_height))
    
    # Create shadow gradient
    shadow_height = new_height
    gradient = np.linspace(0.8, 0.2, shadow_height)[:, np.newaxis]
    gradient = np.tile(gradient, (1, background.shape[1]))
    
    # Position car at the bottom center of background
    final_image = background.copy()
    start_y = background.shape[0] - new_height
    
    # Get the background slice that matches car dimensions
    bg_slice = final_image[start_y:start_y + new_height, :]
    
    # Ensure mask has same dimensions as bg_slice
    bg_mask = cv2.bitwise_not(car_mask_resized)
    
    # Debug prints to verify dimensions
    print(f"bg_slice shape: {bg_slice.shape}")
    print(f"bg_mask shape: {bg_mask.shape}")
    print(f"car_resized shape: {car_resized.shape}")
    
    # Extract regions with verified dimensions
    car_region = cv2.bitwise_and(car_resized, car_resized, mask=car_mask_resized)
    bg_region = cv2.bitwise_and(bg_slice, bg_slice, mask=bg_mask)
    
    # Apply shadow gradient
    shadow_overlay = np.zeros_like(car_region, dtype=np.float32)
    shadow_overlay = shadow_overlay + gradient[:, :, np.newaxis]
    
    # Combine car, shadow, and background
    combined_region = cv2.addWeighted(car_region, 1, bg_region, 1, 0)
    shadowed_region = cv2.multiply(combined_region.astype(np.float32), 
                                 shadow_overlay, 
                                 scale=1/255.0)
    
    # Place the final region in the background
    final_image[start_y:start_y + new_height, :] = shadowed_region.astype(np.uint8)
    
    return final_image

    # Input paths
wall_path = 'wall.png'
floor_path = 'floor.png'
car_path = '1.jpeg'
car_mask_path = '1_car.png'
shadow_mask_path = '1.png'

# Create background
background = create_background(wall_path, floor_path)

# Process car and shadow
final_image = process_car_and_shadow(car_path, car_mask_path, shadow_mask_path, background)

# Save output
cv2.imwrite('final_output.png', final_image)

# Display result
cv2.imshow('Final Result', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
