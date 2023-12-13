import cv2
import numpy as np
from PIL import Image, ImageDraw


def create_longer_car_sprite(car_body, car_color, windshield_alpha=128, image_size=(200, 200)):
    """
    Sample usage;
    longer_car_body = [(70, 45), (140, 45), (155, 180), (55, 180)]
    car_color = (255, 0, 0)  # Red
    longer_car_sprite = create_longer_car_sprite(longer_car_body, car_color)
    longer_car_sprite = longer_car_sprite.rotate(180)
    longer_car_sprite.save('longer_car_sprite.png')
    """
    image = Image.new("RGBA", image_size, (255, 255, 255, 0))

    # Draw the car
    draw = ImageDraw.Draw(image)


    # Calculate tire radius and offset based on the car body size
    tire_radius = min(image_size) // 12  # 5% of the smaller dimension of the image
    tire_offset = tire_radius // 2       # Half the tire radius

    # Calculate tire positions dynamically
    tire_positions = []
    for i, corner in enumerate(car_body):
        x, y = corner
        if i in [0, 1]:  # Top corners, place tires below
            tire_positions.append((x - tire_radius / 2, y + tire_offset))
        else:  # Bottom corners, place tires above
            tire_positions.append((x - tire_radius / 2, y - tire_radius - tire_offset))

    for x, y in tire_positions[:2]:
        draw.ellipse((x, y, x + tire_radius, y + tire_radius), fill=(0, 0, 0))
    x, y = tire_positions[2]
    draw.ellipse((x, y, (x + tire_radius) * 0.98, y + tire_radius), fill=(0, 0, 0))
    x, y = tire_positions[3]
    draw.ellipse((x * 1.09, y, x + tire_radius, y + tire_radius), fill=(0, 0, 0))

    # Draw car body
    draw.polygon(car_body, fill=car_color)

    # Calculate and adjust windshield positions dynamically, closer to the center
    windshield_shift = min(image_size) // 10  # Shift for windshield position
    front_front_multiplier = 0.3  # Multiplier for the front windshield's front side
    front_rear_multiplier = 0.8   # Multiplier for the front windshield's rear side
    front_windshield_y_offset = 15  # Offset for the front windshield's y position

    front_windshield = [
        (car_body[0][0] + windshield_shift * front_front_multiplier, car_body[0][1] + windshield_shift + front_windshield_y_offset),
        (car_body[1][0] - windshield_shift * front_front_multiplier, car_body[1][1] + windshield_shift + front_windshield_y_offset),
        (car_body[1][0] - windshield_shift * front_rear_multiplier, car_body[1][1] + windshield_shift * 2 + front_windshield_y_offset),
        (car_body[0][0] + windshield_shift * front_rear_multiplier, car_body[0][1] + windshield_shift * 2 + front_windshield_y_offset)
    ]

    rear_windshield_y_offset = 0  # Offset for the rear windshield's y position
    rear_windshield = [
        (car_body[2][0] - windshield_shift * 1.5, car_body[2][1] - windshield_shift * 2 + rear_windshield_y_offset),
        (car_body[3][0] + windshield_shift * 1.5, car_body[3][1] - windshield_shift * 2 + rear_windshield_y_offset),
        (car_body[3][0] + windshield_shift, car_body[3][1] - windshield_shift - rear_windshield_y_offset),
        (car_body[2][0] - windshield_shift, car_body[2][1] - windshield_shift - rear_windshield_y_offset)
    ]

    windshield_color = (car_color[0], car_color[1], car_color[2], windshield_alpha)
    for windshield in [front_windshield, rear_windshield]:
        draw.polygon(windshield, fill=windshield_color)

    # Calculate and draw lights dynamically
    light_size = (tire_radius, tire_radius // 2)  # Width and height of lights
    light_offset = tire_offset * 1.5
    lights = [
        (car_body[0][0] - light_size[0] / 2 + light_offset, car_body[0][1] - light_size[1] / 2),
        (car_body[1][0] - light_size[0] / 2 - light_offset, car_body[1][1] - light_size[1] / 2)
    ]

    for x, y in lights:
        draw.ellipse((x, y, x + light_size[0], y + light_size[1]), fill=(255, 255, 0))

    return image


def process_light(image_path='sprites/lights/original.png', light_color=(0, 0, 255), fill_color='red'):
    """
    Sample usage:
    processed_img, output_path = process_light(light_color=(0, 0, 255), fill_color='red')
    """
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Convert the image to binary
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    thresh = cv2.convertScaleAbs(thresh, alpha=(1.0/255.0))

    # Find contours
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contours_to_fill = contours[0:3]

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Find all black pixels (assuming black is [0, 0, 0])
    black_pixels_mask = np.all(img == [0, 0, 0], axis=-1)

    # Paint all the black pixels in the image to light_color
    img[black_pixels_mask] = light_color

    # Determine the fill color for the contours
    if fill_color.lower() == 'green':
        contour_color = (0, 255, 0)
    else:
        contour_color = (255, 0, 0)  # Default to red

    # Flood fill the contours
    for contour in contours_to_fill:
        cv2.drawContours(img, [contour], contourIdx=0, color=contour_color, thickness=cv2.FILLED)

    # Add an alpha channel to the image
    alpha_channel = np.ones(img.shape[:2], dtype=img.dtype) * 255  # Create an all-white alpha channel

    # Make white pixels transparent (assuming white is [255, 255, 255])
    white_pixels_mask = np.all(img == [255, 255, 255], axis=-1)
    alpha_channel[white_pixels_mask] = 0  # Make white pixels fully transparent

    img_rgba = np.dstack((img, alpha_channel))

    # Save the processed image with transparency
    new_file_name = f"light_{str(light_color)}_{'red' if fill_color.lower() == 'red' else 'green'}.png"
    output_path = f"{'/'.join(image_path.split('/')[:-1])}/{new_file_name}"
    cv2.imwrite(output_path, cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGRA))

    return img_rgba, output_path

def process_boulder(image_path='sprites/boulders/original.png', body_color=(0, 155, 150)):
    img = ~cv2.imread(image_path, cv2.IMREAD_UNCHANGED)[..., 3]
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Find all black pixels (assuming black is [0, 0, 0])
    black_pixels_mask = np.all(img == [0, 0, 0], axis=-1)

    img[black_pixels_mask] = body_color
    
    # Make background transparent
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    img[..., 3][~black_pixels_mask] = 0


    new_file_name = f"boulder_{str(body_color)}.png"
    output_path = f"{'/'.join(image_path.split('/')[:-1])}/{new_file_name}"
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGRA))

    return img, output_path

def process_car_image(image_path='sprites/cars/original.png', car_color=(255, 0, 255), light_color=(255, 255, 0)):
    # Convert car_color and light_color to BGR
    car_color = car_color[::-1]
    light_color = light_color[::-1]
    # Read the image with alpha channel
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Paint all transparent pixels to white
    img[img[..., 3] == 0] = (255, 255, 255, 255)
    
    # Binarize the image
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    
    # Convert to greyscale
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    
    # Find contours
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    
    # Assuming the contours of interest are always at indices 8 and 9
    contours_to_fill = contours[3:5]
    mask_contour = contours[1:2]
    
    cv2.drawContours(mask, mask_contour, contourIdx=-1, color=255, thickness=cv2.FILLED)
    # Flood fill the contours to paint the lights
    for contour in contours_to_fill:
        cv2.drawContours(img, [contour], contourIdx=0, color=light_color + (255,), thickness=cv2.FILLED)
    
    img[..., 3] = mask
    # Create a mask to identify the black part of the car (i.e., the body)
    black_mask = np.all(img[..., 0:3] == (0, 0, 0), axis=-1) & (img[..., 3] == 255)
    
    # Paint the black part with the provided car color
    img[black_mask] = car_color + (255,)

    # Convert image to display with matplotlib (BGR to RGB)
    img_display = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    # Save the processed image
    new_file_name = f"car_{str(car_color[::-1])}_{str(light_color[::-1])}.png"
    output_path = f"{'/'.join(image_path.split('/')[:-1])}/{new_file_name}"
    cv2.imwrite(output_path, img)

    return img, output_path


def make_sprites(car_colors, light_colors, boulder_colors, car_light_colors=(255, 255, 0)):
    for car_color in car_colors:
        process_car_image(car_color=car_color, light_color=car_light_colors)
    
    for light_color in light_colors:
        process_light(light_color=light_color, fill_color='red')
        process_light(light_color=light_color, fill_color='green')
        
    for boulder_color in boulder_colors:
        process_boulder(body_color=boulder_color)

if __name__ == '__main__':
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 255, 255) # White
    ]
    car_colors = colors[:3] + colors[4:]
    light_colors = colors[2:]

    make_sprites(car_colors, light_colors, colors)