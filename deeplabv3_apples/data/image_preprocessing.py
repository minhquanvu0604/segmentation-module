import os
import random
from typing import List
from PIL import Image


LABEL = (2, 2, 2)
BLACK = (25, 25, 25)
WHITE = (255, 255, 255)

def to_single_color(image_path, output_path, color=(128, 0, 128)):
    """
    Change all black pixels in the image to purple.
    Black color in the dataset is represented by (25, 25, 25).

    Parameters:
        image_path (str): Path to the input image.
        output_path (str): Path to save the modified image.
    """

    # Open the image
    image = Image.open(image_path).convert("RGB")
    pixels = image.load()

    mask_path = image_path.replace("images-Fuji", "labels-Fuji/PixelLabelData_18").replace("Image", "Label").replace(".jpg", ".png")
    mask_image = Image.open(mask_path).convert("RGB")
    mask_pixels = mask_image.load()

    # Iterate over each pixel and change black to purple
    for i in range(image.width):
        for j in range(image.height):
            # Skip the pixels that are part of the object
            if mask_pixels[i, j] == LABEL:
                continue
            if is_within_range(pixels[i, j], BLACK) or is_within_range(pixels[i, j], WHITE):
                pixels[i, j] = color

    # Save the modified image
    image.save(output_path)
    print(f"Modified image saved to {output_path}")

def to_background_image(image_path, background_images: List[str], output_path):
    # Open the input image
    input_image = Image.open(image_path).convert("RGB")
    input_pixels = input_image.load()

    mask_path = image_path.replace("images-Fuji", "labels-Fuji/PixelLabelData_18").replace("Image", "Label").replace(".jpg", ".png")
    mask_image = Image.open(mask_path).convert("RGB")
    mask_pixels = mask_image.load()

    # Get the size of the input image
    width, height = input_image.size

    if not background_images:
        print("No background images found in the specified folder.")
        return

    # Open a random background image
    background_image_path = random.choice(background_images)
    background_image = Image.open(background_image_path).convert("RGB")
    background_image = background_image.resize((width, height))
    background_pixels = background_image.load()

    # Replace black pixels in the input image
    for y in range(height):
        for x in range(width):

            # Skip the pixels that are part of the object
            if mask_pixels[x, y] == LABEL:
                continue

            # Check if the pixel is black
            if is_within_range(input_pixels[x, y], BLACK) or is_within_range(input_pixels[x, y], WHITE):
                # Replace with the corresponding pixel from the background image
                input_pixels[x, y] = background_pixels[x, y]

    # Save the modified image
    input_image.save(output_path)
    print(f"Modified image saved to {output_path}")

def is_within_range(pixel, target_color, tolerance=10):
    return all(target_color[i] - tolerance <= pixel[i] <= target_color[i] + tolerance for i in range(3))

def batch_synthesiser(dataset_dir, background_dir, output_dir, num_images):
    """
    Synthesizes images by replacing black pixels in the input images with pixels from random background images.

    Parameters:
        dataset_dir (str): Directory containing the dataset of input images.
        background_dir (str): Directory containing background images.
        output_dir (str): Directory where synthesized images will be saved.
        num_images (int): Number of synthesized images to generate.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the list of background images
    background_images = [
        os.path.join(background_dir, f) for f in os.listdir(background_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    if not background_images:
        raise FileNotFoundError("No background images found in the specified folder.")

    # Get the list of input images
    input_images = [
        os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    if not input_images:
        raise FileNotFoundError("No input images found in the specified folder.")

    # Loop through the images repeatedly to reach the specified number
    count = 0
    while count < num_images:
        for image_path in input_images:
            image_name = os.path.basename(image_path)
            if count >= num_images:
                break
            output_path = os.path.join(output_dir, f"synthesized_{count + 1}_from_{image_name}")
            to_background_image(image_path, background_images, output_path)
            count += 1

    print(f"Image synthesis completed. {num_images} images generated.")

if __name__ == '__main__':

    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Batch synthesiser
    background_path = r"D:\APPLE_DATA\fake_backgrounds"
    output_path = r"D:\APPLE_DATA\synthesised_images"
    dataset_dir = r"D:\APPLE_DATA\images-Fuji"


    # background_path = r"/data/minhqvu/APPLE_DATA/fake_backgrounds"
    # output_path = r"/data/minhqvu/APPLE_DATA/synthesised_images"
    # dataset_dir = r"/data/minhqvu/APPLE_DATA/images-Fuji" 
    num_images = 8000
    batch_synthesiser(dataset_dir, background_path, output_path, num_images)
