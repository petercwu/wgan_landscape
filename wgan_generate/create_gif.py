import os
import re
import imageio.v2 as imageio

# Takes generated images from models of each epoch and compiles them to create a GIF file showing showing the progression from epoch 1-500.

image_folder = "INSERT IMAGE FOLDER PATH THERE"
image_files= [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(".png")]

def sort_by_epoch(img_file):
    return [int(num) for num in re.findall(r"\d", img_file)]

image_files.sort(key=sort_by_epoch)
images = [imageio.imread(file) for file in image_files]
gif_path = "wgan-gp_generated_images.gif"
imageio.mimsave(gif_path, images, duration=40)