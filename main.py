import os
import argparse

import numpy as np
import tensorflow as tf
from PIL import Image

def is_image_file(file_path):
    try:
        Image.open(file_path)
        return True
    except IOError:
        return False

model_path = 'deepdanbooru-v4-20200814-sgd-e30/model-resnet_custom_v4.h5'
tags_path = 'deepdanbooru-v4-20200814-sgd-e30/tags.txt' 

# Load the model from the local directory
model = tf.keras.models.load_model(model_path)

# Load tags from the local directory
with open(tags_path, 'r') as f:
    tags = f.read().splitlines()

# Preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    image = np.array(image) / 255.0
    return image

# Predict tags
def predict(image_path, size=20):
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)[0]
    top_indices = predictions.argsort()[-size:][::-1]
    
    results = []
    for i in top_indices:
        results.append((tags[i], predictions[i]))
    
    return results

def main(folder_path, size=20, append_tags=None, prepend_tags=None, interrogate=False):
    append_list = [tag.strip() for tag in append_tags.split(',')] if append_tags else []
    prepend_list = [tag.strip() for tag in prepend_tags.split(',')] if prepend_tags else []

    # Remove duplicates from append_list that are already in prepend_list
    append_list = [tag for tag in append_list if tag not in prepend_list]

    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)
        if is_image_file(image_path):
            predictions = predict(image_path, size)
            
            # Collect tags excluding those that start with 'rating:'
            tags_list = [tag for tag, score in predictions if not tag.startswith('rating:')]

            # Add prepend, detected, and append tags in the correct order, avoiding duplicates
            final_tags = prepend_list.copy()  # Start with prepended tags
            final_tags.extend(tag for tag in tags_list if tag not in prepend_list and tag not in append_list)
            final_tags.extend(append_list)  # Add appended tags

            # Remove the file extension for the output file names
            base_file_name = os.path.splitext(file_name)[0]

            # Output tags in a txt file alongside the image
            tags_file_path = os.path.join(folder_path, f'{base_file_name}.txt')
            with open(tags_file_path, 'w') as f:
                tags_string = ', '.join(final_tags)
                f.write(tags_string)
            
            # Optionally output results in a txt file if interrogate is enabled
            if interrogate:
                results_file_path = os.path.join(folder_path, f'{base_file_name}.interrogate.txt')
                with open(results_file_path, 'w') as f:
                    for tag, score in predictions:
                        f.write(f'{tag}: {score:.4f}\n')
            
            # Print results
            print(f'Image: {image_path}')
            for tag, score in predictions:
                print(f'{tag}: {score:.4f}')
            print('-' * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images in a folder.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing images')
    parser.add_argument('--size', type=int, default=10, help='Number of top tags to display')
    
    # New arguments for appending, prepending tags and enabling interrogation files
    parser.add_argument('--append', type=str, help='Comma-separated tags to append', default="")
    parser.add_argument('--prepend', type=str, help='Comma-separated tags to prepend', default="")
    parser.add_argument('--interrogate', action='store_true', help='Enable generating interrogate files')

    args = parser.parse_args()
    main(args.folder_path, args.size, args.append, args.prepend, args.interrogate)
