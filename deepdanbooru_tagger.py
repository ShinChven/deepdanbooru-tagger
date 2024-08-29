import os
import numpy as np
import tensorflow as tf
from PIL import Image

class DeepDanbooruTagger:
    def __init__(self, model_path, tags_path):
        self.model = tf.keras.models.load_model(model_path)
        with open(tags_path, 'r') as f:
            self.tags = f.read().splitlines()

    def is_image_file(self, file_path):
        try:
            Image.open(file_path)
            return True
        except IOError:
            return False

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        image = np.array(image) / 255.0
        return image

    def predict(self, image_path, size=20):
        image = self.preprocess_image(image_path)
        image = np.expand_dims(image, axis=0)
        predictions = self.model.predict(image)[0]
        top_indices = predictions.argsort()[-size:][::-1]
        
        results = []
        for i in top_indices:
            results.append((self.tags[i], predictions[i]))
        
        return results

    def process_images(self, input_path, size=20, append_tags=None, prepend_tags=None, interrogate=False):
        append_list = [tag.strip() for tag in append_tags.split(',')] if append_tags else []
        prepend_list = [tag.strip() for tag in prepend_tags.split(',')] if prepend_tags else []

        # Remove duplicates from append_list that are already in prepend_list
        append_list = [tag for tag in append_list if tag not in prepend_list]

        # Initialize list to store file paths
        image_files = []

        # Check if input_path is a file or folder
        if os.path.isfile(input_path) and self.is_image_file(input_path):
            image_files.append(input_path)
        elif os.path.isdir(input_path):
            # Loop through all files in the folder
            for file_name in os.listdir(input_path):
                image_path = os.path.join(input_path, file_name)
                if self.is_image_file(image_path):
                    image_files.append(image_path)

        # Process each image file
        for image_path in image_files:
            # Perform prediction
            predictions = self.predict(image_path, size)
                
            # Collect tags excluding those that start with 'rating:'
            tags_list = [tag for tag, score in predictions if not tag.startswith('rating:')]

            # Add prepend, detected, and append tags in the correct order, avoiding duplicates
            final_tags = prepend_list.copy()  # Start with prepended tags
            final_tags.extend(tag for tag in tags_list if tag not in prepend_list and tag not in append_list)
            final_tags.extend(append_list)  # Add appended tags

            # Remove the file extension for the output file names
            base_file_name = os.path.splitext(os.path.basename(image_path))[0]

            # Output tags in a txt file alongside the image
            tags_file_path = os.path.join(os.path.dirname(image_path), f'{base_file_name}.txt')
            with open(tags_file_path, 'w') as f:
                tags_string = ', '.join(final_tags)
                f.write(tags_string)
            
            # Optionally output results in a txt file if interrogate is enabled
            if interrogate:
                results_file_path = os.path.join(os.path.dirname(image_path), f'{base_file_name}.interrogate.txt')
                with open(results_file_path, 'w') as f:
                    for tag, score in predictions:
                        f.write(f'{tag}: {score:.4f}\n')
            
            # Print results
            print(f'Image: {image_path}')
            for tag, score in predictions:
                print(f'{tag}: {score:.4f}')
            print('-' * 50)
