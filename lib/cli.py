import argparse
from deepdanbooru_tagger import DeepDanbooruTagger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images in a folder or a single image file.')
    parser.add_argument('input', type=str, help='Path to the folder containing images or a single image file')
    parser.add_argument('--size', type=int, default=10, help='Number of top tags to display')
    # New arguments for appending, prepending tags and enabling interrogation files
    parser.add_argument('--append', type=str, help='Comma-separated tags to append', default="")
    parser.add_argument('--prepend', type=str, help='Comma-separated tags to prepend', default="")
    parser.add_argument('--interrogate', action='store_true', help='Enable generating interrogate files')
    
    args = parser.parse_args()
    
    model_path = 'deepdanbooru-v4-20200814-sgd-e30/model-resnet_custom_v4.h5'
    tags_path = 'deepdanbooru-v4-20200814-sgd-e30/tags.txt'
    
    # Initialize the tagger with the model and tags
    tagger = DeepDanbooruTagger(model_path, tags_path)
    
    # Process the images in the folder or single image file
    tagger.process_images(args.input, args.size, args.append, args.prepend, args.interrogate)
