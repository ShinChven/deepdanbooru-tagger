# DeepDanbooru Tagger

## Introduction

DeepDanbooru Tagger is a program that tags images using the DeepDanbooru model. This tool supports command-line interface (CLI) for easy and efficient processing of images. Users can tag images, append or prepend custom tags, and optionally generate detailed interrogation files. It is especially useful for preparing training data for Stable Diffusion models.

## Installation

To get started, you can install the dependencies using pip:

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Model Download

You need to download the DeepDanbooru model from the following link:

[DeepDanbooru Model v4-20200814-sgd-e30](https://github.com/KichangKim/DeepDanbooru/releases/tag/v4-20200814-sgd-e30)

Unpack the downloaded model to a directory named `deepdanbooru-v4-20200814-sgd-e30`.

## Usage

### Use the Code

```python
from deepdanbooru_tagger import DeepDanbooruTagger

# Set the model path
model_path = 'deepdanbooru-v4-20200814-sgd-e30/model-resnet_custom_v4.h5'
tags_path = 'deepdanbooru-v4-20200814-sgd-e30/tags.txt'

# Initialize the tagger with the model and tags
tagger = DeepDanbooruTagger(model_path, tags_path)

# Arguments
input_path = '/path/to/image/or/folder'
size = 10  # Number of top tags to display. Note that if a `rating:` tag is predicted, it will be removed from the final tags. Consequently, you may end up with fewer tags than the specified size.
append_tags = "tag1,tag2"  # Comma-separated tags to append
prepend_tags = "tag3,tag4"  # Comma-separated tags to prepend
interrogate = True  # Enable generating interrogate files

# Process the images in the folder or single image file
tagger.process_images(input_path, size, append_tags, prepend_tags, interrogate)

```

### CLI

The program can be run using the command-line interface. Below is a basic example of how to use the tool:

```sh
python cli.py /path/to/image/or/folder --size 10 --append "tag1,tag2" --prepend "tag3,tag4" --interrogate
```

- `input`: Path to the folder containing images or a single image file.
- `--size`: Number of top tags to display (default: 10). Note that if a `rating:` tag is predicted, it will be removed from the final tags. Consequently, you may end up with fewer tags than the specified size.
- `--append`: Comma-separated tags to append (default: "").
- `--prepend`: Comma-separated tags to prepend (default: "").
- `--interrogate`: Enable generating interrogate files.

## Generated Files

When you run the program, it generates the following `.txt` files alongside the images:

1. **Tag Files**: For each image, a `.txt` file is generated containing the final list of tags. The file is named after the image file but with a `.txt` extension. For example, if the image file is `image.jpg`, the tag file will be `image.txt`.

    - The tags in this file are a combination of the predicted tags (excluding `rating:` tags), prepended tags, and appended tags.
    - Tags are separated by commas.

2. **Interrogate Files** (Optional): If the `--interrogate` option is enabled, an additional `.interrogate.txt` file is generated for each image. This file contains the predicted tags along with their confidence scores.

    - The file is named after the image file but with `.interrogate.txt` extension. For example, if the image file is `image.jpg`, the interrogation file will be `image.interrogate.txt`.
    - Each line in this file contains a tag and its corresponding confidence score in the format `tag: score`.

## Bash Script for Streamlined Usage

For experienced programmers, it is recommended to use a bash script to streamline the usage. Create a bash script with the following content:

```sh
#!/bin/bash
cd /path/to/repository
source venv/bin/activate
python lib/cli.py "$@"
```

This script will change to the repository directory, activate the virtual environment, and run the CLI with the provided arguments.

Here's an example of how you might use the script:

```sh
./tag_images.sh /path/to/image/or/folder --size 20 --append "tag1,tag2" --prepend "tag3,tag4" --interrogate
```

This will process the images in the specified directory or file, append and prepend the specified tags, and generate interrogation files.

## LICENSE

[LICENSE](./LICENSE)





