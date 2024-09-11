# triton-embed-image
A Triton Inference Server model repository for embedding images

## Image Embedding
Image embedding is a technique that transforms visual information from an image into a
compact numerical representation, typically in the form of a fixed-length vector. A
good representation captures essential features and characteristics of the image,
allowing for efficient processing and comparison of visual data in various machine
learning and computer vision tasks. Some common use cases for image embeddings include:

* Transfer learning for image classification
  * Allows for making smaller downstream models that need less labeled data
* Image search
  * Find images similar to a known starting image
  * Find images by giving textual descriptions
* Face recognition and verification
* Image clustering and categorization

The [embed_image](docs/embed_image.md) Triton Inference Server deployment allows
clients to send either the raw bytes of an image or a JSON request of the base64
encoded image. Current supported models:

* [SigLIP Vision](docs/siglip_vision.md) (default)

## Running Tasks
Running tasks is orchestrated by using [Taskfile.dev](https://taskfile.dev/)

# Taskfile Instructions

This document provides instructions on how to run tasks defined in the `Taskfile.yml`.  

Create a task.env at the root of project to define enviroment overrides. 

## Tasks Overview

The `Taskfile.yml` includes the following tasks:

- `triton-start`
- `triton-stop`
- `model-import`
- `build-execution-env-all`
- `build-*-env` (with options: `embed_text`, `multilingual_e5_large`, `siglip_text`)

## Task Descriptions

### `triton-start`

Starts the Triton server.

```sh
task triton-start
```

### `triton-stop`

Stops the Triton server.

```sh
task triton-stop
```

### `model-import`

Import model files from huggingface

```sh
task model-import
```

### `task build-execution-env-all`

Builds all the conda pack environments used by Triton

```sh
task build-execution-env-all
```

### `task build-*-env`

Builds specific conda pack environments used by Triton

```sh
#Example 
task build-siglip_vision-env
```

### `Complete Order`

Example of running multiple tasks to stage items needed to run Triton Server

```sh
task build-execution-env-all
task model-import
task triton-start
# Tail logs of running containr
docker logs -f $(docker ps -q --filter "name=triton-inference-server")
```
