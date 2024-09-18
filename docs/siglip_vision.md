# SigLIP Vision
This deployment is a
[Triton Inference Server ensemble](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html#ensemble-models)
that serves the [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384)
vision model. It takes in the raw image bytes and returns the embedding vector (d=1152)
that can be used for zero/few-shot image classification.

The ensemble is made up of the following stages. Code is available in the
[model_repository](../model_repository) directory:
* siglip_vision_process: Takes the raw image bytes and returns the RGB pixel values
  for images that have been resized to 384x384. This is done on the CPU
* siglip_vision: The vision model that takes in the RGB pixel values for 384x384 images
  and returns the embedding vector (d=1152). This is done on the GPU

Dynamic batching is enabled for this deployment, so clients simply send in a single
image to be embedded.

This is a lower level of abstraction, most clients likely should be using
[embed_image](embed_image.md) deployment.

## Example Request
Requests work the same as the [embed_image](embed_image.md) deployment except that
there is no optional base64 encoded image option.

## Performance Analysis
Triton Inference Server's perf_analyzer does not seem to support sending in raw
bytes easily. See [embed_image](embed_image.md) for performance testing.

## Validation
To validate that the model is performing as expected, we use some data from
[ImageNet](https://www.kaggle.com/competitions/imagenet-object-localization-challenge).
The training data was nicely organized into subdirectories with each subdirectory
named after the Synset category and with each file name in a give subdirectory also
containing the {synset}_{file_id}.JPEG.

Working with images from the training data set, I put 10 images for each of the 1,000
categories into `train/{synset}` directory on my local machine. An additional
20 images for each of the 1,000 categories were placed into `valid/{synset}`.

```
train/
  - n01440764/
    - n01440764_3198.JPEG
    - n01440764_3199.JPEG
    - ...
  - n01443537/
    - n01443537_428.JPEG
    - ...
  - ...
```

In addition to the subset of images, I also downloaded the LOC_synset_mapping.txt. This
contains the synset category label and a description of the category. This data will be
used for performing the zero-shot accuracy validation. Here is the first
few lines:

| Label | Text Description |
| :----: | :-----------|
| n01440764 | tench, Tinca tinca |
| n01443537 | goldfish, Carassius auratus |
| n01484850 | great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias |
| n01491361 | tiger shark, Galeocerdo cuvieri |
| n01494475 | hammerhead, hammerhead shark |

### 10-Shot Training of KNN Classifier
As a first check, we will use the training images (10 images per category x 1000
categories) to create a [KNN Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier).
The training images are embedded using the [embed_image][embed_image.md] Triton
Inference Server deployed endpoint. These training image embeddings and their
corresponding category labels were used to fit the classifier.

The validation images are used to measure the performance. Each validation image is
also sent to be embedded. For each image, the classifier finds the 10-nearest training
images. A prediction for the classification of the validation image is based upon the
category of the 10 nearest neighbors and their distance to the validation image. Both
the top-1 accuracy (i.e., was the true label of the validation image the top predicted
class from the nearest neighbors) and also the top-5 accuracy (was the true label in
among the top-5 predicted classes). Results are in the table below.

### Zero-Shot KNN Classifier
The [SigLIP paper](https://arxiv.org/abs/2303.15343) uses zero-shot to measure the
quality of their embedding model. For zero-shot, you use a text description of the
category and embed that using the
[SiglipTextModel](https://huggingface.co/docs/transformers/en/model_doc/siglip#transformers.SiglipTextModel).
This text embedding of the category is what is used to fit the KNN Classifier. After
that, we do the same as before. Taking each validation image embedding, get the 
10 nearest neighbors (where a neighbor now is a text embedding of a category), and
use the neighbors' corresponding category label to predict the classification of the
validation image. We calculate both the top-1 and top-5 accuracy.

The SigLIP paper claims an ImageNet accuracy of 83.2% on the validation data of
ImageNet. They paper notes some tweak to the prompts and a few other details to
improve peformance. The numbers quoted below show a few different variations of
prompting templates and demonstrate comparable accuracy to the paper.

Interesting to note significant improvement in these scores when `padding="max_length"`
is set when calling the processor to tokenize the text. I have no explanation why,
but the Huggingface model card does explicitly call this out. Without padding, top-1
accuracy falls from 0.8194 -> 0.5142 and top-5 accuracy falls from 0.9630 ->
0.7525.

### Results

|           | Top-1 Accuracy | Top-5 Accuracy | Prompt Template |
|:---------:| :------------: | :------------: | :-------------- |
|   10-shot | 0.7448         | 0.9153         |                 |
| Zero-shot | 0.8194         | 0.9630         | A photo of {text}. |
| Zero-shot | 0.8063         | 0.9550         | This is a photo containing images of {text}. |
| Zero-shot | 0.7558         | 0.9210         | This is a photo from ImageNet's {label} category. This category contains photos of {text}. |

### Code
The code is available in [model_repository/siglip_vision/validate.py](../model_repository/siglip_vision/validate.py)