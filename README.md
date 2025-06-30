# coin-detection-CV
Project to implement a multiobject object detection pipeline

# Description
Multi object detection

# Models
Standardization of images
convert the images into a predefined size before passing them into the model

## Preprocessing
Not necessary but potentially useful

- Border enhancement:

  - input: raw images

  - output: black and white where the white are the borders of the objects in the image

- Spectral analysis of channels:

  - define which are the statistical propertioes of the colors that the coins have

- Filter to augment the contrast:

  - Canny filter


## Models
- YOLO
  - generates bounding boxes
- SAM
  - generates bounding boxes and masks

We may use this one for its versatility

To take into account
We ned to generate both bounding box and masks

### Input to the model
Add an adapter to pass multiple images to the model e.g. the original image, the output of the canny filter, and other filters, and make sure the model understand that each part what part of the input vector correspond to what information. 

### Fine tuning 
- YOLO
  - ultralitics

# Evaluation
## Conceptual framework
To define if two objects correspond to the same then use one of the following:

Jaccard index: TP / (TP + FN + FP)

You may want to use the Sørensen index=2 TP/ (2 TP+FP+FN) instead, however, we will focus on the Jaccard index bvecause it is more sensitive to diffference between sets.

## Input
- List of bounding boxed from ground truth
- List of bounding boxes predicted by the model

## Output
A number that tell us the quality of the model prediction

 

## Algorithms
We will use two metrics to asses the quality of the models:

The Jaccard index

The F1 score

The final metric will be sujm of the two metrics i.e. Jaccard index + F1 score

### Jaccard index
We will generate two images and we will compare the Jaccard index between them, this Jaccard index will tell us what is the overall quality of the object detection but will not take into account explicitly the detection of each individula coin.

#### How to compute:

- Inputs:
  - GTOBB (Ground Truth Overall Bounding Box): black and white image where the write part is the superposition of all of the ground truth bounding boxes
  - POBB (Predicted Overall Bounding Box): black and white image where the write part is the superposition of all of the predicted bounding boxes
- Algorithm
  - Compute the Jaccard index between these two images is computed, that is, we compare GTOBB and POBB pixel by pixel by assigning TP, FP, FP labels to each pixel and then computing thre resulting Jaccard Index.

### F1 score
We will compute the quality of the predictions coin by coin, that is, we will compute who well does the algorithm creates bounding boxes that can be assigned to ground truth bounding boxes.

- How to compute:
  - Use the [hungarian algorithm](https://miguel-mendez-ai.com/2024/08/25/mot-tracking-metrics) to predicted bounding boxes to ground truth bounding boxes,
- Output:
  - TP: True Positives (matches with IoU above threshold)
  - FP: False Positives (unmatched predictions)
  - FN: False Negatives (unmatched ground truth objects)
- Compute the F1 score using this information

