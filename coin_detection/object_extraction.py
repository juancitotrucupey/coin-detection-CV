from transformers import pipeline
import os
import numpy as np
from PIL import Image, ImageDraw
import json
from scipy.optimize import linear_sum_assignment
from pydantic import BaseModel

# Set the path to the COCO JSON file
COCO_JSON_PATH = "/Users/juan/Desktop/Non Lyric stuff/Technical_tests/Coin_Detection_CV/source_code/coin-detection-CV/coin_detection/dataset/_annotations.coco.json"
N_IMAGES_TO_PROCESS = 40
# TODO: iterate over the iou threshold to compute the ROC curve
IOU_THRESHOLD = 0.5

class BoundingBox(BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int

class Centroid(BaseModel):
    x: int
    y: int

class Radius(BaseModel):
    radius: int

class Prediction(BaseModel):
    bounding_box: BoundingBox
    centroid: Centroid
    radius: Radius
    label: str
    score: float

class Assignment(BaseModel):
    prediction_bounding_box: BoundingBox
    ground_truth_bounding_box: BoundingBox
    iou: float

class HungarianAssignment(BaseModel):
    tp_count: int
    fp_count: int
    fn_count: int
    assignments: list[Assignment]

class Metrics(BaseModel):
    precision: float
    recall: float
    f1_score: float

class JaccardIndex(BaseModel):
    simple_average: float
    weighted_average: float
    min_jaccard_index: float
    max_jaccard_index: float

def load_coco_json(path) -> dict[str, list[BoundingBox]]:
    with open(path, 'r') as f:
        coco_data = json.load(f)

    image_to_annotations = {}
    for image in coco_data['images']:
        annotations = [
            annotation for annotation in coco_data['annotations'] if annotation['image_id'] == image['id'] and annotation['category_id'] == 1
        ]
        bounding_boxes = []
        for annotation in annotations:
            xmin, ymin, width, height = annotation['bbox']
            xmax = xmin + width
            ymax = ymin + height
            bounding_boxes.append(BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))
        image_to_annotations[image['file_name']] = bounding_boxes
    return image_to_annotations


def compute_jaccard_index_union(predictions: list[Prediction], image_annotations: list[BoundingBox], image_width: int, image_height: int) -> float:
    """
    Compute Jaccard Index between union of predicted bounding boxes and union of ground truth bounding boxes.
    
    Args:
        predictions: List of Prediction objects
        image_annotations: List of BoundingBox objects
        image_width: Width of the image
        image_height: Height of the image
    
    Returns:
        Jaccard Index (float between 0 and 1)
    """
    
    # Create binary masks for the image
    pred_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    gt_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    # Fill predicted bounding boxes mask
    print(f"    📦 Processing {len(predictions)} predictions for Jaccard Index...")
    for i, bounding_box in enumerate(predictions):
        box = bounding_box.bounding_box
        xmin, ymin, xmax, ymax = box.xmin, box.ymin, box.xmax, box.ymax
        
        # Ensure coordinates are within image bounds
        xmin = max(0, min(int(xmin), image_width - 1))
        ymin = max(0, min(int(ymin), image_height - 1))
        xmax = max(0, min(int(xmax), image_width))
        ymax = max(0, min(int(ymax), image_height))
        
        # Fill the mask region
        pred_mask[ymin:ymax, xmin:xmax] = 1
        print(f"      🎯 Prediction {i+1}: [{xmin}, {ymin}, {xmax}, {ymax}]")
    
    # Fill ground truth bounding boxes mask
    gt_count = 0
    for bounding_box in image_annotations:
        # Ensure coordinates are within image bounds
        xmin = max(0, min(int(bounding_box.xmin), image_width - 1))
        ymin = max(0, min(int(bounding_box.ymin), image_height - 1))
        xmax = max(0, min(int(bounding_box.xmax), image_width))
        ymax = max(0, min(int(bounding_box.ymax), image_height))
        
        # Fill the mask region
        gt_mask[ymin:ymax, xmin:xmax] = 1
        gt_count += 1
        print(f"      ✅ Ground Truth {gt_count}: [{xmin}, {ymin}, {xmax}, {ymax}]")
    
    # Compute intersection and union
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    # Compute Jaccard Index
    if union == 0:
        jaccard_index = 1.0 if intersection == 0 else 0.0
    else:
        jaccard_index = intersection / union
    
    print(f"    📊 Intersection pixels: {intersection}")
    print(f"    📊 Union pixels: {union}")
    print(f"    🎯 Jaccard Index: {jaccard_index:.4f}")
    
    return jaccard_index

def get_predictions(image, detector, labels) -> list[Prediction]:
    predictions = detector(
        image,
        candidate_labels=labels,
    )
    return [
        Prediction(
            bounding_box=BoundingBox(
                xmin=prediction["box"]["xmin"], 
                ymin=prediction["box"]["ymin"], 
                xmax=prediction["box"]["xmax"], 
                ymax=prediction["box"]["ymax"]), 
            centroid=get_centroid_and_radius(
                BoundingBox(xmin=prediction["box"]["xmin"], 
                ymin=prediction["box"]["ymin"], 
                xmax=prediction["box"]["xmax"], 
                ymax=prediction["box"]["ymax"]
                )
            )[0], 
            radius=get_centroid_and_radius(
                BoundingBox(xmin=prediction["box"]["xmin"], 
                ymin=prediction["box"]["ymin"], 
                xmax=prediction["box"]["xmax"], 
                ymax=prediction["box"]["ymax"]
                )
            )[1], 
            label=labels[0], 
            score=prediction["score"]
            ) for prediction in predictions
            ]

def get_centroid_and_radius(bounding_box: BoundingBox) -> tuple[Centroid, Radius]:
    xmin, ymin, xmax, ymax = bounding_box.xmin, bounding_box.ymin, bounding_box.xmax, bounding_box.ymax
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    radius = (xmax - xmin) / 2
    return Centroid(x=int(x_center), y=int(y_center)), Radius(radius=int(radius))

def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: BoundingBox
        box2: BoundingBox
    
    Returns:
        IoU value (float between 0 and 1)
    """
    # Handle different box formats
    x1_min, y1_min, x1_max, y1_max = box1.xmin, box1.ymin, box1.xmax, box1.ymax
    
    x2_min, y2_min, x2_max, y2_max = box2.xmin, box2.ymin, box2.xmax, box2.ymax
    
    # Compute intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmin >= inter_xmax or inter_ymin >= inter_ymax:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # Compute union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area

def hungarian_assignment(predictions: list[Prediction], ground_truths: list[BoundingBox], iou_threshold: float = 0.5) -> HungarianAssignment:
    """
    Use Hungarian algorithm to assign predictions to ground truths based on IoU.
    
    Args:
        predictions: List of prediction objects with 'box' attribute
        ground_truths: List of ground truth annotations with 'bbox' attribute
        iou_threshold: Minimum IoU for a valid match
    
    Returns:
        tp_count: Number of true positives
        fp_count: Number of false positives  
        fn_count: Number of false negatives
        assignments: List of (pred_idx, gt_idx, iou) for matched pairs
    """
    if len(predictions) == 0 and len(ground_truths) == 0:
        return HungarianAssignment(tp_count=0, fp_count=0, fn_count=0, assignments=[])
    
    if len(predictions) == 0:
        return HungarianAssignment(tp_count=0, fp_count=0, fn_count=len(ground_truths), assignments=[])
    
    if len(ground_truths) == 0:
        return HungarianAssignment(tp_count=0, fp_count=len(predictions), fn_count=0, assignments=[])
    
    # Create cost matrix (using negative IoU as cost since Hungarian minimizes)
    cost_matrix = np.zeros((len(predictions), len(ground_truths)))
    iou_matrix = np.zeros((len(predictions), len(ground_truths)))
    
    for i, prediction in enumerate(predictions):
        pred_box = prediction.bounding_box
        for j, gt_annotation in enumerate(ground_truths):
            gt_box = gt_annotation
            iou = compute_iou(pred_box, gt_box)
            iou_matrix[i, j] = iou
            # Use negative IoU as cost (Hungarian minimizes cost)
            cost_matrix[i, j] = 1.0 - iou
            
    # Apply Hungarian algorithm
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # Count TP, FP, FN based on IoU threshold
    tp_count = 0
    fp_count = 0
    fn_count = 0
    assignments = []
    
    # Track which predictions and ground truths are matched
    matched_predictions = set()
    matched_ground_truths = set()
    
    # Check assignments from Hungarian algorithm
    # TODO: check if the assignment is correct
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        iou = iou_matrix[pred_idx, gt_idx]
        if iou >= iou_threshold:
            # Valid match
            tp_count += 1
            matched_predictions.add(pred_idx)
            matched_ground_truths.add(gt_idx)
            assignments.append(Assignment(prediction_bounding_box=predictions[pred_idx].bounding_box, ground_truth_bounding_box=ground_truths[gt_idx], iou=float(iou)))
        # If IoU is below threshold, both prediction and GT remain unmatched
    
    # Count false positives (unmatched predictions)
    fp_count = len(predictions) - len(matched_predictions)
    
    # Count false negatives (unmatched ground truths with category_id == 1)
    fn_count = len(ground_truths) - len(matched_ground_truths)
    
    return HungarianAssignment(tp_count=tp_count, fp_count=fp_count, fn_count=fn_count, assignments=assignments)

def compute_metrics(tp_count: int, fp_count: int, fn_count: int) -> Metrics:
    """
    Compute precision, recall, and F1 score from TP, FP, FN counts.
    
    Args:
        tp_count: Total true positives
        fp_count: Total false positives
        fn_count: Total false negatives
    
    Returns:
        Metrics object with precision, recall, and F1 score
    """
    # Compute precision
    if tp_count + fp_count == 0:
        precision = 0.0
    else:
        precision = tp_count / (tp_count + fp_count)
    
    # Compute recall
    if tp_count + fn_count == 0:
        recall = 0.0
    else:
        recall = tp_count / (tp_count + fn_count)
    
    # Compute F1 score
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return Metrics(precision=precision, recall=recall, f1_score=f1_score)

if __name__ == "__main__":
    print("Starting object detection process...")
    

    # Load model
    checkpoint = "google/owlv2-base-patch16-ensemble"
    print(f"Loading model: {checkpoint}")
    detector = pipeline(model=checkpoint, task="zero-shot-object-detection")
    print("Model loaded successfully!")
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    print("Output directory ready.")
    
    # Check if dataset folder exists
    if not os.path.exists("dataset"):
        print("ERROR: 'dataset' folder not found!")
        print("Please create a 'dataset' folder and add .jpg images to process.")
        exit(1)
    
    # Load images from dataset folder
    print("Load JSON file")
    image_to_bounding_boxes = load_coco_json(COCO_JSON_PATH)
    
    if len(image_to_bounding_boxes.keys()) <= 0:
        print("WARNING: No .jpg images found in 'dataset' folder!")
        print("Please add .jpg images to the 'dataset' folder.")
        exit(1)
        
    print(f"Found {len(image_to_bounding_boxes.keys())} images to process")
    
    images_to_predict = []
    image_names = []  # Keep track of original filenames
    
    counter = 0
    for image_file in image_to_bounding_boxes:
        print(f"  - Loading: {image_file}")
        image = Image.open(f"dataset/{image_file}")
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        images_to_predict.append(image)
        image_names.append(image_file)
        counter += 1
        if counter >= N_IMAGES_TO_PROCESS:
            break
        
    # Process images
    print(f"\nStarting object detection on {len(images_to_predict)} images...")
    images_to_show = []  # Move this outside the loop
    jaccard_indices = []  # Store Jaccard indices for all images
    gt_counts = []  # Store ground truth counts for each image
    
    # F1 score tracking variables
    # TODO: iterate over the iou threshold to compute the ROC curve
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    print(f"🎯 Using IoU threshold: {IOU_THRESHOLD}")
    
    for idx, (image, image_name) in enumerate(zip(images_to_predict, image_names)):
        print(f"\nProcessing image {idx + 1}/{len(images_to_predict)}: {image_name}")
        
        # Get image dimensions
        image_width, image_height = image.size
        print(f"  Image dimensions: {image_width} x {image_height}")
        
        # Count ground truth bounding boxes for this image
        gt_count = len(image_to_bounding_boxes[image_name])
        gt_counts.append(gt_count)
        print(f"  Ground truth objects: {gt_count}")
        
        # Run detection
        print("  Running object detection...")
        predictions = get_predictions(image, detector, ["coin"])
        print(f"  Found {len(predictions)} objects")
        
        # Compute Hungarian assignment and F1 metrics for this image
        print("  Computing Hungarian assignment...")
        assignment = hungarian_assignment(
            predictions, 
            image_to_bounding_boxes[image_name], 
            IOU_THRESHOLD
        )

        print(f" {image_name}: TP: {assignment.tp_count}, FP: {assignment.fp_count}, FN: {assignment.fn_count}")
        
        # Add to totals
        total_tp += assignment.tp_count
        total_fp += assignment.fp_count
        total_fn += assignment.fn_count
        
        print(f"    TP: {assignment.tp_count}, FP: {assignment.fp_count}, FN: {assignment.fn_count}")
        if assignment.assignments:
            print(f"    Assignments:")
            for assignment in assignment.assignments:
                print(f"      Prediction {assignment.prediction_bounding_box} -> GT {assignment.ground_truth_bounding_box} (IoU: {assignment.iou:.3f})")
        
        # Compute Jaccard Index
        print("  Computing Jaccard Index...")
        jaccard_index = compute_jaccard_index_union(
            predictions, 
            image_to_bounding_boxes[image_name], 
            image_width, 
            image_height
        )
        jaccard_indices.append(jaccard_index)
        
        # Create a copy for drawing (to preserve original)
        image_with_boxes = image.copy()
        draw = ImageDraw.Draw(image_with_boxes)
        
        # Draw bounding boxes
        for i, prediction in enumerate(predictions):
            box = prediction.bounding_box
            label = prediction.label
            score = prediction.score
            
            print(f"    Object {i + 1}: {label} (confidence: {score:.3f})")
            
            xmin, ymin, xmax, ymax = box.xmin, box.ymin, box.xmax, box.ymax
            # Draw predicted bounding box
            draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=3)
            draw.text((xmin, ymin), f"{label}: {round(score, 2)}", fill="red")
        # Draw ground truth bounding box
        for bounding_box in image_to_bounding_boxes[image_name]:
            xmin, ymin, xmax, ymax = bounding_box.xmin, bounding_box.ymin, bounding_box.xmax, bounding_box.ymax
            draw.rectangle((xmin, ymin, xmax, ymax), outline="green", width=3)
            draw.text((xmin, ymin), f"GT", fill="green")
        # Save individual image
        output_filename = f"output/{os.path.splitext(image_name)[0]}_detected.jpg"
        image_with_boxes.save(output_filename)
        print(f"  Saved annotated image to: {output_filename}")
        
        images_to_show.append(image_with_boxes)

    # Compute F1 metrics
    metrics = compute_metrics(total_tp, total_fp, total_fn)
    
    # Print Jaccard Index statistics
    print(f"\n{'='*60}")
    print("📊 JACCARD INDEX RESULTS")
    print(f"{'='*60}")
    if jaccard_indices:
        # Compute simple average
        avg_jaccard = sum(jaccard_indices) / len(jaccard_indices)
        min_jaccard = min(jaccard_indices)
        max_jaccard = max(jaccard_indices)
        
        # Compute weighted average using ground truth counts as weights
        total_weighted_score = sum(jaccard_idx * gt_count for jaccard_idx, gt_count in zip(jaccard_indices, gt_counts))
        total_weight = sum(gt_counts)
        
        if total_weight > 0:
            weighted_avg_jaccard = total_weighted_score / total_weight
        else:
            weighted_avg_jaccard = 0.0
        
        jaccard_index = JaccardIndex(
            simple_average=avg_jaccard,
            weighted_average=weighted_avg_jaccard,
            min_jaccard_index=min_jaccard,
            max_jaccard_index=max_jaccard
        )
        
        print(f"🎯 Simple Average Jaccard Index: {avg_jaccard:.4f}")
        print(f"⚖️  Weighted Average Jaccard Index: {weighted_avg_jaccard:.4f}")
        print(f"📏 Minimum Jaccard Index: {min_jaccard:.4f}")
        print(f"📏 Maximum Jaccard Index: {max_jaccard:.4f}")
        print(f"🏷️  Total ground truth objects: {total_weight}")
    
    # Print F1 Score statistics
    print(f"\n{'='*60}")
    print("📊 F1 SCORE RESULTS")
    print(f"{'='*60}")
    print(f"🎯 IoU Threshold: {IOU_THRESHOLD}")
    print(f"✅ Total True Positives (TP): {total_tp}")
    print(f"❌ Total False Positives (FP): {total_fp}")
    print(f"⭕ Total False Negatives (FN): {total_fn}")
    print(f"📊 Precision: {metrics.precision:.4f}")
    print(f"📊 Recall: {metrics.recall:.4f}")
    print(f"🎯 F1 Score: {metrics.f1_score:.4f}")
    
    print(f"\nProcess completed successfully!")
    print(f"Processed {len(images_to_predict)} images")
    print(f"Output files saved in 'output/' directory")