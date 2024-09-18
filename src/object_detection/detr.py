import os
import sys
import json
import shutil
from typing import List, Tuple, Dict, Any
from pathlib import Path
from functools import partial

from datasets import load_dataset, Dataset
from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image
import albumentations
import numpy as np
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForObjectDetection,
    AutoImageProcessor,
)
from transformers.image_transforms import center_to_corners_format
from peft import IA3Config, LoraConfig, LNTuningConfig, get_peft_model
from peft.peft_model import PeftModel
from torch.nn import functional as F

# ============================================================
# Dataset loading functions
# ============================================================
# @todo: This is a pretty terrible name for a function
def load_images_into_dataset(dataset: Dataset, image_path: str) -> Dataset:
    """
    Load images into the dataset

    Args:
    dataset: The dataset to load the images into
    image_path: The path to the images
    """
    return dataset.map(
        lambda sample: {
            "image": Image.open(os.path.join(image_path, sample["file_name"]))
        }
    )


def load_local_dataset(dataset_path):
    """
    Load a dataset from a local path

    Args:
    dataset_path: The path to the dataset

    """
    train_dataset_path = os.path.join(dataset_path, "train")
    val_dataset_path = os.path.join(dataset_path, "val")

    dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(train_dataset_path, "metadata.jsonl"),
            "val": os.path.join(val_dataset_path, "metadata.jsonl"),
        },
    )

    dataset["train"] = load_images_into_dataset(dataset["train"], train_dataset_path)
    dataset["val"] = load_images_into_dataset(dataset["val"], val_dataset_path)

    return dataset


# ============================================================
# Augmentation functions
# ============================================================
def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        categories (List[int]): list of categories/class labels corresponding to provided bounding boxes
        areas (List[float]): list of corresponding areas to provided bounding boxes
        bboxes (List[Tuple[float]]): list of bounding boxes provided in COCO format
            ([center_x, center_y, width, height] in absolute coordinates)

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }


def augment_and_transform_batch(
    examples, transform, image_processor, return_pixel_mask=False
):
    """Apply augmentations and format annotations in COCO format for object detection task"""

    images = []
    annotations = []
    for image_id, image, objects in zip(
        examples["image_id"], examples["image"], examples["objects"]
    ):
        image = np.array(image.convert("RGB"))

        # apply augmentations
        output = transform(
            image=image, bboxes=objects["bbox"], category=objects["category"]
        )
        images.append(output["image"])

        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], objects["area"], output["bboxes"]
        )
        annotations.append(formatted_annotations)

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(
        images=images, annotations=annotations, return_tensors="pt"
    )

    if not return_pixel_mask:
        result.pop("pixel_mask", None)

    return result


def apply_image_augmentations(dataset: Dataset, image_processor):
    """
    Apply augmentations to the images in the dataset

    Args:
    dataset: The dataset to apply the augmentations to
    """
    train_transform = albumentations.Compose(
        [
            albumentations.Perspective(p=0.1),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightnessContrast(p=0.5),
            albumentations.HueSaturationValue(p=0.1),
        ],
        bbox_params=albumentations.BboxParams(
            format="coco", label_fields=["category"], clip=True, min_area=10
        ),
    )

    validation_transform = albumentations.Compose(
        [albumentations.NoOp()],
        bbox_params=albumentations.BboxParams(
            format="coco", label_fields=["category"], clip=True
        ),
    )

    train_transform_batch = partial(
        augment_and_transform_batch,
        transform=train_transform,
        image_processor=image_processor,
    )

    validation_transform_batch = partial(
        augment_and_transform_batch,
        transform=validation_transform,
        image_processor=image_processor,
    )

    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["val"] = dataset["val"].with_transform(validation_transform_batch)
    return dataset

def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data


def convert_bbox_yolo_to_pascal(boxes, image_size):
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (Tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])

    return boxes


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


@torch.no_grad()
def compute_metrics(evaluation_results, image_processor, threshold=0.0, id2label=None):
    """
    Compute mean average mAP, mAR and their variants for the object detection task.

    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.
        threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """

    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    # Collect targets in the required format for metric computation
    for batch in targets:
        # collect image sizes, we will need them for predictions post processing
        batch_image_sizes = torch.tensor(np.array([x["orig_size"] for x in batch]))
        image_sizes.append(batch_image_sizes)
        # collect targets in the required format for metric computation
        # boxes were converted to YOLO format needed for model training
        # here we will convert them to Pascal VOC format (x_min, y_min, x_max, y_max)
        for image_target in batch:
            boxes = torch.tensor(image_target["boxes"])
            boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then image_processor convert them to Pascal VOC format
    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(
            logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes)
        )
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    # Compute metrics
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # Replace list of per class metrics with separate metric for each class
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(
        classes, map_per_class, mar_100_per_class
    ):
        class_name = (
            id2label[class_id.item()] if id2label is not None else class_id.item()
        )
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

    return metrics

import wandb
os.environ["WANDB_PROJECT"] = "nngu2-mcai"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

def build_trainer(
    model, image_processor, train_dataset, id2label, eval_dataset, output_path
) -> Trainer:
    training_args = TrainingArguments(
        output_dir=os.path.join(output_path, "outputs"),
        num_train_epochs=60,
        fp16=False,
        per_device_train_batch_size=8,
        dataloader_num_workers=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        weight_decay=1e-4,
        max_grad_norm=0.01,
        metric_for_best_model="eval_map",
        greater_is_better=True,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_steps=10,
        logging_steps=10,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        push_to_hub=False,
        # report_to=["wandb"],
        resume_from_checkpoint=True
    )

    eval_compute_metrics_fn = partial(
        compute_metrics,
        image_processor=image_processor,
        id2label=id2label,
        threshold=0.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    return trainer


def train_model_on_dataset(
    model, image_processor, id2label, label2id, dataset_path, output_path
):
    """
    Train the model on the dataset

    Args:
    model: The model to train
    image_processor: The image processor
    train_dataset: The dataset to train on
    eval_dataset: The dataset to evaluate on
    """
    dataset = load_local_dataset(dataset_path)
    dataset = apply_image_augmentations(dataset, image_processor)

    trainer = build_trainer(
        model=model,
        image_processor=image_processor,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        output_path=output_path,
        id2label=id2label,
    )
    trainer.train()
    trainer.save_model(os.path.join(output_path, "model.pth"))
    return trainer

def train_base_model(
    checkpoint: Path,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    dataset_dir: str,
    output_dir: str,
):
    """
    Train the base model

    Args:
    checkpoint: The path to the checkpoint
    """
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to("cuda")

    train_model_on_dataset(
        model=model,
        image_processor=image_processor,
        dataset_path=dataset_dir,
        output_path=output_dir,
        id2label=id2label,
        label2id=label2id,
    )


def train_peft_model(
    checkpoint: Path,
    peft_config,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    dataset_dir: str,
    output_dir: str,
):
    """
    Train the PEFT model

    Args:
    checkpoint: The path to the checkpoint
    """
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    peft_model = get_peft_model(model, peft_config)
    train_model_on_dataset(
        model=peft_model,
        image_processor=image_processor,
        dataset_path=dataset_dir,
        output_path=output_dir,
        id2label=id2label,
        label2id=label2id,
    )


# ============================================================
# PEFT specific Training functions  
# ============================================================
def train_peft_model_lora(
    checkpoint: Path,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    dataset_dir: str,
    output_dir: str,
):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["k_proj", "v_proj", "q_proj", "out_proj", "fc1", "fc2"],
        modules_to_save=["class_labels_classifier", "bbox_predictor"],
    )
    train_peft_model(
        checkpoint, lora_config, id2label, label2id, dataset_dir, output_dir
    )


def train_peft_model_ia3(
    checkpoint: Path,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    dataset_dir: str,
    output_dir: str,
):
    ia3_config = IA3Config(
        target_modules=["k_proj", "v_proj", "q_proj", "out_proj", "fc1", "fc2"],
        feedforward_modules=["fc1", "fc2"],
        modules_to_save=["class_labels_classifier", "bbox_predictor"],
    )

    train_peft_model(
        checkpoint, ia3_config, id2label, label2id, dataset_dir, output_dir
    )


def train_peft_model_lntuning(
    checkpoint: Path,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    dataset_dir: str,
    output_dir: str,
):
    lntuning_config = LNTuningConfig(
        target_modules=[
            "self_attn_layer_norm",
            "final_layer_norm",
            "encoder_attn_layer_norm",
            "layernorm",
        ],
        modules_to_save=["class_labels_classifier", "bbox_predictor"],
    )
    train_peft_model(
        checkpoint, lntuning_config, id2label, label2id, dataset_dir, output_dir
    )

def evaluate_base_model(
    dataset_dir: str,
    checkpoint: Path,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    prediction_output_dir: str,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    to_save = []
    for data in os.listdir(dataset_dir):
        if not data.endswith(".jpg") and not data.endswith(".png"): continue
        print("Evaluating", data, flush=True, file=sys.stdout)
        
        if os.path.exists(os.path.join(prediction_output_dir, f"threshold_0.99", f"{data}.json")): continue

        image = Image.open(os.path.join(dataset_dir, data))
        image = image.convert("RGB")
        inputs = image_processor(images=[image], return_tensors="pt")
        outputs = model(**inputs.to(device))
        target_sizes = torch.tensor([[image.size[1], image.size[0]]])
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98, 0.99]:
            threshold_output_dir = os.path.join(prediction_output_dir, f"threshold_{threshold}")
            prediction_json_path = os.path.join(threshold_output_dir, f"{data}.json")

            if not os.path.exists(threshold_output_dir):
                os.makedirs(threshold_output_dir)

            results = image_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                to_save.append({
                    "category": id2label[label.item()],
                    "category_id": label.item(),
                    "confidence": round(score.item(), 3),
                    "xmin": box[0],
                    "ymin": box[1],
                    "xmax": box[2],
                    "ymax": box[3]
                })


            with open(prediction_json_path, "w") as f:
                f.write("")
                for t in to_save:
                    if t["confidence"] <= threshold: continue
                    f.write(json.dumps(t))
                    f.write("\n")