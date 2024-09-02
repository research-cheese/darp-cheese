from object_detection.detr import train_base_model, train_peft_model_ia3, train_peft_model_lora, train_peft_model_lntuning
import os

id2label = {0: "construction", 1: "nature", 2: "pedestrian", 3: "vehicle"}
label2id = {v: k for k, v in id2label.items()}

for dataset in ["airsim/dust-10/train"]:
    for checkpoint in [
        "hustvl/yolos-tiny",
        "facebook/detr-resnet-50",
        "facebook/detr-resnet-101",
        "hustvl/yolos-small",
        "hustvl/yolos-base",
        "google/owlvit-base-patch32",
        "google/owlvit-large-patch32",
    ]:
        output_dir = os.path.join(
            "output",
            dataset,
            checkpoint
        )
        output_dir_base = os.path.join(
            output_dir,
            "base"
        )
        dataset_dir = os.path.join("data", dataset)

        train_base_model(
            checkpoint=checkpoint,
            id2label=id2label,
            label2id=label2id,
            dataset_dir=dataset_dir,
            output_dir=output_dir_base,
        )

        output_base_model_path = os.path.join(output_dir_base, "model.pth")
        
        # Train PEFT models ========================================
        train_peft_model_lora(
            checkpoint=output_base_model_path,
            id2label=id2label,
            label2id=label2id,
            dataset_dir=dataset_dir,
            output_dir=os.path.join(output_dir, "lora")
        )
        
        train_peft_model_ia3(
            checkpoint=output_base_model_path,
            id2label=id2label,
            label2id=label2id,
            dataset_dir=dataset_dir,
            output_dir=os.path.join(output_dir, "ia3"),
        )

        train_peft_model_lntuning(
            checkpoint=output_base_model_path,
            id2label=id2label,
            label2id=label2id,
            dataset_dir=dataset_dir,
            output_dir=os.path.join(output_dir, "lntuning"),
        )