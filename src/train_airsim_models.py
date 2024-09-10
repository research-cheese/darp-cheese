from object_detection.detr import (
    train_base_model,
    evaluate_base_model,
    train_peft_model_ia3,
    train_peft_model_lora,
    train_peft_model_lntuning,
)
import os

from training_configs import AirsimObjectDetectionConfig

for checkpoint in AirsimObjectDetectionConfig.checkpoints:
    output_dir_base = os.path.join(
        "/scratch",
        "nngu2",
        "output",
        AirsimObjectDetectionConfig.base_dataset,
        checkpoint,
        "base",
    )
    output_dir_base_model_path = os.path.join(output_dir_base, "model.pth")

    if not os.path.exists(output_dir_base_model_path):
        print("Training", output_dir_base_model_path)
        train_base_model(
            checkpoint=checkpoint,
            id2label=AirsimObjectDetectionConfig.id2label,
            label2id=AirsimObjectDetectionConfig.label2id,
            dataset_dir=AirsimObjectDetectionConfig.base_dataset,
            output_dir=output_dir_base,
        )
        evaluate_base_model(
            checkpoint=output_dir_base_model_path,
            id2label=AirsimObjectDetectionConfig.id2label,
            label2id=AirsimObjectDetectionConfig.label2id,
            dataset_dir=AirsimObjectDetectionConfig.test_dataset,
            prediction_output_dir=os.path.join(output_dir_base, "predictions", AirsimObjectDetectionConfig.test_dataset)
        )
        
        evaluate_base_model(
            checkpoint=output_dir_base_model_path,
            id2label=AirsimObjectDetectionConfig.id2label,
            label2id=AirsimObjectDetectionConfig.label2id,
            dataset_dir=AirsimObjectDetectionConfig.val_dataset,
            prediction_output_dir=os.path.join(output_dir_base, "predictions", AirsimObjectDetectionConfig.val_dataset)
        )

    for dataset in AirsimObjectDetectionConfig.datasets:

        output_dir = os.path.join("/scratch", "nngu2", "output", dataset, checkpoint)
        output_dir_base_dataset_path = os.path.join(output_dir, "base")
        output_dir_base_dataset_model_path = os.path.join(
            output_dir, "base", "model.pth"
        )
        if not os.path.exists(output_dir_base_dataset_model_path):
            print("Training", output_dir_base_dataset_model_path)
            train_base_model(
                checkpoint=output_dir_base_model_path,
                id2label=AirsimObjectDetectionConfig.id2label,
                label2id=AirsimObjectDetectionConfig.label2id,
                dataset_dir=dataset,
                output_dir=output_dir_base_dataset_path,
            )
            
            evaluate_base_model(
                checkpoint=output_dir_base_dataset_model_path,
                id2label=AirsimObjectDetectionConfig.id2label,
                label2id=AirsimObjectDetectionConfig.label2id,
                dataset_dir=AirsimObjectDetectionConfig.test_dataset,
                prediction_output_dir=os.path.join(output_dir_base_dataset_path, "predictions", AirsimObjectDetectionConfig.test_dataset)
            )
            
            evaluate_base_model(
                checkpoint=output_dir_base_dataset_model_path,
                id2label=AirsimObjectDetectionConfig.id2label,
                label2id=AirsimObjectDetectionConfig.label2id,
                dataset_dir=AirsimObjectDetectionConfig.val_dataset,
                prediction_output_dir=os.path.join(output_dir_base_dataset_path, "predictions", AirsimObjectDetectionConfig.val_dataset)
            )

        # output_base_model_path = os.path.join(output_dir_base, "model.pth")

        # Train PEFT models ========================================
        # train_peft_model_lora(
        #     checkpoint=output_base_model_path,
        #     id2label=AirsimObjectDetectionConfig.id2label,
        #     label2id=AirsimObjectDetectionConfig.label2id,
        #     dataset_dir=dataset_dir,
        #     output_dir=os.path.join(output_dir, "lora")
        # )

        # train_peft_model_ia3(
        #     checkpoint=output_base_model_path,
        #     id2label=AirsimObjectDetectionConfig.id2label,
        #     label2id=AirsimObjectDetectionConfig.label2id,
        #     dataset_dir=dataset_dir,
        #     output_dir=os.path.join(output_dir, "ia3"),
        # )

        # train_peft_model_lntuning(
        #     checkpoint=output_base_model_path,
        #     id2label=AirsimObjectDetectionConfig.id2label,
        #     label2id=AirsimObjectDetectionConfig.label2id,
        #     dataset_dir=dataset_dir,
        #     output_dir=os.path.join(output_dir, "lntuning"),
        # )
