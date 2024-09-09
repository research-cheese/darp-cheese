from object_detection.detr import evaluate_base_model
import os

from training_configs import AirsimObjectDetectionConfig

for eval_dataset in [
    AirsimObjectDetectionConfig.test_dataset,
    AirsimObjectDetectionConfig.val_dataset
]:
    for checkpoint in AirsimObjectDetectionConfig.checkpoints:
        output_dir_base = os.path.join(
            "output",
            AirsimObjectDetectionConfig.base_dataset,
            checkpoint,
            "base"
        )
        output_dir_base_model_path = os.path.join(output_dir_base, "model.pth")

        evaluate_base_model(
            checkpoint=output_dir_base_model_path,
            id2label=AirsimObjectDetectionConfig.id2label,
            label2id=AirsimObjectDetectionConfig.label2id,
            dataset_dir=eval_dataset,
            prediction_output_dir=os.path.join(output_dir_base, "predictions", eval_dataset)
        )

        for dataset in AirsimObjectDetectionConfig.datasets:

            dataset_output_dir = os.path.join(
                "output",
                dataset,
                checkpoint,
                "base"
            )
            dataset_output_dir_model_path = os.path.join(dataset_output_dir, "model.pth")

            evaluate_base_model(
                checkpoint=dataset_output_dir_model_path,
                id2label=AirsimObjectDetectionConfig.id2label,
                label2id=AirsimObjectDetectionConfig.label2id,
                dataset_dir=eval_dataset,
                prediction_output_dir=os.path.join(dataset_output_dir, "predictions", eval_dataset)
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