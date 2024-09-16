import pandas
import shutil
import json
import os

from training_configs import AirsimObjectDetectionConfig

# ========================================
# GROUND TRUTH
# ========================================
eval_dataset_paths = [
    "data/mcai/airsim-data/mixed/test",
    "data/mcai/airsim-data/mixed/val",
]


def get_name_from_path(path):
    return path.split("/")[-1]


def create_mario_str(list):
    return "\n".join([json.dumps(s) for s in list])


def create_mario_ground_truth(ground_truth_metadata):
    ground_truth_list = []
    objects = ground_truth_metadata["objects"]
    for i in range(len(objects["id"])):
        xmin = objects["bbox"][i][0]
        ymin = objects["bbox"][i][1]
        xmax = xmin + objects["bbox"][i][2]
        ymax = ymin + objects["bbox"][i][3]
        ground_truth_list.append(
            {
                "class": AirsimObjectDetectionConfig.id2label[objects["category"][i]],
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    return create_mario_str(ground_truth_list)


def mariofy(model, dataset, eval_dataset, output_dir, threshold):
    ground_truth_metadata_path = f"{eval_dataset}/train/metadata.jsonl"
    predictions_folder_path = f"output/{dataset}/{model}/base/predictions/{eval_dataset}/train/threshold_{threshold}"

    images_folder_path = f"{eval_dataset}/train"

    ground_truth_metadata_list = pandas.read_json(ground_truth_metadata_path, lines=True).to_dict(orient="records")
    for ground_truth_metadata in ground_truth_metadata_list:
        print(ground_truth_metadata)
        os.mkdir(f"{output_dir}/{ground_truth_metadata['file_name']}")
        file_name = ground_truth_metadata["file_name"]
        image_path = f"{images_folder_path}/{file_name}"
        prediction_path = f"{predictions_folder_path}/{file_name}.json"

        shutil.copy(image_path, f"{output_dir}/{file_name}/image.png")
        shutil.copy(prediction_path, f"{output_dir}/{file_name}/prediction.jsonl")

        with open(f"{output_dir}/{file_name}/ground_truth.jsonl", "w") as f:
            f.write(create_mario_ground_truth(ground_truth_metadata))

for model in AirsimObjectDetectionConfig.checkpoints:
    for dataset in AirsimObjectDetectionConfig.datasets:
        for eval_dataset in eval_dataset_paths:
            for threshold in AirsimObjectDetectionConfig.thresholds:
                mario_output_path = f"mario/{model}//threshold_{threshold}/{get_name_from_path(dataset)}/{get_name_from_path(eval_dataset)}"
                if os.path.exists(mario_output_path): shutil.rmtree(mario_output_path)
                os.makedirs(mario_output_path, exist_ok=True)                    
                mariofy(
                    model=model,
                    dataset=dataset,
                    eval_dataset=eval_dataset,
                    output_dir=mario_output_path,
                    threshold=threshold,
                )
