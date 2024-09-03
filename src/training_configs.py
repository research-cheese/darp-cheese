class AirsimObjectDetectionConfig:
    id2label = {0: "construction", 1: "nature", 2: "pedestrian", 3: "vehicle"}
    label2id = {v: k for k, v in id2label.items()}

    base_dataset = "airsim/train-10000/train"
    datasets = ["airsim/dust-10/train"]
    checkpoints = [
        # Ultralytics   
        "Ultralytics/YOLOv8",

        # Facebook
        "facebook/detr-resnet-50",
        "facebook/detr-resnet-101",

        # Peking University
        "PekingU/rtdetr_r101vd_coco_o365"
    ]
