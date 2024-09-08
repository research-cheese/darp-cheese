class AirsimObjectDetectionConfig:
    id2label = {0: "construction", 1: "nature", 2: "pedestrian", 3: "vehicle"}
    label2id = {v: k for k, v in id2label.items()}

    base_dataset = "data/mcai/airsim-data/mixed/base"
    datasets = [
        "data/mcai/airsim-data/mixed/dust-10",
        "data/mcai/airsim-data/mixed/dust-100",
        "data/mcai/airsim-data/mixed/dust-1000",
        "data/mcai/airsim-data/mixed/fog-10",
        "data/mcai/airsim-data/mixed/fog-100",
        "data/mcai/airsim-data/mixed/fog-1000",
        "data/mcai/airsim-data/mixed/snow-10",
        "data/mcai/airsim-data/mixed/snow-100",
        "data/mcai/airsim-data/mixed/snow-1000",
        "data/mcai/airsim-data/mixed/rain-10",
        "data/mcai/airsim-data/mixed/rain-100",
        "data/mcai/airsim-data/mixed/rain-1000",
        "data/mcai/airsim-data/mixed/maple_leaf-10",
        "data/mcai/airsim-data/mixed/maple_leaf-100",
        "data/mcai/airsim-data/mixed/maple_leaf-1000",
    ]
    checkpoints = [
        # Peking University
        "PekingU/rtdetr_r18vd_coco_o365",
        # HUST Vision Lab
        "hustvl/yolos-tiny",
        "hustvl/yolos-small",
        "hustvl/yolos-base",
        # Facebook
        "facebook/detr-resnet-50",
        "facebook/detr-resnet-101",
    ]