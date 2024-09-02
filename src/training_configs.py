class AirsimConfig:
    id2label = {0: "construction", 1: "nature", 2: "pedestrian", 3: "vehicle"}
    label2id = {v: k for k, v in id2label.items()}

    datasets = ["airsim/dust-10/train"]
    checkpoints = [
        "hustvl/yolos-tiny",
        "hustvl/yolos-small",
        "hustvl/yolos-base",
        "facebook/detr-resnet-50",
        "facebook/detr-resnet-101",
        "google/owlvit-base-patch32",
        "google/owlvit-large-patch32",
    ]
