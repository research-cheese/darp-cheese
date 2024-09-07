from peft import IA3Config, LoraConfig, LNTuningConfig


def formatted_peft_configs(lora_config=None, ia3_config=None, lntuning_config=None):
    return {"lora": lora_config, "ia3": ia3_config, "lntuning": lntuning_config}


class ModelConfigs:
    peft_configs = {
        # ================================================================
        # Peking University
        # ================================================================
        "PekingU/rtdetr_r18vd_coco_o365": formatted_peft_configs(),
        
        # ================================================================
        # HUST Vision Lab
        # ================================================================
        "hustvl/yolos-tiny": formatted_peft_configs(
            lora_config=LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                target_modules=["k_proj", "v_proj", "q_proj", "out_proj", "fc1", "fc2"],
                modules_to_save=["class_labels_classifier", "bbox_predictor"],
            ),
        ),
        "hustvl/yolos-small": formatted_peft_configs(),
        "hustvl/yolos-base": formatted_peft_configs(),
        # ================================================================
        # Facebook
        # ================================================================
        "facebook/detr-resnet-50": formatted_peft_configs(
            lora_config=LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                target_modules=["k_proj", "v_proj", "q_proj", "out_proj", "fc1", "fc2"],
                modules_to_save=["class_labels_classifier", "bbox_predictor"],
            ),
            ia3_config=IA3Config(
                target_modules=["k_proj", "v_proj", "q_proj", "out_proj", "fc1", "fc2"],
                feedforward_modules=["fc1", "fc2"],
                modules_to_save=["class_labels_classifier", "bbox_predictor"],
            ),
            lntuning_config=LNTuningConfig(
                target_modules=[
                    "self_attn_layer_norm",
                    "final_layer_norm",
                    "encoder_attn_layer_norm",
                    "layernorm",
                ],
                modules_to_save=["class_labels_classifier", "bbox_predictor"],
            ),
        ),
        "facebook/detr-resnet-101": formatted_peft_configs(
            lora_config=LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                target_modules=["k_proj", "v_proj", "q_proj", "out_proj", "fc1", "fc2"],
                modules_to_save=["class_labels_classifier", "bbox_predictor"],
            ),
            ia3_config=IA3Config(
                target_modules=["k_proj", "v_proj", "q_proj", "out_proj", "fc1", "fc2"],
                feedforward_modules=["fc1", "fc2"],
                modules_to_save=["class_labels_classifier", "bbox_predictor"],
            ),
            lntuning_config=LNTuningConfig(
                target_modules=[
                    "self_attn_layer_norm",
                    "final_layer_norm",
                    "encoder_attn_layer_norm",
                    "layernorm",
                ],
                modules_to_save=["class_labels_classifier", "bbox_predictor"],
            ),
        ),
    }
