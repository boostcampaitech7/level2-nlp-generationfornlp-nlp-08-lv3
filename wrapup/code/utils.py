from peft import LoraConfig
import json

def get_peft_config():
    # config 파일 로드
    with open("config.json", "r") as f:
        config = json.load(f)
    
    # LoRA 설정
    if config.get("use_lora", False):
        peft_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["lora_target_modules"],
            bias=config["lora_bias"],
            task_type=config["lora_task_type"]
        )
        return peft_config
    return None
