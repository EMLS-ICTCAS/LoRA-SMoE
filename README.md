# LoRA-SMoE
微调数据集:datasets/commonsense_15k.json

1.获得参数敏感性，参考badam实现
/home/xjz/proj/LoRA-SMoE/get_sensitivity.py

2.Allocate the number of experts per parameter block with parameter sensitivity
/home/xjz/proj/LoRA-SMoE/get_sensitivity.py

3.fine-tuning
revise peft/tuners/LoRA-SMoE.py   make testpath = 'your_expert_num.json'
rename it lora.py then run qwen2loramoe


