from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transf.train import *

def main():
    # 配置参数
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    dataset_name = "FreedomIntelligence/medical-o1-reasoning-SFT"
    
    # 1. 加载模型和分词器
    print("正在加载模型和分词器...")
    tokenizer, model = load_model_and_tokenizer(
        model_name=model_name,
        torch_dtype=torch.float16
    )
    
    # 2. 使用LoRA微调模型
    print("开始训练模型...")
    trained_model = train_model_with_lora(
        model=model,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        dataset_lang="zh",
        dataset_split="train[0:500]",
        lora_r=8,
        lora_alpha=16,
        lora_target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        lora_bias="none",
        output_dir="./results",
        per_device_batch_size=2,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=50,
        learning_rate=2e-4,
        fp16=True,
        max_length=512
    )
    
    # 3. 测试训练后的模型
    print("\n训练完成，测试模型...")
    prompt = "一个患有急性阑尾炎的病人已经发病5天，腹痛稍有减轻但仍然发热，在体检时发现右下腹有压痛的包块，此时应如何处理？"
    response = generate_response(
        prompt=prompt,
        tokenizer=tokenizer,
        model=trained_model,
        max_new_tokens=1024,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    
    print("\n模型回复：")
    print(response)

if __name__ == "__main__":
    # 确保所有需要的函数都已定义
    from model_loading import load_model_and_tokenizer
    from model_training import train_model_with_lora
    from model_inference import generate_response
    
    main()