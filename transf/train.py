from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

def train_model_with_lora(
    model,
    tokenizer,
    dataset_name="FreedomIntelligence/medical-o1-reasoning-SFT",
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
):
    """
    使用LoRA微调语言模型
    
    参数:
        model: 预训练模型
        tokenizer: 分词器
        dataset_name: 数据集名称
        dataset_lang: 数据集语言
        dataset_split: 数据集分割
        lora_r: LoRA矩阵的秩
        lora_alpha: LoRA缩放因子
        lora_target_modules: 应用LoRA的目标模块
        lora_dropout: LoRA dropout率
        lora_bias: 是否对偏置项应用LoRA
        output_dir: 输出目录
        per_device_batch_size: 每设备批次大小
        num_train_epochs: 训练轮数
        logging_steps: 日志记录步数间隔
        save_steps: 模型保存步数间隔
        learning_rate: 学习率
        fp16: 是否使用fp16训练
        max_length: 最大序列长度
    """
    
    # 应用LoRA配置
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    
    # 加载和预处理数据集
    dataset = load_dataset(
        dataset_name, 
        dataset_lang, 
        split=dataset_split, 
        trust_remote_code=True
    )
    
    def format_prompt(example):
        prompt = f"""你是一个医学问答专家，请根据以下信息进行推理回答：
问题：{example['Question']}
推理过程：{example['Complex_CoT']}
请回答："""
        
        input_ids = tokenizer(
            prompt, 
            truncation=True, 
            padding="max_length", 
            max_length=max_length
        )["input_ids"]
        
        labels = tokenizer(
            example["Response"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length
        )["input_ids"]
        
        return {
            "input_ids": input_ids,
            "labels": labels
        }
    
    tokenized_dataset = dataset.map(format_prompt)
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        save_steps=save_steps,
        learning_rate=learning_rate,
        fp16=fp16
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # 创建并启动训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    trainer.train()
    
    return model

# 使用示例
if __name__ == "__main__":
    # 加载模型和分词器
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        torch_dtype=torch.float16
    )
    model.cuda()
    
    # 调用训练函数
    trained_model = train_model_with_lora(
        model=model,
        tokenizer=tokenizer,
        dataset_name="FreedomIntelligence/medical-o1-reasoning-SFT",
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