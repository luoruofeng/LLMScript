import os
import torch
from transformers import WEIGHTS_NAME, CONFIG_NAME

def save_trained_model(model, tokenizer, output_dir):
    """
    保存训练完成的模型和分词器
    
    参数:
        model: 训练完成的模型
        tokenizer: 分词器
        output_dir: 保存目录路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理分布式模型的情况
    model_to_save = model.module if hasattr(model, 'module') else model
    
    # 保存模型权重和配置[2,3](@ref)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    
    # 保存分词器[2,3](@ref)
    tokenizer.save_pretrained(output_dir)
    
    print(f"模型和分词器已保存到 {output_dir}")
