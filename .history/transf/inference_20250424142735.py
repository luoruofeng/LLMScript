from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model_and_tokenizer(model_name: str, torch_dtype=torch.float16):
    """
    加载tokenizer和模型
    
    参数:
        model_name: 模型名称/路径
        torch_dtype: 模型数据类型，默认为torch.float16
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        torch_dtype=torch_dtype
    )
    model.eval()
    model.cuda()
    return tokenizer, model

def generate_response(
    prompt: str,
    tokenizer,
    model,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    do_sample: bool = True
):
    """
    生成模型回复
    
    参数:
        prompt: 输入提示文本
        tokenizer: 已加载的tokenizer
        model: 已加载的模型
        max_new_tokens: 最大生成token数
        temperature: 采样温度
        top_k: top-k采样参数
        top_p: top-p采样参数
        do_sample: 是否使用采样
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    # 配置参数
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    prompt = "一个患有急性阑尾炎的病人已经发病5天，腹痛稍有减轻但仍然发热，在体检时发现右下腹有压痛的包块，此时应如何处理？"
    
    # 加载模型
    tokenizer, model = load_model_and_tokenizer(model_name)
    
    # 生成回复
    response = generate_response(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=1024,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    
    # 输出结果
    print("模型回复：")
    print(response)

if __name__ == "__main__":
    main()