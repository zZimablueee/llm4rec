import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# --- 梦蔚：极速验证配置 ---
model_id = "Qwen/Qwen2-0.5B" 
root = "/Users/mengweiwu/Downloads/qwenrec"
data_path = os.path.join(root, "qwen_train.jsonl")

def dry_run():
    print("🚀 正在加载极小版 Qwen (0.5B)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # 1. 扩充词表：把你的暗号标签加进去
    new_tokens = []
    for p in ['a', 'b', 'c']: # 假设是三层量化
        for i in range(2048):
            new_tokens.append(f"<{p}_{i}>")
    
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # 2. 加载数据
    dataset = load_dataset("json", data_files=data_path, split="train")

    def preprocess(examples):
        # 将指令、输入和输出拼成一个完整的句子
        texts = [f"{ins} {inp} {out} {tokenizer.eos_token}" 
                 for ins, inp, out in zip(examples['instruction'], examples['input'], examples['output'])]
        
        # 统一进行编码，长度设为一致
        model_inputs = tokenizer(texts, max_length=128, truncation=True, padding="max_length")
        
        # 对于 Causal LM，标签（labels）通常就是输入（input_ids）的一个副本
        # 模型会自动把 input 移动一位来预测下一个词
        model_inputs["labels"] = [list(ids) for ids in model_inputs["input_ids"]]
        
        return model_inputs
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    tokenized_dataset = dataset.map(preprocess, batched=True)

    # 3. 训练参数：只跑 1 个 Step 验证闭环
    train_args = TrainingArguments(
        output_dir="./qwen_checkpoints",
        per_device_train_batch_size=1,
        max_steps=1, 
        logging_steps=1,
        use_cpu=True, # 强制 CPU，省去 CUDA 麻烦
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_dataset,
    )

    print("🏃 开始最后一步：数据流冒烟测试...")
    trainer.train()
    print("\n" + "="*30)
    print("🏆 梦蔚！全链路已打通！")
    print("你的数据已经成功喂进了 Qwen 的嘴里。")
    print("现在，你可以拿着代码，放心地去租 80G A100 跑大数据集了！")
    print("="*30)

if __name__ == "__main__":
    dry_run()