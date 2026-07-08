from transformers import AutoTokenizer, AutoConfig
import os

# 1. 准备保存目录
output_dir = "qwen_init2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. 加载基础 Tokenizer
# 如果你的下载依然很慢，可以加上镜像站环境变量：
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
model_id = "Qwen/Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# 3. 构造语义 ID Token
# 注意：这些大小 (2048, 2048, 1024) 必须和你 RQ-VAE 的配置一模一样
special_tokens = []
for prefix, size in [("a", 2048), ("b", 2048), ("c", 1024)]:
    for i in range(size):
        special_tokens.append(f"<{prefix}_{i}>")

# 4. 添加边界标记
special_tokens.extend(["<|hist_clk_start|>", "<|hist_clk_end|>"])

# 5. 正式注入词典
# special_tokens=True 确保这些词被视为一个整体，不会被拆散
tokenizer.add_tokens(special_tokens, special_tokens=True)
tokenizer.save_pretrained(output_dir)

# 6. 同步保存 Config
# 这是为了让模型启动时知道 vocab_size 已经从 15w+ 变成了 15w+5000 多
config = AutoConfig.from_pretrained(model_id)
config.vocab_size = len(tokenizer)
config.save_pretrained(output_dir)

print(f"✅ Qwen2 Tokenizer 初始化完成！")
print(f"最终词表大小: {len(tokenizer)}")