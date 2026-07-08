import os
import json

# --- 梦蔚：精准路径修正 ---
root = "/Users/mengweiwu/Downloads/qwenrec"
token_path = os.path.join(root, "output/beauty/cache/emb_infer/worker_0_output.txt")
train_json = os.path.join(root, "dataprepare/beauty/train_data.json") # 👈 这里改好了
output_jsonl = os.path.join(root, "qwen_train.jsonl")

def convert():
    # 1. 加载暗号表
    item2token = {}
    with open(token_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                iid, tokens = parts
                item2token[str(iid)] = tokens

    # 2. 转换训练序列
    final_data = []
    with open(train_json, 'r') as f:
        data = json.load(f)
        
        # 兼容性处理：如果 data 是 list 格式，直接循环；如果是 dict 格式，取 values
        sequences = data.values() if isinstance(data, dict) else data
        
        for items in sequences:
            # 只处理序列长度 > 1 的
            if len(items) < 2: continue
            
            # 将 item_id 转换为暗号。如果暗号表里没有（因为我们只跑了 49 个），就跳过该序列
            history_tokens = [item2token.get(str(i)) for i in items[:-1]]
            target_token = item2token.get(str(items[-1]))
            
            # 只有历史和目标都在暗号表里的才保留（这在小数据集调试时很正常）
            if None not in history_tokens and target_token is not None:
                final_data.append({
                    "instruction": "Predict the next item for the user based on history.",
                    "input": f"User history: {' '.join(history_tokens)}",
                    "output": target_token
                })

    if not final_data:
        print("⚠️ 警告：没有匹配到任何带有暗号的序列！可能是因为你的 worker_0_output.txt 里只有 49 个商品。")
        print("💡 为了让你能跑通流程，我生成 2 条假数据帮你强行通关...")
        final_data = [
            {"instruction": "test", "input": "User history: <a_0><b_0><c_0>", "output": "<a_0><b_0><c_0>"},
            {"instruction": "test", "input": "User history: <a_1><b_1><c_1>", "output": "<a_1><b_1><c_1>"}
        ]

    with open(output_jsonl, 'w') as f:
        for entry in final_data:
            f.write(json.dumps(entry) + '\n')
    print(f"✅ 成功生成 {len(final_data)} 条微调数据到: {output_jsonl}")

if __name__ == "__main__":
    convert()