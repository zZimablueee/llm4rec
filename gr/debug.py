import os, re, json, torch
from collections import Counter
from transformers import AutoTokenizer, Qwen2ForCausalLM, AutoConfig
from safetensors.torch import load_file
from tqdm import tqdm

# ── 路径配置 ──────────────────────────────────────────────────
BASE_CACHE     = "/root/autodl-tmp/qwenrec/cache"
QWEN_PATH      = "/root/autodl-tmp/qwenrec/qwen_init2"
CKPT_BASE      = "/root/autodl-tmp/qwenrec/cache/checkpoints/global_step_ep10_v2"
ITEM2TOKEN_DIR = os.path.join(BASE_CACHE, "emb_infer/sinkhorn10")
TRAIN_DATA     = os.path.join(BASE_CACHE, "train_data.json")

# ── 加载 tokenizer & model ────────────────────────────────────
print("Loading tokenizer & model...")

# 1. 从基座模型加载 tokenizer（并设置 pad_token，与训练一致）
tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token    # 关键！
tokenizer.padding_side = 'left'

# 2. 从基座模型加载 config
config = AutoConfig.from_pretrained(QWEN_PATH)
model  = Qwen2ForCausalLM(config)

# 3. 自动找到最新的 checkpoint 目录
ckpt_dirs  = [d for d in os.listdir(CKPT_BASE) if d.startswith("checkpoint-")]
ckpt_dirs_sorted = sorted(ckpt_dirs, key=lambda x: int(x.split("-")[1]))
if not ckpt_dirs_sorted:
    raise RuntimeError(f"No checkpoint found in {CKPT_BASE}")
latest_ckpt_dir = os.path.join(CKPT_BASE, ckpt_dirs_sorted[-1])

# 4. 加载模型权重
state_dict = load_file(os.path.join(latest_ckpt_dir, "model.safetensors"))
# 在 debug.py 的 load_file 之后加上这一行
model.lm_head.weight = model.model.embed_tokens.weight
model.load_state_dict(state_dict, strict=False)
model.cuda().eval()
print(f"Model loaded from: {latest_ckpt_dir}")

# ── 加载 item2token ───────────────────────────────────────────
item2token_dict = {}
for fname in os.listdir(ITEM2TOKEN_DIR):
    with open(os.path.join(ITEM2TOKEN_DIR, fname)) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                item2token_dict[parts[0]] = parts[1]
print(f"item2token size: {len(item2token_dict)}")

# ── 加载训练数据 ──────────────────────────────────────────────
with open(TRAIN_DATA) as f:
    train_data = json.load(f)

# ════════════════════════════════════════════════════════════
# 检查 A：统计目标 item 第一个 token（a-level）的分布
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("检查 A：目标 item 的 a-level token 分布")
print("="*60)

first_token_counter = Counter()
total_users = 0

for user_id, item_seq in train_data.items():
    # 留一法：从末尾找最后一个有 token 的 item 作为 target
    target_item = None
    for i in range(len(item_seq) - 1, -1, -1):
        if str(item_seq[i]) in item2token_dict:
            target_item = str(item_seq[i])
            break
    if target_item is None:
        continue

    token_str   = item2token_dict[target_item]
    tokens      = re.findall(r'<[a-c]_\d+>', token_str)
    if len(tokens) != 3:
        continue

    first_token_counter[tokens[0]] += 1
    total_users += 1

print(f"有效用户总数: {total_users}")
print(f"a-level token 种类数: {len(first_token_counter)}")
print(f"\nTop 10 最高频 a-level token:")
for token, cnt in first_token_counter.most_common(10):
    pct = cnt / total_users * 100
    print(f"  {token:15s}  count={cnt:6d}  占比={pct:.2f}%")

top1_token, top1_cnt = first_token_counter.most_common(1)[0]
print(f"\n⚠️  最高频 token '{top1_token}' 占比: {top1_cnt/total_users*100:.2f}%")
if top1_cnt / total_users > 0.20:
    print("  → 严重失衡！模型只需永远输出这个token就能让loss很低，推荐能力为零。")
else:
    print("  → 分布尚可，a-level 不是主要问题。")

# ════════════════════════════════════════════════════════════
# 检查 B：Teacher Forcing 下第一个 token 的 Top-1 准确率
# 只取前 500 个用户，速度快
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("检查 B：Teacher Forcing Top-1 准确率（前500用户）")
print("="*60)

correct_top1  = 0
correct_top10 = 0
total_b       = 0
predict_counter = Counter()   # 记录模型实际预测的 a-level token 分布

sample_users = list(train_data.items())[:500]

for user_id, item_seq in tqdm(sample_users):
    # 留一法构造 history 和 target
    target_item, target_idx = None, -1
    for i in range(len(item_seq) - 1, -1, -1):
        if str(item_seq[i]) in item2token_dict:
            target_item = str(item_seq[i])
            target_idx  = i
            break
    if target_item is None or target_idx == 0:
        continue

    history_tokens = [
        item2token_dict[str(x)]
        for x in item_seq[:target_idx]
        if str(x) in item2token_dict
    ][-50:]
    if not history_tokens:
        continue

    prompt = '<|hist_clk_start|>' + ''.join(history_tokens) + '<|hist_clk_end|>'
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    target_token_str = item2token_dict[target_item]
    target_tokens    = re.findall(r'<[a-c]_\d+>', target_token_str)
    if len(target_tokens) != 3:
        continue

    # 目标第一个 token 的 id
    target_first_id = tokenizer.convert_tokens_to_ids(target_tokens[0])
    if target_first_id == tokenizer.unk_token_id:
        continue

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]   # 最后一个位置的 logits

    # Top-1 预测
    pred_id   = logits.argmax().item()
    pred_token = tokenizer.convert_ids_to_tokens(pred_id)
    predict_counter[pred_token] += 1

    # Top-10 预测
    top10_ids = logits.topk(10).indices.tolist()

    if pred_id == target_first_id:
        correct_top1 += 1
    if target_first_id in top10_ids:
        correct_top10 += 1

    total_b += 1

print(f"\n总样本数: {total_b}")
print(f"Teacher Forcing Top-1  准确率: {correct_top1/total_b:.4f}  ({correct_top1}/{total_b})")
print(f"Teacher Forcing Top-10 准确率: {correct_top10/total_b:.4f}  ({correct_top10}/{total_b})")

print(f"\n模型实际预测的 a-level token Top-10（看是否都集中在同一个）:")
for token, cnt in predict_counter.most_common(10):
    print(f"  {token:15s}  count={cnt:4d}  占比={cnt/total_b*100:.1f}%")

# ════════════════════════════════════════════════════════════
# 综合诊断
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("综合诊断")
print("="*60)
top1_pct = top1_cnt / total_users
tf_acc   = correct_top1 / total_b if total_b > 0 else 0

if top1_pct > 0.20:
    print("→ [A] a-level 严重失衡，根源在 RQ-VAE 码本分配，需重训 RQ-VAE 或加 loss weighting")
elif tf_acc > 0.60:
    print("→ [B] Teacher Forcing 准确率高但推理退化，曝光偏差问题，考虑 scheduled sampling")
elif tf_acc < 0.10:
    print("→ [C] Teacher Forcing 准确率极低，训练 labels 构造有 bug，需检查 custom_dataset.py")
else:
    print(f"→ Teacher Forcing 准确率={tf_acc:.4f}，a-level 占比={top1_pct:.2%}，需结合 custom_dataset.py 进一步排查")
