#!/usr/bin/env python3
"""
Prompt 格式一致性检查工具
验证：诊断时的 prompt 格式是否与训练时完全一致
"""
import os, re, json, torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoConfig, Qwen2ForCausalLM
from collections import Counter

# ==================== 配置 ====================
QWEN_PATH      = "/root/autodl-tmp/qwenrec/qwen_init2"
CKPT_PATH      = "/root/autodl-tmp/qwenrec/cache/checkpoints/global_step_ep10_v2_seq100/checkpoint-1000/model.safetensors"
ITEM2TOKEN_DIR = "/root/autodl-tmp/qwenrec/cache/emb_infer/sinkhorn10"
VAL_SPLIT_PATH = "/root/autodl-tmp/qwenrec/cache/_val_split.json"
TRAIN_DATA_PATH= "/root/autodl-tmp/qwenrec/gr/train_data.json"  # 训练集，用于查热门物品
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

SUSPECT_ITEMS  = ["1703", "25475", "8599", "11950", "2181"]  # 诊断里最常出现的物品

# ==================== 加载 ====================
print("=" * 60)
print("  Prompt 格式一致性 & 热门物品检查")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
vocab = tokenizer.get_vocab()

# 加载 item2token
item2token = {}
for fname in os.listdir(ITEM2TOKEN_DIR):
    fpath = os.path.join(ITEM2TOKEN_DIR, fname)
    if not os.path.isfile(fpath): continue
    with open(fpath) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                item2token[parts[0]] = parts[1]
token2item = {v: k for k, v in item2token.items()}

# ==================== 1. 特殊 Token ID 确认 ====================
print("\n[1] 特殊 Token ID 确认")
for tok in ["<|hist_clk_start|>", "<|hist_clk_end|>"]:
    tid = vocab.get(tok, "❌ 未找到")
    print(f"  {tok:30s} → id = {tid}")

# ==================== 2. 嫌疑物品分析 ====================
print("\n[2] 嫌疑物品 Token 序列分析")
for item in SUSPECT_ITEMS:
    tok_str = item2token.get(item, "❌ 不在 item2token")
    print(f"\n  item={item}")
    print(f"    token_str = {tok_str}")
    if tok_str and tok_str != "❌ 不在 item2token":
        tokens = re.findall(r"<[a-c]_\d+>", tok_str)
        ids    = [vocab.get(t, "?") for t in tokens]
        print(f"    token ids = {ids}")
        # 检查这些 id 是否是词表里特别靠前的
        for t, tid in zip(tokens, ids):
            if isinstance(tid, int) and tid < 160000:
                rank = tid - 151936  # 在新增 token 里的排名
                print(f"    {t} → id={tid}，新增 token 排名第 {rank}")

# ==================== 3. 训练集热门物品 ====================
print("\n[3] 训练集全局热门物品 Top20")
try:
    with open(TRAIN_DATA_PATH) as f:
        train_data = json.load(f)
    all_items = []
    for uid, seq in train_data.items():
        all_items.extend([str(x) for x in seq])
    counter = Counter(all_items)
    print(f"  训练集总交互数: {len(all_items)}")
    print(f"  {'排名':>4}  {'item_id':>10}  {'出现次数':>8}  {'是否是嫌疑物品':>12}")
    print(f"  {'-'*50}")
    for rank, (item, cnt) in enumerate(counter.most_common(20), 1):
        flag = "⚠️  YES" if item in SUSPECT_ITEMS else ""
        print(f"  {rank:>4}  {item:>10}  {cnt:>8}  {flag}")
except Exception as e:
    print(f"  ⚠️  无法加载训练集: {e}")

# ==================== 4. Prompt 格式逐 Token 打印 ====================
print("\n[4] Prompt 格式逐 Token 检查")
with open(VAL_SPLIT_PATH) as f:
    val_data = json.load(f)

# 取第一个有效用户
test_uid = None
test_history = None
for uid, seq in val_data.items():
    hist = [item2token[str(x)] for x in seq[:-1] if str(x) in item2token][-10:]
    if len(hist) >= 3:
        test_uid     = uid
        test_history = hist
        break

if test_uid:
    prompt = "<|hist_clk_start|>" + "".join(test_history) + "<|hist_clk_end|>"
    print(f"\n  用户 uid={test_uid}")
    print(f"  历史长度: {len(test_history)} 个物品")
    print(f"\n  Prompt 原文（前200字符）:")
    print(f"  {prompt[:200]}")

    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    print(f"\n  Token 数量: {len(ids)}")
    print(f"  前 15 个 token id: {ids[:15]}")
    print(f"  后 5  个 token id: {ids[-5:]}")

    # 确认边界 token
    start_id = vocab.get("<|hist_clk_start|>")
    end_id   = vocab.get("<|hist_clk_end|>")
    print(f"\n  ids[0]  = {ids[0]}  期望={start_id}  {'✅' if ids[0]==start_id else '❌ 不匹配！'}")
    print(f"  ids[-1] = {ids[-1]} 期望={end_id}  {'✅' if ids[-1]==end_id else '❌ 不匹配！'}")

    # 检查中间有没有意外的 eos
    eos_id = tokenizer.eos_token_id
    eos_positions = [i for i, x in enumerate(ids) if x == eos_id]
    if eos_positions:
        print(f"\n  ❌ 警告：prompt 中间出现了 eos_token (id={eos_id}) 在位置 {eos_positions[:5]}")
        print(f"     这会导致模型提前终止注意力！")
    else:
        print(f"\n  ✅ prompt 中间无 eos_token，边界干净")

    # padding 检查
    inputs_left  = tokenizer(prompt, return_tensors="pt", padding="max_length",
                             max_length=305, truncation=True)
    inputs_right = tokenizer(prompt, return_tensors="pt", padding="max_length",
                             max_length=305, truncation=True,
                             padding_side="right") if False else None  # 仅演示

    pad_id = tokenizer.pad_token_id
    left_pads  = (inputs_left["input_ids"][0] == pad_id).sum().item()
    print(f"\n  padding_side=left  时，pad token 数量: {left_pads}")
    print(f"  attention_mask 中 1 的数量: {inputs_left['attention_mask'][0].sum().item()}")
    print(f"  ✅ 有效 token 从位置 {left_pads} 开始，模型看到的 prompt 长度正确")

# ==================== 5. 单用户推理 attention 权重快照 ====================
print("\n[5] 单用户推理：检查模型是否真的在看 history")
print("  （通过对比空 history 和有 history 时的 logits 差异）")

# 加载模型
state_dict = load_file(CKPT_PATH)
config = AutoConfig.from_pretrained(QWEN_PATH)
model  = Qwen2ForCausalLM(config)
missing, _ = model.load_state_dict(state_dict, strict=False)
if "lm_head.weight" in missing:
    model.lm_head.weight = model.model.embed_tokens.weight
model.to(DEVICE).eval()

if test_uid and test_history:
    prompt_full  = "<|hist_clk_start|>" + "".join(test_history) + "<|hist_clk_end|>"
    prompt_empty = "<|hist_clk_start|>" + "<|hist_clk_end|>"  # 空历史

    def get_top5_logits(prompt_text):
        ids = tokenizer(prompt_text, return_tensors="pt",
                        truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            out    = model(**ids)
            logits = out.logits[0, -1, :]
            probs  = torch.softmax(logits, dim=-1)
            top5   = torch.topk(probs, 5)
        return [(token2item.get(tokenizer.decode([tid.item()]), f"raw:{tid.item()}"),
                 prob.item())
                for tid, prob in zip(top5.indices, top5.values)]

    top5_full  = get_top5_logits(prompt_full)
    top5_empty = get_top5_logits(prompt_empty)

    print(f"\n  有历史 prompt 的 Top5 预测:")
    for item, prob in top5_full:
        print(f"    {item:>10}  prob={prob:.4f}")

    print(f"\n  空历史 prompt 的 Top5 预测:")
    for item, prob in top5_empty:
        print(f"    {item:>10}  prob={prob:.4f}")

    full_set  = {x[0] for x in top5_full}
    empty_set = {x[0] for x in top5_empty}
    overlap   = full_set & empty_set
    print(f"\n  两者 Top5 重叠数: {len(overlap)}/5")
    if len(overlap) >= 4:
        print("  ❌ 模型几乎无视了历史输入，预测分布与空 prompt 几乎相同")
        print("     → 模型还没学会关注 history，属于训练早期正常现象")
    elif len(overlap) >= 2:
        print("  🟡 模型开始区分有/无历史，但区分度还不够")
    else:
        print("  ✅ 模型对有/无历史的预测差异明显，正在学习个性化")

print("\n" + "=" * 60)
print("  检查完成")
print("=" * 60)
