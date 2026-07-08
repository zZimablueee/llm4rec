#!/usr/bin/env python3

# cd /root/autodl-tmp/qwenrec/gr && python check_labels.py

"""
检查训练样本的 input_ids / labels 是否正确
核心问题：labels 中 -100 的范围是否精确覆盖且仅覆盖 prompt 部分
"""
import sys, os
sys.path.insert(0, "/root/autodl-tmp/qwenrec/gr")

import json
import torch
from transformers import AutoTokenizer

# ==================== 改成你自己的路径 ====================
QWEN_PATH       = "/root/autodl-tmp/qwenrec/qwen_init2"
TRAIN_SCRIPT    = "/root/autodl-tmp/qwenrec/gr/train_gr.py"
VAL_SPLIT_PATH  = "/root/autodl-tmp/qwenrec/cache/_val_split.json"
ITEM2TOKEN_DIR  = "/root/autodl-tmp/qwenrec/cache/emb_infer/sinkhorn10"
# ==========================================================

tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

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

# 加载验证集，取前3个用户手动构造训练样本
with open(VAL_SPLIT_PATH) as f:
    val_data = json.load(f)

print("=" * 70)
print("  Labels 正确性检查")
print("=" * 70)

# 尝试直接 import 你的 Dataset 类
try:
    # 动态 import train_gr.py 里的 Dataset
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_gr", TRAIN_SCRIPT)
    train_module = importlib.util.load_from_spec(spec)  # 会报错，走 except
except:
    train_module = None

# ── 手动模拟训练样本构造 ──────────────────────────────────────
# 按照你训练代码的逻辑：
#   input  = <|hist_clk_start|> + history_tokens + <|hist_clk_end|> + target_tokens
#   labels = [-100] * prompt_len + target_token_ids + [eos_id]

checked = 0
for uid, seq in val_data.items():
    if len(seq) < 2:
        continue

    history_items = [str(x) for x in seq[:-1] if str(x) in item2token][-10:]
    target_item   = str(seq[-1])

    if target_item not in item2token:
        continue
    if len(history_items) < 2:
        continue

    history_str = "".join(item2token[h] for h in history_items)
    target_str  = item2token[target_item]

    prompt    = f"<|hist_clk_start|>{history_str}<|hist_clk_end|>"
    full_text = prompt + target_str

    prompt_ids = tokenizer(prompt,     add_special_tokens=False)["input_ids"]
    full_ids   = tokenizer(full_text,  add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_str, add_special_tokens=False)["input_ids"]

    eos_id = tokenizer.eos_token_id
    input_ids = full_ids + [eos_id]
    labels    = [-100] * len(prompt_ids) + target_ids + [eos_id]

    print(f"\n{'─'*70}")
    print(f"  用户 uid={uid}  |  历史长度={len(history_items)}  |  target={target_item}")
    print(f"{'─'*70}")
    print(f"  prompt token 数   : {len(prompt_ids)}")
    print(f"  target token 数   : {len(target_ids)}  → {target_str}")
    print(f"  input_ids 总长度  : {len(input_ids)}")
    print(f"  labels 总长度     : {len(labels)}")

    # 长度一致性
    if len(input_ids) != len(labels):
        print(f"  ❌ 致命错误：input_ids 和 labels 长度不一致！")
        print(f"     input_ids={len(input_ids)}  labels={len(labels)}")
    else:
        print(f"  ✅ 长度一致")

    # -100 覆盖范围
    mask_positions    = [i for i, x in enumerate(labels) if x == -100]
    nonmask_positions = [i for i, x in enumerate(labels) if x != -100]

    print(f"\n  labels 中 -100 的位置: [{mask_positions[0]}  ~  {mask_positions[-1]}]  共 {len(mask_positions)} 个")
    print(f"  labels 中有梯度的位置: [{nonmask_positions[0]} ~ {nonmask_positions[-1]}]  共 {len(nonmask_positions)} 个")

    # 关键检查1：-100 是否精确覆盖 prompt
    expected_mask_end = len(prompt_ids) - 1
    if mask_positions[-1] == expected_mask_end:
        print(f"  ✅ -100 精确覆盖 prompt（位置 0~{expected_mask_end}），没有多也没有少")
    elif mask_positions[-1] > expected_mask_end:
        over = mask_positions[-1] - expected_mask_end
        print(f"  ❌ -100 多覆盖了 {over} 个 target token！模型看不到这部分目标！")
    else:
        print(f"  ⚠️  -100 覆盖不足，prompt 末尾有 {expected_mask_end - mask_positions[-1]} 个 token 被当成目标")

    # 关键检查2：有梯度部分能否 decode 回 target
    grad_token_ids = [labels[i] for i in nonmask_positions]
    decoded = tokenizer.decode(grad_token_ids, skip_special_tokens=False)
    expected_decoded = target_str + tokenizer.decode([eos_id])

    print(f"\n  有梯度部分 decode 结果 : {decoded}")
    print(f"  期望结果               : {expected_decoded}")

    if target_str in decoded:
        print(f"  ✅ 有梯度部分正确对应 target item token 序列")
    else:
        print(f"  ❌ 有梯度部分与 target 不符！训练目标可能错误！")

    # 关键检查3：全是 -100 的极端情况
    if len(nonmask_positions) == 0:
        print(f"\n  ❌ 致命：labels 全是 -100，模型完全没有梯度！")
    elif len(nonmask_positions) <= 2:
        print(f"\n  ⚠️  有梯度的 token 只有 {len(nonmask_positions)} 个，信号极弱")

    # 打印前20个 token 对照表
    print(f"\n  {'pos':>4}  {'input_id':>10}  {'label':>10}  {'input_decode':>20}  {'label_decode':>20}")
    print(f"  {'─'*70}")
    for i in range(min(len(input_ids), 20)):
        inp_tok = tokenizer.decode([input_ids[i]])
        lbl_str = tokenizer.decode([labels[i]]) if labels[i] != -100 else "[-100]"
        marker  = "◀ target start" if i == len(prompt_ids) else ""
        print(f"  {i:>4}  {input_ids[i]:>10}  {labels[i]:>10}  {inp_tok:>20}  {lbl_str:>20}  {marker}")

    checked += 1
    if checked >= 3:
        break

print(f"\n{'='*70}")
print(f"  检查了 {checked} 条样本")
print(f"{'='*70}")
