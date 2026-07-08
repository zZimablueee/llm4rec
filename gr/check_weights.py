#!/usr/bin/env python3

# python /root/autodl-tmp/qwenrec/gr/check_weights.py

"""
检查 lm_head 和 embed_tokens 是否真的共享同一块内存
以及新增 token 在输出端的 logit 初始值是否正常
"""
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoConfig, Qwen2ForCausalLM

QWEN_PATH  = "/root/autodl-tmp/qwenrec/qwen_init2"
CKPT_PATH  = "/root/autodl-tmp/qwenrec/cache/checkpoints/global_step_ep10_v2_seq100/checkpoint-1000/model.safetensors"

tokenizer  = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
config     = AutoConfig.from_pretrained(QWEN_PATH)
model      = Qwen2ForCausalLM(config)
state_dict = load_file(CKPT_PATH)
missing, _ = model.load_state_dict(state_dict, strict=False)

if "lm_head.weight" in missing:
    model.lm_head.weight = model.model.embed_tokens.weight
    print("⚠️  lm_head 缺失，已手动绑定")
else:
    print("✅ lm_head 权重在 checkpoint 中存在")

print("\n[1] 权重是否真的共享同一块内存（data_ptr 相同）")
ptr_embed = model.model.embed_tokens.weight.data_ptr()
ptr_lmhead = model.lm_head.weight.data_ptr()
print(f"  embed_tokens.weight data_ptr : {ptr_embed}")
print(f"  lm_head.weight      data_ptr : {ptr_lmhead}")
if ptr_embed == ptr_lmhead:
    print("  ✅ 同一块内存，tied weights 生效")
else:
    print("  ❌ 不同内存！lm_head 和 embed_tokens 是两份独立权重！")
    print("     → 新增 token 在输出端可能是随机初始化，logit 极低")

print("\n[2] 新增 token 在 lm_head 输出端的权重统计")
orig_vocab = 151936
new_vocab  = len(tokenizer)
W = model.lm_head.weight.data  # shape: [vocab_size, hidden_dim]

old_w = W[:orig_vocab]
new_w = W[orig_vocab:]

print(f"  原始 token 权重  均值={old_w.mean():.6f}  std={old_w.std():.6f}")
print(f"  新增 token 权重  均值={new_w.mean():.6f}  std={new_w.std():.6f}")

ratio = new_w.std() / old_w.std()
print(f"  新/旧 std 比值: {ratio:.3f}")
if ratio < 0.1:
    print("  ❌ 新增 token 输出权重几乎是零！模型输出端无法感知新 token！")
elif ratio < 0.5:
    print("  ⚠️  新增 token 输出权重偏小，输出端学习信号较弱")
else:
    print("  ✅ 新增 token 输出权重正常")

print("\n[3] 嫌疑物品 vs 随机物品的 lm_head 输出权重 L2 norm 对比")
import os, re
ITEM2TOKEN_DIR = "/root/autodl-tmp/qwenrec/cache/emb_infer/sinkhorn10"
vocab = tokenizer.get_vocab()

item2token = {}
for fname in os.listdir(ITEM2TOKEN_DIR):
    fpath = os.path.join(ITEM2TOKEN_DIR, fname)
    if not os.path.isfile(fpath): continue
    with open(fpath) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                item2token[parts[0]] = parts[1]

def get_item_lmhead_norm(item_id):
    tok_str = item2token.get(item_id, "")
    tokens  = re.findall(r"<[a-c]_\d+>", tok_str)
    norms   = []
    for t in tokens:
        tid = vocab.get(t)
        if tid is not None:
            norms.append(W[tid].norm().item())
    return tokens, norms

suspects = ["1703", "25475", "8599", "11950", "2181"]
randoms  = list(item2token.keys())[500:505]  # 随机抽5个

print(f"\n  {'item':>10}  {'tokens':>30}  {'lm_head L2 norms'}")
print(f"  {'─'*70}")
print("  --- 嫌疑物品 ---")
for item in suspects:
    toks, norms = get_item_lmhead_norm(item)
    print(f"  {item:>10}  {str(toks):>30}  {[f'{n:.3f}' for n in norms]}")

print("  --- 随机物品 ---")
for item in randoms:
    toks, norms = get_item_lmhead_norm(item)
    print(f"  {item:>10}  {str(toks):>30}  {[f'{n:.3f}' for n in norms]}")
