#!/usr/bin/env python3

# cd /root/autodl-tmp/qwenrec/gr/overfit_test.py

import os, re, json, torch
from transformers import AutoTokenizer, AutoConfig, Qwen2ForCausalLM
from safetensors.torch import load_file
from torch.optim import AdamW

QWEN_PATH      = "/root/autodl-tmp/qwenrec/qwen_init2"
CKPT_PATH      = "/root/autodl-tmp/qwenrec/cache/checkpoints/global_step_ep10_v2_seq100/checkpoint-1000/model.safetensors"
ITEM2TOKEN_DIR = "/root/autodl-tmp/qwenrec/cache/emb_infer/sinkhorn10"
VAL_SPLIT_PATH = "/root/autodl-tmp/qwenrec/cache/_val_split.json"
DEVICE         = "cuda"
STEPS          = 100
LR             = 3e-4

# ── 先清显存 ──────────────────────────────────────────────
torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

item2token = {}
for fname in os.listdir(ITEM2TOKEN_DIR):
    fpath = os.path.join(ITEM2TOKEN_DIR, fname)
    if not os.path.isfile(fpath): continue
    with open(fpath) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                item2token[parts[0]] = parts[1]

config = AutoConfig.from_pretrained(QWEN_PATH)
model  = Qwen2ForCausalLM(config)

state_dict = load_file(CKPT_PATH)
missing, _ = model.load_state_dict(state_dict, strict=False)
if "lm_head.weight" in missing:
    model.lm_head.weight = model.model.embed_tokens.weight

# ── bf16 + 梯度检查点，省显存 ─────────────────────────────
model = model.to(torch.bfloat16).to(DEVICE).train()
model.gradient_checkpointing_enable()

# ── 取第一个有效用户 ──────────────────────────────────────
with open(VAL_SPLIT_PATH) as f:
    val_data = json.load(f)

uid, seq, history_str, target_str = None, None, None, None
for u, s in val_data.items():
    hist = [item2token[str(x)] for x in s[:-1] if str(x) in item2token][-10:]
    tgt  = item2token.get(str(s[-1]))
    if len(hist) >= 3 and tgt:
        uid         = u
        seq         = s
        history_str = "".join(hist)
        target_str  = tgt
        break

print(f"过拟合测试用户: uid={uid}")
print(f"target item   : {seq[-1]}  →  {target_str}")

prompt    = f"<|hist_clk_start|>{history_str}<|hist_clk_end|>"
full_text = prompt + target_str

prompt_ids = tokenizer(prompt,    add_special_tokens=False)["input_ids"]
full_ids   = tokenizer(full_text, add_special_tokens=False)["input_ids"]
eos_id     = tokenizer.eos_token_id

input_ids = torch.tensor([full_ids + [eos_id]], device=DEVICE)
labels    = torch.tensor(
    [[-100] * len(prompt_ids) + full_ids[len(prompt_ids):] + [eos_id]],
    device=DEVICE
)

vocab           = tokenizer.get_vocab()
target_tokens   = re.findall(r"<[a-c]_\d+>", target_str)
target_ids_list = [vocab[t] for t in target_tokens]

optimizer = AdamW(model.parameters(), lr=LR)

print(f"\n{'step':>6}  {'loss':>10}  {'预测 tokens':>40}  {'correct':>8}")
print("─" * 70)

for step in range(1, STEPS + 1):
    optimizer.zero_grad()
    out  = model(input_ids=input_ids, labels=labels)
    loss = out.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 10 == 0:
        model.eval()
        with torch.no_grad():
            prompt_tensor = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(DEVICE)
            gen = model.generate(
                input_ids=prompt_tensor,
                max_new_tokens=3,
                do_sample=False,
                num_beams=1
            )
        new_ids   = gen[0][prompt_tensor.shape[1]:].tolist()
        pred_toks = [tokenizer.decode([i]) for i in new_ids]
        correct   = "✅" if new_ids == target_ids_list else "❌"
        print(f"  {step:>4}  {loss.item():>10.4f}  {str(pred_toks):>40}  {correct}")
        model.train()

print("\n─" * 70)
print("判定：")
print("  loss < 0.5 且最终预测正确 → ✅ 训练逻辑无误，步数不够而已")
print("  loss 一直 > 2.0           → ❌ 底层有硬伤")
