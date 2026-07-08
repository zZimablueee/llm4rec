#!/usr/bin/env python3
# python /root/autodl-tmp/qwenrec/gr/overfit_test_3users.py
import os, re, json, torch
from transformers import AutoTokenizer, AutoConfig, Qwen2ForCausalLM
from safetensors.torch import load_file
from torch.optim import AdamW

QWEN_PATH      = "/root/autodl-tmp/qwenrec/qwen_init2"
CKPT_PATH      = "/root/autodl-tmp/qwenrec/cache/checkpoints/global_step_ep10_v2_seq100/checkpoint-1000/model.safetensors"
ITEM2TOKEN_DIR = "/root/autodl-tmp/qwenrec/cache/emb_infer/sinkhorn10"
VAL_SPLIT_PATH = "/root/autodl-tmp/qwenrec/cache/_val_split.json"
DEVICE = "cuda"
STEPS  = 60
LR     = 3e-4

torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
eos_id = tokenizer.eos_token_id

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
model = model.to(torch.bfloat16).to(DEVICE).train()
model.gradient_checkpointing_enable()

with open(VAL_SPLIT_PATH) as f:
    val_data = json.load(f)

selected = []
used_targets = set()
for uid, seq in val_data.items():
    if len(seq) < 4: continue
    hist = [item2token[str(x)] for x in seq[:-1] if str(x) in item2token][-10:]
    tgt  = item2token.get(str(seq[-1]))
    if len(hist) < 3 or not tgt or tgt in used_targets: continue
    selected.append((uid, hist, tgt, seq[-1]))
    used_targets.add(tgt)
    if len(selected) == 3: break

print("选取的 3 个用户：")
for uid, hist, tgt, item_id in selected:
    print(f"  uid={uid}  target={item_id}  tokens={tgt}")

vocab = tokenizer.get_vocab()
samples = []
for _, hist, tgt, _ in selected:
    prompt     = f"<|hist_clk_start|>{''.join(hist)}<|hist_clk_end|>"
    full_text  = prompt + tgt
    prompt_ids = tokenizer(prompt,    add_special_tokens=False)["input_ids"]
    full_ids   = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    input_ids  = torch.tensor([full_ids + [eos_id]], device=DEVICE)
    labels     = torch.tensor(
        [[-100] * len(prompt_ids) + full_ids[len(prompt_ids):] + [eos_id]],
        device=DEVICE
    )
    target_ids = [vocab[t] for t in re.findall(r"<[a-c]_\d+>", tgt)]
    samples.append((prompt, input_ids, labels, target_ids))

optimizer = AdamW(model.parameters(), lr=LR)

print(f"\n{'step':>5}  {'loss':>8}  {'用户1':^35}  {'用户2':^35}  {'用户3':^35}")
print("─" * 120)

def should_print(step):
    # 前20步每步打，之后每5步打一次
    return step <= 20 or step % 5 == 0

for step in range(1, STEPS + 1):
    optimizer.zero_grad()
    total_loss = 0
    for _, input_ids, labels, _ in samples:
        out = model(input_ids=input_ids, labels=labels)
        (out.loss / len(samples)).backward()
        total_loss += out.loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if should_print(step):
        model.eval()
        results = []
        with torch.no_grad():
            for prompt, _, _, target_ids in samples:
                enc = tokenizer(prompt, return_tensors="pt",
                                add_special_tokens=False).to(DEVICE)
                gen = model.generate(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    max_new_tokens=3,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=eos_id,
                )
                new_ids  = gen[0][enc.input_ids.shape[1]:].tolist()
                pred_str = "".join(tokenizer.decode([i]) for i in new_ids)
                ok       = "✅" if new_ids == target_ids else f"❌ {pred_str}"
                results.append(f"{ok:^35}")
        model.train()
        print(f"  {step:>3}  {total_loss/len(samples):>8.4f}  {'  '.join(results)}")

print("\n─" * 120)
