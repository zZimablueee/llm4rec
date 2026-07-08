#!/usr/bin/env python3
# cd /root/autodl-tmp/qwenrec/gr && python comprehensive_diag.py
"""
综合诊断脚本 v3：HR@10/20 + nDCG@10/20，参考值对标论文 RQ-VAE baseline。
"""
import os, re, json, sys, math, torch
import numpy as np
from collections import Counter
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen2ForCausalLM, AutoConfig
from tqdm import tqdm

# ==================== 配置 ====================
BASE_CACHE       = "/root/autodl-tmp/qwenrec/cache"
QWEN_PATH        = "/root/autodl-tmp/qwenrec/qwen_init2"
CKPT_BASE        = "/root/autodl-tmp/qwenrec/cache/checkpoints/global_step_ep10_v2_seq100"
ITEM2TOKEN_DIR   = os.path.join(BASE_CACHE, "emb_infer/sinkhorn10")
VAL_SPLIT_PATH   = os.path.join(BASE_CACHE, "_val_split.json")

ORIG_VOCAB_SIZE  = 151936
NUM_SAMPLE_USERS = 2000
BEAM_SIZE        = 50
TOP_K_LIST       = [10, 20]   # 只测 @10 和 @20
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# RQ-VAE 在三个数据集上的结果（来自 R³-VAE 论文 Table 1）
# 格式：Recall@10, Recall@20, NDCG@10, NDCG@20
PAPER_RQVAE = {
    "Beauty":  {"Recall@10": 0.059, "Recall@20": 0.091, "NDCG@10": 0.033, "NDCG@20": 0.041},
    "Sports":  {"Recall@10": 0.031, "Recall@20": 0.048, "NDCG@10": 0.017, "NDCG@20": 0.021},
    "Toys":    {"Recall@10": 0.051, "Recall@20": 0.074, "NDCG@10": 0.028, "NDCG@20": 0.035},
}

# ==================== 工具函数 ====================
def find_latest_checkpoint(base):
    if not os.path.isdir(base):
        print(f"❌ 输出目录不存在: {base}"); sys.exit(1)
    ckpts = [d for d in os.listdir(base)
             if d.startswith("checkpoint-") and os.path.isdir(os.path.join(base, d))]
    if not ckpts:
        print("❌ 没有任何 checkpoint"); sys.exit(1)
    latest = max(ckpts, key=lambda x: int(x.split("-")[1]))
    return os.path.join(base, latest)

def print_section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def ndcg_at_k(hit_position, k):
    """hit_position: 1-based，未命中返回 0.0"""
    if hit_position is None or hit_position > k:
        return 0.0
    return 1.0 / math.log2(hit_position + 1)

# ==================== 1. 训练信号健康度 ====================
def check_training_log(ckpt_dir):
    print_section("1️⃣  训练信号健康度")
    state_path = os.path.join(ckpt_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        print("  ⚠️  trainer_state.json 不存在，跳过"); return

    with open(state_path) as f:
        state = json.load(f)

    log_history = [x for x in state.get("log_history", []) if "loss" in x]
    if not log_history:
        print("  ⚠️  暂无 loss 记录"); return

    print(f"  当前步数 : {state.get('global_step', '?')}")
    print(f"  最佳指标 : {state.get('best_metric', '?')}")
    print(f"\n  Loss 趋势（均匀采样最多 8 个点）:")
    step = max(1, len(log_history) // 8)
    for entry in log_history[::step]:
        lr_str = f"{entry['learning_rate']:.2e}" if "learning_rate" in entry else "?"
        print(f"    step {entry['step']:>6d} | loss = {entry['loss']:.4f} | lr = {lr_str}")

    first_loss = log_history[0]["loss"]
    last_loss  = log_history[-1]["loss"]
    delta      = last_loss - first_loss
    print(f"\n  Loss 变化: {first_loss:.4f} → {last_loss:.4f}  ({delta:+.4f})")
    has_nan = any(math.isnan(e["loss"]) for e in log_history)
    if has_nan:          print("  ❌ 检测到 NaN loss！")
    elif delta > -0.05:  print("  ❌ Loss 几乎没有下降！")
    elif delta > -0.5:   print("  🟡 Loss 缓慢下降，继续观察")
    else:                print("  ✅ Loss 下降明显，模型在正常学习")

# ==================== 2. 加载模型和字典 ====================
def load_model_and_dicts(ckpt_dir):
    print_section("2️⃣  加载模型和字典")
    print(f"  使用 checkpoint: {os.path.basename(ckpt_dir)}")

    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    config = AutoConfig.from_pretrained(QWEN_PATH)
    model  = Qwen2ForCausalLM(config)

    ckpt_path = os.path.join(ckpt_dir, "model.safetensors")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    if not os.path.exists(ckpt_path):
        print("  ❌ 未找到模型权重文件"); sys.exit(1)

    state_dict = load_file(ckpt_path) if ckpt_path.endswith(".safetensors") \
                 else torch.load(ckpt_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if "lm_head.weight" in missing:
        model.lm_head.weight = model.model.embed_tokens.weight
        print("  ⚠️  lm_head 缺失，已从 embed_tokens 恢复（tied weights）")

    model.to(DEVICE)
    model.eval()
    print(f"  模型加载完成，设备: {DEVICE}")

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
    print(f"  item2token 大小: {len(item2token)}")
    return tokenizer, model, item2token, token2item

# ==================== 3. 嵌入层检查 ====================
def check_embedding_update(model):
    print_section("3️⃣  新 token 嵌入层状态")
    emb     = model.model.embed_tokens.weight.float()
    new_emb = emb[ORIG_VOCAB_SIZE:]
    print(f"  当前词表大小  : {emb.shape[0]}")
    print(f"  新增 token 数 : {new_emb.shape[0]}")
    if new_emb.shape[0] == 0:
        print("  ⚠️  词表未扩充"); return
    std_val  = new_emb.std().item()
    orig_std = emb[:ORIG_VOCAB_SIZE].std().item()
    ratio    = std_val / orig_std if orig_std > 0 else 0
    print(f"  新 token std  : {std_val:.6f}")
    print(f"  原始 token std: {orig_std:.6f}")
    print(f"  std 比值      : {ratio:.3f}  {'✅ 嵌入正常更新' if ratio > 0.1 else '❌ 嵌入未充分更新'}")

# ==================== 4. 核心评估：Recall/HR + nDCG @ 10/20 ====================
def check_diversity_and_metrics(tokenizer, model, item2token, token2item):
    print_section("4️⃣  推荐质量评估（Recall & nDCG @ 10 / 20）")

    # 构建 Trie
    vocab = tokenizer.get_vocab()
    trie  = {}
    for token_str in item2token.values():
        tokens = re.findall(r'<[a-c]_\d+>', token_str)
        if len(tokens) != 3: continue
        a = vocab.get(tokens[0])
        b = vocab.get(tokens[1])
        c = vocab.get(tokens[2])
        if None in (a, b, c): continue
        trie.setdefault(a, {}).setdefault(b, set()).add(c)
    valid_a = list(trie.keys())
    eos_id  = tokenizer.eos_token_id
    print(f"  Trie 根节点数: {len(valid_a)}")

    with open(VAL_SPLIT_PATH) as f:
        val_data = json.load(f)
    all_users = list(val_data.keys())
    users = list(np.random.choice(
        all_users, min(NUM_SAMPLE_USERS, len(all_users)), replace=False
    ))

    all_top1       = []
    all_preds_flat = []
    num_valid      = 0
    hits  = {k: 0   for k in TOP_K_LIST}
    ndcgs = {k: 0.0 for k in TOP_K_LIST}

    for uid in tqdm(users, desc="  推理中"):
        seq = val_data[uid]
        if len(seq) < 2: continue
        ground_truth = str(seq[-1])
        history = [item2token[str(x)] for x in seq[:-1] if str(x) in item2token][-50:]
        if not history: continue

        prompt   = "<|hist_clk_start|>" + "".join(history) + "<|hist_clk_end|>"
        inputs   = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(DEVICE)
        base_len = inputs["input_ids"].shape[1]

        def prefix_fn(batch_id, input_ids, _len=base_len):
            gen = input_ids[_len:]
            if len(gen) == 0:   return valid_a
            elif len(gen) == 1: return list(trie.get(gen[0].item(), {}).keys())
            elif len(gen) == 2: return list(trie.get(gen[0].item(), {}).get(gen[1].item(), set()))
            else:               return [eos_id]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4,
                num_beams=BEAM_SIZE,
                num_return_sequences=BEAM_SIZE,
                prefix_allowed_tokens_fn=prefix_fn,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_id,
                early_stopping=False,
            )

        preds = []
        for k in range(BEAM_SIZE):
            gen_text = tokenizer.decode(outputs[k][base_len:], skip_special_tokens=False)
            match    = re.search(r"<a_\d+><b_\d+><c_\d+>", gen_text)
            if match:
                item = token2item.get(match.group(0))
                if item and item not in preds:
                    preds.append(item)

        if not preds: continue
        num_valid += 1
        all_top1.append(preds[0])
        all_preds_flat.extend(preds[:max(TOP_K_LIST)])

        # 找命中位置（1-based）
        hit_pos = None
        for rank, pred in enumerate(preds, start=1):
            if pred == ground_truth:
                hit_pos = rank
                break

        for k in TOP_K_LIST:
            if hit_pos is not None and hit_pos <= k:
                hits[k]  += 1
                ndcgs[k] += ndcg_at_k(hit_pos, k)

    if num_valid == 0:
        print("  ❌ 没有用户能生成有效推荐"); return

    # -------- 结果表格 --------
    r10  = hits[10]  / num_valid
    r20  = hits[20]  / num_valid
    n10  = ndcgs[10] / num_valid
    n20  = ndcgs[20] / num_valid

    print(f"\n  评估用户数: {num_valid}  |  数据集: Electronics")
    print(f"\n  {'':4} {'Recall@10':>10} {'Recall@20':>10} {'NDCG@10':>10} {'NDCG@20':>10}")
    print(f"  {'-'*46}")
    print(f"  {'你的模型':<4} {r10:>10.4f} {r20:>10.4f} {n10:>10.4f} {n20:>10.4f}")
    print(f"  {'-'*46}")

    # 打印三个数据集的 RQ-VAE 参考值
    for ds, vals in PAPER_RQVAE.items():
        print(f"  {'RQ-VAE/'+ds:<14} "
              f"{vals['Recall@10']:>10.4f} "
              f"{vals['Recall@20']:>10.4f} "
              f"{vals['NDCG@10']:>10.4f} "
              f"{vals['NDCG@20']:>10.4f}")
    print(f"  {'-'*46}")
    print(f"  ⚠️  注：论文三个数据集规模均 < 36K 用户，Electronics 为 728K 用户，直接数值对比仅供参考")

    # -------- 多样性分析 --------
    unique_top1 = len(set(all_top1))
    diversity   = unique_top1 / num_valid
    print(f"\n  Top1 多样性: {unique_top1}/{num_valid} = {diversity:.2%}")
    if diversity < 0.3:
        print("  ❌ 多样性低，模型存在热门坍缩")
    elif diversity < 0.6:
        print("  🟡 多样性尚可，模型正在分化")
    else:
        print("  ✅ 多样性良好")

    print(f"\n  最常推荐的 10 个物品（高频 = 坍缩信号）:")
    for item, cnt in Counter(all_preds_flat).most_common(10):
        print(f"    item {item}: {cnt} 次")

    # -------- 决策建议 --------
    print_section("💡  是否上推理的决策建议")
    # 参考基准：Sports RQ-VAE 的 50%（难度修正后的合理下限）
    threshold_r10 = PAPER_RQVAE["Sports"]["Recall@10"] * 0.5   # 0.0155
    threshold_r20 = PAPER_RQVAE["Sports"]["Recall@20"] * 0.5   # 0.0240

    print(f"  当前 Recall@10 = {r10:.4f}  （难度修正参考线 ≥ {threshold_r10:.4f}）")
    print(f"  当前 Recall@20 = {r20:.4f}  （难度修正参考线 ≥ {threshold_r20:.4f}）")

    if r10 >= threshold_r10 and r20 >= threshold_r20:
        print("\n  ✅ 指标达到难度修正后的合理水平，可以考虑上全量推理")
        print("  建议：先用 1 万用户跑小批量推理验证管道，再全量")
    elif r20 >= threshold_r20 * 0.7:
        print("\n  🟡 指标接近参考线，建议再训练 5000~10000 步后复测")
        print(f"  当前 Recall@20={r20:.4f}，目标 ≥ {threshold_r20:.4f}")
    else:
        print("\n  ❌ 指标偏低，建议继续训练")
        print(f"  当前 Recall@20={r20:.4f}，目标 ≥ {threshold_r20:.4f}")

# ==================== 5. 管道完整性 ====================
def check_pipeline_integrity(item2token, tokenizer):
    print_section("5️⃣  管道完整性")
    max_hist  = 100
    dummy_tok = "<a_0><b_0><c_0>"
    prompt    = "<|hist_clk_start|>" + dummy_tok * max_hist + "<|hist_clk_end|>" + dummy_tok
    ids       = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    print(f"  最大序列 token 数: {len(ids)}（需 ≤ 305）")
    print(f"  {'✅ 序列长度安全' if len(ids) <= 305 else '❌ 序列超长！'}")

    special_tokens = ["<|hist_clk_start|>", "<|hist_clk_end|>"]
    vocab = tokenizer.get_vocab()
    for tok in special_tokens:
        print(f"  {'✅' if tok in vocab else '❌'} 特殊 token: {tok}")

    missing = sum(
        1 for token_str in list(item2token.values())[:500]
        for t in re.findall(r"<[a-c]_\d+>", token_str)
        if t not in vocab
    )
    print(f"  item token 缺失（抽查500）: {missing}  {'✅' if missing == 0 else '❌'}")

# ==================== 主入口 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("       生成式推荐综合诊断工具 v3")
    print("=" * 60)

    ckpt_dir = find_latest_checkpoint(CKPT_BASE)
    check_training_log(ckpt_dir)
    tokenizer, model, item2token, token2item = load_model_and_dicts(ckpt_dir)
    check_embedding_update(model)
    check_diversity_and_metrics(tokenizer, model, item2token, token2item)
    check_pipeline_integrity(item2token, tokenizer)

    print("\n" + "=" * 60)
    print("           诊断完成")
    print("=" * 60)
