import json

# 读取训练序列
with open('/root/autodl-tmp/qwenrec/cache/train_data.json') as f:
    train_data = json.load(f)

test_input = {}   # 用户历史（去掉最后一个）
test_target = {}  # 测试目标（最后一个）

for user_id, item_seq in train_data.items():
    if len(item_seq) < 2:
        continue  # 序列太短跳过
    test_input[user_id] = item_seq[:-1]   # 去掉最后一个
    test_target[user_id] = item_seq[-1]   # 最后一个作为ground truth

with open('/root/autodl-tmp/qwenrec/cache/test_input.json', 'w') as f:
    json.dump(test_input, f)
with open('/root/autodl-tmp/qwenrec/cache/test_target.json', 'w') as f:
    json.dump(test_target, f)

print(f"测试用户数: {len(test_target)}")
print(f"样例 - 用户1 输入: {test_input.get('1', [])}")
print(f"样例 - 用户1 目标: {test_target.get('1', '')}")
