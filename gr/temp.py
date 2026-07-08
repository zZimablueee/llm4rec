cat > /tmp/check_labels.py << 'EOF'
import sys, json, os
sys.path.insert(0, '/root/autodl-tmp/qwenrec')

from transformers import HfArgumentParser
from arguments import ModelArguments, DataTrainingArguments, TrainingArguments
from utils import load_model_and_tokenizer, load_item2token_dict, BASE_CACHE
from custom_dataset import CustomTrainDataset

# 加载 config
config_path = '/root/autodl-tmp/qwenrec/gr/train_config.json'  # 你的实际 config 路径
with open(config_path) as f:
    config = json.load(f)

config_dict = {}
for _, value_dict in config.items():
    for k, v in value_dict.items():
        config_dict[k] = v

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_dict(config_dict, allow_extra_keys=True)

model, tokenizer = load_model_and_tokenizer(model_args)
item2token_dict = load_item2token_dict(os.path.join(BASE_CACHE, data_args.item2token_dict))

train_tmp = os.path.join(BASE_CACHE, "_train_split.json")
dataset = CustomTrainDataset(
    json_path=train_tmp,
    item2token_dict=item2token_dict,
    tokenizer=tokenizer,
    data_args=data_args
)

print("=== Label 检查（前5个样本）===\n")
for i in range(5):
    sample = dataset[i]
    input_ids = sample['input_ids'].tolist()
    labels    = sample['labels'].tolist()

    non_masked = [(idx, tok, lab)
                  for idx, (tok, lab) in enumerate(zip(input_ids, labels))
                  if lab != -100]

    print(f"--- Sample {i} ---")
    print(f"  总 token 数:        {len(input_ids)}")
    print(f"  非-100 label 数量:  {len(non_masked)}  ← 期望是 3")
    for idx, tok, lab in non_masked:
        print(f"    pos={idx:4d}  token_id={tok:6d}  decoded='{tokenizer.decode([tok])}'  label={lab}")
    print()
EOF

# python3 /root/autodl-tmp/qwenrec/gr/temp.py
