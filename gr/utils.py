import os
from transformers import AutoTokenizer, Qwen2ForCausalLM, AutoConfig

BASE_CACHE = "/root/autodl-tmp/qwenrec/cache"
QWEN_PATH  = "/root/autodl-tmp/qwenrec/qwen_init2"

def load_model_and_tokenizer(model_args):
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token          # ← 就这一行，救命用的
    config = AutoConfig.from_pretrained(QWEN_PATH)
    model = Qwen2ForCausalLM(config)
    return model, tokenizer

def load_item2token_dict(item2token_dict_path):
    item2token_dict = dict()
    for file in os.listdir(item2token_dict_path):
        path = os.path.join(item2token_dict_path, file)
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                item_id, token = line.split('\t')
                item2token_dict[item_id] = token
    print("length of item2token_dict is:", len(item2token_dict))
    return item2token_dict
