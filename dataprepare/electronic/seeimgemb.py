import pickle
import numpy as np

# 文件路径
pkl_path = "/root/autodl-tmp/qwenrec/dataprepare/electronic/item_img_emb_512.pkl"

print("正在加载图像特征文件...")
with open(pkl_path, 'rb') as f:
    img_emb_dict = pickle.load(f)

# 1. 顶层类型
print(f"顶层数据类型: {type(img_emb_dict)}")

# 2. 总元素数
print(f"总元素数: {len(img_emb_dict)}")

# 3. 看前两个键和对应 value 的信息
keys = list(img_emb_dict.keys())
print(f"前5个键: {keys[:5]}")

if keys:
    first_key = keys[0]
    val = img_emb_dict[first_key]
    print(f"\n键的类型: {type(first_key)}")
    print(f"第一个键的值: {first_key}")
    print(f"对应值的类型: {type(val)}")
    
    # 如果是 numpy 数组，看形状和数据类型
    if isinstance(val, np.ndarray):
        print(f"  数组形状: {val.shape}")
        print(f"  数组数据类型: {val.dtype}")
        print(f"  前5个值: {val[:5]}")
    else:
        # 防止是 list 等其他类型，也输出信息
        print(f"  值的前5个元素（如果是可索引的话）: {val[:5] if hasattr(val, '__getitem__') else val}")

# 4. 检查是否所有 value 都是 512 维向量
shapes = set()
for v in img_emb_dict.values():
    if isinstance(v, np.ndarray):
        shapes.add(v.shape)
print(f"\n所有唯一向量形状: {shapes}")

# 5. 统计缺失情况（比如有没有 None）
none_count = sum(1 for v in img_emb_dict.values() if v is None)
print(f"值为 None 的个数: {none_count}")

print("\n✅ 检查完成")