import pickle
with open('indexer.pkl', 'rb') as f:
    idx = pickle.load(f)
    print("包含的特征字段:", idx['f'].keys())
    # 随便看一个特征的前几个映射
    feat_name = list(idx['f'].keys())[0]
    print(f"特征 {feat_name} 的前5个编码:", list(idx['f'][feat_name].items())[:5])