from torch.utils.data import DataLoader
from utils.data_load import CustomDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

dataset = CustomDataset(
    "/Users/minkexiu/Documents/GitHub/ML_runCodeFromBook/大规模语言模型：从理论到实践/RLHF_PPO/data/train_data.json",
    AutoTokenizer.from_pretrained('/Users/minkexiu/Downloads/GitHub/ML_runCodeFromBook/大规模语言模型：从理论到实践/RLHF_PPO/trained_models/Qwen1.5-0.5B-Chat')
)

# print(dataset.)


from config import Config
from transformers import AutoTokenizer

config = Config()
tokenizer = AutoTokenizer.from_pretrained(config.gpt_model)
# 创建自定义数据集实例
dataset = CustomDataset(config.data_path, tokenizer)

# 创建数据加载器并指定批次大小
batch_size = 2
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)

# 使用生成器函数按需读取数据
for batch in data_loader:
    print(batch)
    # 在每个批次中进行模型训练
    # batch 包含了一个批次的样本数据
    # 在这里执行模型训练操作
    pass