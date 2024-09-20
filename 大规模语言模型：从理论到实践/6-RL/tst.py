import random, os, tqdm, time, json
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

import sys
sys.path.append("../../../../")

random.seed(618)
np.random.seed(907)

new_base_path = os.path.join(
    "/Users/minkexiu/Downloads/",
    "/".join(
        os.getcwd().split("/")[-1*(len(sys.path[-1].split("/")) - 1):]
    ),
)
print("storage dir:", new_base_path)
print("code dir:", os.getcwd())

## 创建文件夹。
if not os.path.exists(new_base_path):
    os.makedirs(
        new_base_path
    )
if not os.path.exists(os.path.join(new_base_path, "preprocessedData")):
    os.makedirs(
        os.path.join(new_base_path, "preprocessedData")
    )
if not os.path.exists(os.path.join(new_base_path, "originalData")):
    os.makedirs(
        os.path.join(new_base_path, "originalData")
    )
if not os.path.exists(os.path.join(new_base_path, "trained_models")):
    os.makedirs(
        os.path.join(new_base_path, "trained_models")
    )

def create_originalData_path(filename_or_path):
    return os.path.join(new_base_path, "originalData", filename_or_path)
def create_preprocessedData_path(filename_or_path):
    return os.path.join(new_base_path, "preprocessedData", filename_or_path)
def create_trained_models_path(filename_or_path):
    return os.path.join(new_base_path, "trained_models", filename_or_path)

def millisec2datetime(timestamp):
    time_local = time.localtime(timestamp/1000)
    return time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    
def run_finish():
    # 假设你的字体文件是 'myfont.ttf' 并且位于当前目录下  
    font = FontProperties(fname="/Users/minkexiu/Documents/GitHub/ML_Tryout/SimHei.ttf", size=24)  
    # 创建一个空白的图形  
    fig, ax = plt.subplots()  
    ax.imshow(
        plt.imread("/Users/minkexiu/Downloads/wallhaven-dgxpyg.jpg")
    )
    # 在图形中添加文字  
    ax.text(
        ax.get_xlim()[1] * 0.5, 
        ax.get_ylim()[0] * 0.5, 
        f"程序于这个点跑完：\n{millisec2datetime(time.time()*1000)}", fontproperties=font, ha="center", va="center", color="red"
    )  
    # 设置图形的布局  
    # ax.set_xlim(0, 1)  
    # ax.set_ylim(0, 1)  
    ax.set_xticks([])  
    ax.set_yticks([])  
    ax.patch.set_color("blue")
    # 显示图形  
    plt.show()
        
tqdm.tqdm.pandas() ## 引入这个，就可以在apply的时候用progress_apply了。

import IPython
def kill_current_kernel():
    '''杀死当前的kernel释放内存空间。'''
    IPython.Application.instance().kernel.do_shutdown(True) 
    
def simply_show_data(df1):
    print(df1.shape)
    display(df1.head())
    
def wait_flag(saved_flag_path, time_interval_sec=10):
    print("waiting for", saved_flag_path)
    time_count = 0
    while True:
        if os.path.exists(saved_flag_path):
            break
        time.sleep(time_interval_sec)
        time_count+=time_interval_sec
        print(time_count, end=" ")
    print("finish!!")


from transformers import pipeline, set_seed
import json

def generate_examples(prompt_list, model_name='gpt2', max_length=50, num_return_sequences=2, seed=42):
    generator = pipeline('text-generation', model=model_name, device=0)
    set_seed(seed)
    examples = []
    for prompt in prompt_list:
        result = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
        example = {'prompt': prompt}
        for i, res in enumerate(result):
            answer = res['generated_text'].lstrip().removeprefix(prompt).strip()
            example[f'answer{i + 1}'] = answer
        examples.append(example)
        print(json.dumps(example, indent=2))
    return examples

prompts = [
    "What is the latest news on the stock market?",
    "What is the current state of the economy?",
    "What are the latest developments in technology?",
    "What is the political situation in the Middle East?",
    "What are the latest trends in fashion and beauty?",
    "What are the top travel destinations for this year?",
    "What are some healthy recipes for a vegan diet?",
    "What are the most important events happening in the world today?",
    "What are some tips for improving mental health?",
    "What are the best ways to save money for retirement?",
    "What are some popular new books or movies?",
    "What are some effective ways to reduce stress?",
    "What are the latest developments in artificial intelligence?",
    "What are some top-rated restaurants in your city?",
    "What are the best ways to stay fit and healthy?",
    "What are some tips for successful entrepreneurship?",
    "What are some effective ways to improve productivity?",
    "What are the latest developments in climate change research?",
    "What are some top-rated TV shows or movies on streaming services?",
    "What are some fun activities to do on weekends?",
    "What are some effective ways to manage time and prioritize tasks?",
    "What are the latest trends in home decor and design?",
    "What are the best ways to develop a successful career?",
    "What are some popular new products or gadgets?",
    "What are some effective ways to improve communication skills?",
    "What are some tips for successful relationships?",
    "What are the latest developments in space exploration?",
    "What are some top-rated online courses or certifications?",
    "What are some effective ways to improve public speaking skills?",
    "What are the latest trends in digital marketing?",
    "What are some fun and creative DIY projects?",
    "What are some effective ways to improve leadership skills?"
]

# generated_examples = generate_examples(prompts)

# # Save generated examples to import in Label Studio
# with open('ls_input_data.json', 'w') as f:
#     json.dump(generated_examples, f, indent=2)

import codecs

# This file is generated by Label Studio after completing annotations
data_path = 'ls_export_data.json'

with codecs.open(data_path, 'r', encoding='utf-8') as f:
      data = json.load(f)
# print(data)

sys.path.append("trlx/examples/summarize_rlhf/reward_model/")

import os

import torch
from datasets import load_dataset
from reward_model import GPTRewardModel
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments

def create_comparison_dataset_ls(path: str):
    with codecs.open(data_path, 'r', encoding='utf-8') as f:
          data = json.load(f)
    pairs = []
    for sample in data:
        chosen = None
        rejected = None
        for annotation in sample['annotations']:
            if annotation['result'][0]['value']['selected'] == 'left':
                chosen = sample['data']['prompt'] + '\n' + sample['data']['answer1']
                rejected = sample['data']['prompt'] + '\n' + sample['data']['answer2']
            else:
                chosen = sample['data']['prompt'] + '\n' + sample['data']['answer2']
                rejected = sample['data']['prompt'] + '\n' + sample['data']['answer1']
            pair = {
                'chosen': chosen,
                'rejected': rejected
            }
            pairs.append(pair)
    return pairs

class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        for pair in tqdm(pairs):
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer(
                "<|startoftext|>" + chosen + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_encodings_dict = tokenizer(
                "<|startoftext|>" + rejected + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
            self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
            self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
            self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
        )


class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
        return batch


def compute_metrics(eval_preds):
    chosen_end_scores = eval_preds.predictions[0]  # chosen scores
    rejected_end_scores = eval_preds.predictions[1]  # rejected scores

    result = {}
    acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    result["accuracy"] = acc

    return result

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

if not os.path.exists("rm_checkpoint"):
    os.mkdir("rm_checkpoint")

# Initialize the reward model from the GPT-2 model (optionally SFT GPT-2)
model = GPTRewardModel("gpt2")

# Freeze the first 70% of the hidden layers of the reward model backbone
layers = model.transformer.h
num_layers = len(layers)
num_unfrozen = int(0.3 * num_layers)
for layer in layers[:-num_unfrozen]:
    layer.requires_grad_(False)

# Create the comparisons datasets
pairs = create_comparison_dataset_ls(data_path)
train_size = int(0.8 * len(pairs))  # 80% training, 20% validation
train_pairs = pairs[0:train_size]
val_pairs = pairs[train_size:]


# Make pairwise datasets for training
max_length = 550
train_dataset = PairwiseDataset(train_pairs, tokenizer, max_length=max_length)
val_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)

# Create the collator to gather batches of pairwise comparisons
data_collator = DataCollatorReward()
