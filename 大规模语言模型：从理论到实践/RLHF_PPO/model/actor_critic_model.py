import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
from utils.tools import Tools
from peft import LoraConfig, get_peft_model, PeftModel
from config import LoraArguments


class LoraModel(PeftModel):
    def __init__(self, config: Config, model):
        lora_args = LoraArguments()
        lora_config = LoraConfig(
            r=lora_args.lora_r, ## 把秩降到这个数。
            lora_alpha=lora_args.lora_alpha, ## 这个是一个扩张系数。
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            task_type="CAUSAL_LM",
        )
        super().__init__(model, lora_config)
        self.v_head = torch.nn.Linear(1024, 1, bias=False).to(config.device)
        if lora_args.is_reload_trained_params:
            super().from_pretrained(model, config.save_lora_path)
            self.v_head.load_state_dict(torch.load(config.save_v_head_path))
        for name, module in self.named_modules():
            if 'lora_' in name:
                for param in module.parameters():
                    param.requires_grad = True

    def forward(self, input_ids, attention_mask, tools: Tools):
        res = super().forward(input_ids, attention_mask, output_hidden_states=True)

        ## res.hidden_states 就是每一个单词的，算是embedding吧。应该为1024维。
        ## 所以这边就直接把这1024维用一个全连接层转换为1维。也就是把每一个单词的embedding从1024维压缩为1维。
        values = self.v_head(res.hidden_states[0]).squeeze(-1)[:, :-1]
        values = tools.filter_mask(values)

        ## res.logits应该表示的是每一个输出的位置上，“为词表中各个单词的概率”。
        ## 然后probs就是各个输出词的对应概率。
        probs = tools.probs_from_logits(res.logits[:, :-1, :], input_ids[:, 1:])
        probs = tools.filter_mask(probs)
        return probs, values


class ActorCriticLoraModel(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        ## xmk：加载一个原始的文本生成模型。
        model = AutoModelForCausalLM.from_pretrained(config.gpt_model).to(config.device).eval()
        ##
        self.model = LoraModel(config, model) ## xmk：我的理解就是ho，传入一些lora系列的参数，使得我们能够对原来的model里面的特定部分做lora
        self.tokenizer = AutoTokenizer.from_pretrained(config.gpt_model)

    def forward(self, input_ids, attention_mask, tools: Tools):
        probs, values = self.model(input_ids, attention_mask, tools)
        return probs, values

    @torch.no_grad()
    def actor_generate(self, input_ids):
        generated_ids = self.model.generate(input_ids, max_new_tokens=512, top_p=1.0,
                                            num_beams=1,
                                            do_sample=False)
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        response_id = generated_ids[:, input_ids.shape[1]:]
        return generated_ids, response, response_id
