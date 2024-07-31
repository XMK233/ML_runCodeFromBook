import torch
from config import Config
from utils.tools import Tools


class PPO:
    def __init__(self, actor_critic_model, config: Config, actor_critic_opt):
        self.actor_critic_model = actor_critic_model
        self.config = config
        self.actor_critic_opt = actor_critic_opt

    def train(
            self,
            prompt_generate_ids, ## generate出来的回复。感觉回复得最像一句话。包括了prompt和纯回答的单词id。
            attention_mask, ## （顾名思义）
            prob_refs, ## 感觉应该是纯回答里各个单词的概率。这个概率怎么生成的呢：首先用 prompt_generate_ids 和对应的mask，送到reference_model的forward里面去得到logits，然后把logits
                    ## 转为概率，再把 纯回答的单词id位置 对应的概率拿出来，就是 prob_refs。所以这里也是只有纯回答的。
            reward, ## 感觉好像就是表示，各个纯回答的情感是正面还是负面。
            tools: Tools
    ):
        with torch.no_grad():
            _, old_values = self.actor_critic_model(prompt_generate_ids, attention_mask, tools)  # 计算每个token的价值
            ## 这个old_values是用lora_model的forward生成的。
            ## xmk：我实验了一下，lora_model的forward和reference_model的forward应该不是一回事。
            ## 我原本以为，lora_model的forward就是其初始化的时候用的model的forward，也就是跟reference_model是一回事。
            ## 但我把 lora_model 和 reference_model打出来，发现它们的结构还是有所不同。所以推测两个forward不是一回事。
            ## 话说回来，如果是一回事，那还有什么意义，不就起不到对比的效果了嘛。
        for _ in range(self.config.ppo_epochs):
            # 获得 actor_critic 模型新的 probs 和 token 对应的价值。
            ## 我这边看了一眼，感觉forward方法返回的probs，values
            ## 中的probs，就是模型的原始forward方法返回的各个单词对应的logits；
            ## 而values呢，则是模型的原始forward方法返回的各个单词的算是embedding的东西用全连接层映射成的1维变量。
            new_probs, new_values = self.actor_critic_model(prompt_generate_ids, attention_mask, tools)
            # 计算奖励值
            ## 其实就是判断new_probs和probs_ref之间的差异，差异越大，reward越小。
            ## 底下返回的non_score_rewards是原始的、每个单词对应的reward。
            ## 而rewards变量，则是将non_score_rewards里面最后一个单词的reward加上【reward这个参数变量】得到的。
            ## 总之这里代码写得有点糊涂，参数reward，返回还叫reward，整一个给我闹麻了。
            rewards, non_score_rewards = self.compute_rewards(reward, new_probs, prob_refs)  # 计算reward

            ## 【TODO】计算损失。
            loss = self.loss(
                new_probs=new_probs, ## 被训练模型的，新的，模型的原始forward方法返回的各个单词对应的logits。
                old_values=old_values, ## 被训练模型的，原来的，模型的原始forward方法返回的各个单词的算是embedding的东西用全连接层映射成的1维变量。
                new_values=new_values, ## 被训练模型的，新的，模型的原始forward方法返回的各个单词的算是embedding的东西用全连接层映射成的1维变量。
                rewards=rewards, ## 每一个单词的一个reward，前面算出来的，是最后一个单词加了【reward这个参数变量】得到的。算是用 old_probs new_probs 和【reward这个参数变量】算出来的。
                old_probs=prob_refs, ## 参考模型的，原始forward方法返回的各个单词对应的logits。
            )

            self.actor_critic_opt.zero_grad()
            loss.backward()
            self.actor_critic_opt.step()
            print(loss)

    def loss(self, new_probs, old_values, new_values, rewards, old_probs):
        """
        计算actor模型和评价模型的loss
        :param new_probs: actor模型生成的probs
        :param old_values: ppo 优化之前的价值
        :param new_values: ppo 优化过程中新的价值
        :param rewards: 每次生成token对应的奖励
        :param old_probs: reference模型生成的probs
        :return: actor loss 和 critic loss
        """
        """Calculate policy and value losses."""
        loss = torch.tensor(0.0)
        for new_prob, old_value, new_value, reward, old_prob in zip(new_probs, old_values, new_values, rewards,
                                                                    old_probs):
            new_prob = new_prob.unsqueeze(0)
            old_value = old_value.unsqueeze(0)
            new_value = new_value.unsqueeze(0)
            reward = reward.unsqueeze(0)
            old_prob = old_prob.unsqueeze(0)
            last_gae_lam = 0
            advantages_reversed = []
            gen_len = new_prob.shape[1]
            # GAE 计算优势函数，当前token获得的奖励(真实的) + 未来获得的价值(这个是上帝视角，不包含当前token) - 包含当前token在上帝视角下的价值
            # 当前token获得的奖励(真实的) + 未来获得的价值(这个是上帝视角，不包含当前token) 比 包含当前token在上帝视角下的价值 要准
            for t in reversed(range(gen_len)):
                next_values = old_value[:, t + 1] if t < gen_len - 1 else 0.0
                delta = reward[:, t] + self.config.gamma * next_values - old_value[:, t]
                last_gae_lam = delta + self.config.gamma * self.config.lam * last_gae_lam
                advantages_reversed.append(last_gae_lam)
            advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
            returns = advantages + old_value  # Q值，当前token获得的奖励(真实的) + 未来获得的价值(这个是上帝视角，不包含当前token)
            advantages = self.whiten(advantages)
            advantages = advantages.detach()
            value_clipped = torch.clamp(new_value,
                                        old_value - self.config.cliprange_value,
                                        old_value + self.config.cliprange_value)  # 截断防止训练废了
            vf_loss1 = (new_value - returns) ** 2  # 上帝视角的价值减去Q值的误差，用于优化上帝模型
            vf_loss2 = (value_clipped - returns) ** 2
            vf_loss = torch.mean(torch.max(vf_loss2, vf_loss1))

            ratio = torch.exp(new_prob - old_prob)  # 控制优化范围，防止训练离原始模型偏差过大
            pg_losses = -advantages * ratio  # importance sampling
            pg_losses2 = -advantages * torch.clamp(ratio,
                                                   1.0 - self.config.cliprange,
                                                   1.0 + self.config.cliprange)  # 截断防止训练废了
            pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
            loss += pg_loss + self.config.vf_coef * vf_loss
        return loss

    def compute_rewards(self, scores, probs, ref_probs):
        """
        计算reward值,由于对每一个token不能给与即使的奖励，这里使用kl散度补偿
        :param scores:reward model给出的奖励值，每条句子只有一个值
        :param probs: actor model生成的probs
        :param ref_probs: reference model 生成的probs
        :return: 返回每个token的奖励值
        """
        rewards, non_score_rewards = [], []
        for score, prob, ref_prob in zip(scores, probs, ref_probs):
            kl = prob - ref_prob  # (seq_len, )

            ## 首先算基础的reward，就是不掺入score的reward。
            non_score_reward = -self.config.kl_ctl_value * kl  # (seq_len, )
            non_score_rewards.append(non_score_reward)

            ## 然后呢，复制一份reward，在这份里面的最后一位掺入score。
            reward = non_score_reward.clone()  # 前面每一个token的reward都来自KL惩罚
            reward[-1] += score  # 在最后一位加上人工给的reward
            rewards.append(reward)

        ## 所以返回的部分是两部分，第一部分是掺入了score的、对于每一个词的奖励
        return rewards, non_score_rewards  # (batch, seq_len)

    @staticmethod
    def whiten(values, shift_mean=True):
        """
        归一化
        :param values: 要归一化的值
        :param shift_mean: 负一化方式
        :return: 返回归一化之后的结果
        """
        mean, var = torch.mean(values), torch.var(values)
        whitened = (values - mean) * torch.rsqrt(var + 1e-8)
        if not shift_mean:
            whitened += mean
        return whitened
