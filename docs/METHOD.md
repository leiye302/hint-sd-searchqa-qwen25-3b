# HINT-SD与Search-QA适配边界

算法：[wgcyeo/HINT-SD](https://github.com/wgcyeo/HINT-SD)，`bc95b44192b06328a88773dbf3b345fa1c499093`；环境：[ZJU-REAL/SDAR](https://github.com/ZJU-REAL/SDAR)，`29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`。`configs/upstream.json`逐文件锁住两套未修改原源码。

## 保留原版方法

从Qwen2.5-Instruct开始，不做RETHINK SFT。每题采样4条轨迹。失败轨迹交给EMA Teacher回看，最多选择3个关键失败决策并给局部反馈；成功轨迹不蒸馏。空/非法反馈、无有效目标按作者原版skip/noop逻辑，不回退成GRPO。

Student条件是原先实际可见前缀；Teacher在**相同实际Student前缀**上追加反馈，对同一Student采样completion评分：

\[
\mathcal L=\mathbb E_{t\in S,\,v\sim\pi_\theta(\cdot\mid x_t)}
\left[\log\pi_\theta(v\mid x_t)-\log\pi_{\bar\theta}(v\mid x_t,h_t)\right],
\qquad\bar\theta\leftarrow0.999\bar\theta+0.001\theta.
\]

直接调用作者reverse-KL：Student top100＋剩余概率tail桶；selected completion/EOS按原版mask，包含其中的reasoning，并非只监督action内部。EMA无梯度，溢出时跳过更新。LoRA q/v r32、alpha64、dropout.05；LR5e-6、原版linear/5%warmup；batch1、累积2、rollout4；反馈温度.6、512tokens、轨迹12000chars、最多3步JSON/parser不改。无GRPO、SDAR sigmoid gate、reference KL或RETHINK注册。

## 必要适配

1. 环境改为SDAR原生Search-QA：原始问题prompt、SearchMemory、动作projection、SearchEnv转移/终止/归一化EM和top3 Wiki18 E5检索不改。
2. 反馈提示仅替换任务术语为检索QA的证据/实体/查询/答案错误，JSON结构不改；不注入参考答案、不调用付费API。Teacher可看完整失败轨迹，Student不看未来反馈。
3. 作者的BFCL Multi记录/选步接口作为通用DTO：`backend=bfcl` **不加载BFCL数据或环境**，实际环境是SearchEnv、训练数据为QA Parquet。
4. 一训练进程＋三TP1 vLLM；每次Student/EMA同步用新LoRA ID避免旧缓存，不改优化器/目标。原生检索服务共享同四张分配卡。
5. history4是原生最近search/info历史，不是所有旧reasoning。prompt经原生Qwen模板后左截断4096；记录真实prompt/completion IDs，Teacher从该前缀派生，避免重建完整不可见历史。
6. 评测改为共享448/51713口径；存盘仅在作者序列化外增加完整性/SHA/原子发布/最新两份轮换。

## 比较边界

150是本实验更新预算，不声称所有作者环境原来都训150。HINT-SD原版8轨迹/更新，GRPO系1024；相同更新次数不是相同采样预算。Teacher更新方式、LoRA/全参数、训练量应分开报告。两个HINT-SD尺寸除backbone/入口标签外方法、数据、评测相同。

CPU小模型只能验证梯度、EMA、checkpoint和接口；不代表GPU收敛，不声称复现作者BFCL/AppWorld论文分数。
