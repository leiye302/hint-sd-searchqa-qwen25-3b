# HINT-SD-Multi · Search-QA · Qwen2.5-3B-Instruct

Four A100 **80GB** GPUs; 150 optimizer updates; standalone public-source adaptation.

本仓库运行 **Qwen2.5-3B-Instruct 的 HINT-SD-Multi**，不是 WebShop、GRPO、SkillSD 或 RETHINK 方法。算法来自[官方 HINT-SD](https://github.com/wgcyeo/HINT-SD)，Search-QA 环境、检索器、提示词及奖励来自[官方 SDAR](https://github.com/ZJU-REAL/SDAR)。**SDAR 没有 HINT-SD 入口**，不能将 GRPO+OPSD 改名当作 HINT-SD。

两个原仓库固定提交并保持源文件不改；不需要本机文件、私有服务器、付费 API、人工 hint 或已有 checkpoint。直接调用作者的 Trainer、蒸馏损失、选步规则、EMA 和 LoRA。适配清单见 [METHOD](docs/METHOD.md)；评测细节见 [EVALUATION](docs/EVALUATION.md)。

## 固定设置

| 项目 | 配置 |
| --- | --- |
| 起点 | Qwen/Qwen2.5-3B-Instruct，固定 revision；Built with Qwen |
| 方法 | HINT-SD-Multi，EMA teacher；失败轨迹最多选3个关键决策 |
| 更新量 | 150；batch1 × 梯度累积2 × 每题4条 rollout = **8条轨迹/更新** |
| LoRA | q_proj/v_proj；rank32、alpha64、dropout0.05 |
| 优化 | LR5e-6，原版linear/5%warmup，EMA0.999，Student top100+tail reverse KL |
| GPU | 第1张训练；其余3张独立TP1 vLLM；原生E5/FAISS共享这四卡 |
| 环境 | 最多4动作、history4、prompt4096左截断、response512、检索top3 |
| 评测 | 0/20/40/60/80/100/120/140：固定448题；150：完整51,713题一次 |
| 断点 | 每20更新和最终150保存，完整且SHA验证后才轮换，最新两份 |

HINT-SD **没有SDAR的beta门控**，不硬加beta=5。此前GRPO系列是128题×8=1024轨迹/更新；本仓库保留原版HINT-SD的8轨迹/更新。相同150更新**不是相同采样量或计算预算**。

## 1. 安装（Linux/CUDA）

建议CUDA12.8兼容驱动、4×A10080GB、256GiB系统内存、350–500GiB可用磁盘。原版Wiki18 E5索引合并后约64.56GB，另有原始分块、语料和环境。40GB卡不属于此配方；不要通过换检索器或缩减语料冒充同口径。

安装Git、[uv](https://docs.astral.sh/uv/)和Conda：

```bash
git clone https://github.com/leiye302/hint-sd-searchqa-qwen25-3b.git
cd hint-sd-searchqa-qwen25-3b
bash scripts/setup.sh
```

脚本拉取两个公共固定源码版本，为HINT-SD按作者 `uv.lock` 创建Python>=3.13环境，并创建 `.env-searchqa` Python3.12 CPU环境。**不要混装SDAR旧训练依赖**。单独准备原生GPU检索环境：

```bash
conda create -y --prefix "$PWD/.env-retriever" python=3.10
conda install -y --prefix "$PWD/.env-retriever" -c pytorch -c nvidia faiss-gpu=1.8.0
conda run --prefix "$PWD/.env-retriever" python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
conda run --prefix "$PWD/.env-retriever" python -m pip install -r configs/requirements-retriever.txt
```

三个环境都在checkout内，避免原版HINT-SD与原生FAISS依赖冲突。控制入口用Python3.10+即可，训练由作者锁定环境启动。`.env.example` 只有可选配置，无密钥。

## 2. 固定模型、QA数据与检索资产

```bash
.env-searchqa/bin/python -m hint_searchqa.setup --download-model 3b
.env-searchqa/bin/python src/data.py --prepare --verify
.env-searchqa/bin/python src/retrieval.py --prepare --verify
.env-searchqa/bin/python -m hint_searchqa.setup --verify
.env-searchqa/bin/python scripts/check_release.py
.env-searchqa/bin/python src/env_smoke.py
```

revision和SHA均在 `configs/`。训练NQ+HotpotQA共169,615题；完整测试七源51,713题。预处理、448题选择与先前Search-QA仓库一致。监控子集排除规范化题文本与训练的精确重叠；原始全量测试的10条精确文本重叠记录在收据中，**不暗改全集**。

`src/env_smoke.py` 是真实SearchEnv＋本地HTTP检索 **fixture**，只验证协议/反馈/归一化奖励，不代表真实E5质量或Actor性能。下载大索引不属于训练单步耗时。

## 3. 检查、训练与续跑

仅用自己获分配且空闲的四张80GB卡。编号不同时传 `--gpus 4,5,6,7`。

```bash
bash scripts/train.sh --dry-run
# 独立1更新＋2题GPU smoke，不计作448/51713题正式分数
bash scripts/train.sh --smoke
# 正式150更新
bash scripts/train.sh
# 同一模型、数据、配置、代码，从最新完整断点继续
bash scripts/train.sh --resume
```

默认输出 `runs/hint_sd_3b/`，拒绝覆盖已有目录。控制器自动启动本轮自己的原生E5检索、CPU环境和训练/推理，**不要另开占用8000端口的检索服务**；可用 `--retriever-port` 更换端口。QA服务只监听loopback；原版E5服务监听所有接口，需服务器防火墙限制外部访问。占卡/占端口检查不会杀他人进程；退出只结束自己启动的进程组。

日志：`training.log`、`retriever.log`、`environment.log`、`metrics.jsonl`、`rollouts/train.jsonl`、`feedback.jsonl`。断点包含Student/EMA adapter、optimizer、scheduler、RNG、Trainer状态与绑定收据，不只是LoRA权重。新断点完整校验后只保留最近两份；损坏/未完成文件保留作证据。150更新**先存完整断点，再测全量**；全量中断后resume只重试评测，不训练151。`final_adapter/` 不是可续训断点。

## 4. 独立评测

```bash
bash scripts/eval.sh --eval-base --eval-scope subset
bash scripts/eval.sh --eval-checkpoint runs/hint_sd_3b/checkpoint-150 --eval-scope full --output runs/final_full_eval
.env-searchqa/bin/python src/report.py --run-dir runs/hint_sd_3b
```

输出目录必须是新的。验证是 **Student独立贪心rollout**，不生成Teacher反馈、不向Student给hint。与先前两个尺度Search-QA仓库同题目/顺序/预算/EM计分；HINT-SD不复用GRPO的Ray Trainer，不声称训练代码相同。

每次分批存 `evaluation/update_NNN/attempt-*/`；只有题目完整且ID/顺序/来源无替换、重复或遗漏才发布summary。报告逐题micro EM、七源macro EM；subset/full曲线分开。448题约10分钟只是目标，**不是超时截断规则**；真实GPU耗时需实测。

## 验证与许可边界

包含CPU原生反向传播/EMA/存盘恢复测试、SearchEnv HTTP测试、评测边界测试和Linux干净检出CI。**CPU通过不等于4×A100正式训练、vLLM多卡或全量E5检索已验证**；首次运行请先做GPU smoke。不随仓库发布训练成绩或私有实验资料。第三方许可见 [THIRD_PARTY](THIRD_PARTY.md)，3B模型须遵守Qwen Research License。
