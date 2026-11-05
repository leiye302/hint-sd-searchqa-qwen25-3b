# 共享Search-QA评测口径

训练169,615题：NQ79,168＋HotpotQA90,447。测试51,713题：NQ3,610、TriviaQA11,313、PopQA14,267、HotpotQA7,405、2Wiki12,576、MuSiQue2,417、Bamboogle125。数据revision及SHA在 `configs/data.json`。

监控448题＝七源各64题、seed2026。两尺寸与先前六个GRPO/GRPO+OPSD/RLSD仓库同题目ID/顺序/来源，不重复随机抽取。子集排除规范化题文本与训练的精确重叠及内部重复；全量保留原始全集，已知10条训练精确文本重叠写入收据，不把精确文本去重说成语义去重。

0/20/40/60/80/100/120/140测448题；150只测一次完整51,713题。监控有序ID SHA为 `29d67a3da4b11211f8cfd1cd95c286375936c3e075a80109207c273c332b7bda`；全量为 `b1c79fb8529c7ddbd1051e008995676baa5bb9dbe3c6d34a1ade2c0d2f14d427`。

预处理、选择、校验及汇总等15个公共文件与先前仓库逐字节相同，见 `configs/comparison.json`。**HINT-SD的rollout/evaluate与GRPO Ray Trainer不同**，本仓库保证评测协议一致，不声称所有代码相同。

Student独立贪心：temperature0、n1、最多4动作、history4、prompt4096左截断、response512、原版top3检索。原生parser和奖励不改，不按reasoning格式筛题。Teacher/EMA反馈不参与验证；ground-truth只在原生CPU答案验证器，不进入Actor prompt或Teacher反馈。

每题完整episode只计一次normalized Exact Match：最后完整answer与参考aliases按原生规则匹配。micro EM＝总答对/题数；macro EM＝七源EM平均。两者不混用。

每批512题，尾批不丢弃。ID/顺序/来源/总数逐项校验；分批保存到 `evaluation/update_NNN/attempt-*/batch_*.json`，全部完成才发布summary/latest。失败保留证据但不把部分结果当正式分数。再次评测用新attempt；subset/full曲线分开，2题GPU smoke显式标smoke。

最终150先保存包含Student/EMA/optimizer/scheduler/RNG的完整断点并SHA验收，再做全量；中断后resume只恢复150并评测，不做151更新。独立评测需显式checkpoint/scope和新输出目录。

448题约10分钟是目标耗时，不是超时停止规则；实际时间写入wall_seconds/questions_per_second，未实测前不保证。HTTP CPU fixture验证环境，不是实际E5/FAISS检索或Actor分数。
