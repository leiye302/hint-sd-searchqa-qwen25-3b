# Third-party sources and licenses

Root Apache-2.0 LICENSE covers adaptation code, not third-party weights/data.

- [HINT-SD](https://github.com/wgcyeo/HINT-SD), `bc95b44192b06328a88773dbf3b345fa1c499093`: Apache-2.0. Native Trainer, reverse-KL, LoRA/EMA initialization, failure parser, generic rollout record interfaces; fetched unchanged.
- [SDAR](https://github.com/ZJU-REAL/SDAR), `29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`: Apache-2.0. Search-QA prompt/history/environment/reward, original Search-R1 preprocessing and E5/FAISS. Follow upstream notices, including SkyRL-Gym and Search-R1.
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct): Apache-2.0; [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct): Qwen Research License and its restrictions. Built with Qwen. Fixed snapshots in `configs/models.json`; review upstream cards before use/redistribution.
- [Search-R1 QA assets](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train): downloaded separately, not rehosted. Follow each dataset's original terms: NQ, HotpotQA, TriviaQA, PopQA, 2Wiki, MuSiQue, Bamboogle.
- [Wiki18 index](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index), [corpus](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus), [E5-base-v2](https://huggingface.co/intfloat/e5-base-v2): fixed revisions/SHA in `configs/retrieval.json`; follow their cards and original Wikipedia attribution/license. No index/corpus uploaded here.

Independent Search-QA adaptation; no official endorsement or claim of reproducing author benchmark scores. Public-source provenance in `configs/upstream.json` and `configs/comparison.json`.
