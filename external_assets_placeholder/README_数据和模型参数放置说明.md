# 数据和模型参数放置说明

本目录仅用于说明外部资产如何恢复，不存放真实数据集和模型参数。

请从单独提交的“模型参数和数据集”压缩包中恢复以下内容：

```text
../trajectory_generation/data_splits/
../trajectory_generation/corpus/
../trajectory_generation/emb_corpus/
../training_rl/LLaMA-Factory/data/
../training_rl/LLaMA-Factory/saves/
```

基础模型可放在云端缓存目录，例如：

```text
/hy-tmp/cache/modelscope/LLM-Research/Meta-Llama-3-8B-Instruct
```

如果实际路径不同，请修改`../training_rl/LLaMA-Factory/examples/train_lora/`中的yaml配置。
