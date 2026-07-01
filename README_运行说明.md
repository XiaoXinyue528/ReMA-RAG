# ReMA-RAG项目代码运行说明

本代码包对应《面向PlantUML类图生成的强化多智能体检索增强生成系统》算法模块，包含轨迹生成、LoRA微调、结构化评价、结构奖励后训练和消融实验相关代码。

## 1. 目录结构

```text
project_code_submission_20260630/
├─ trajectory_generation/          # GPT-4o-mini驱动的MA-RAG-UML轨迹生成代码
│  ├─ agents/                      # Planner、Step Definer、Extractor、Coder等智能体逻辑
│  ├─ src/                         # API调用、结构化输出、检索等辅助代码
│  ├─ scripts/                     # 批量运行与进度检查脚本
│  ├─ data_splits/                 # 数据集占位目录，数据集需单独放入
│  ├─ corpus/                      # RAG样例库占位目录
│  └─ emb_corpus/                  # 向量索引/嵌入缓存占位目录
├─ training_rl/                    # 学生模型训练、评价与奖励优化代码
│  ├─ LLaMA-Factory/               # 已包含本项目使用的LLaMA-Factory源码与配置
│  ├─ cloud_scripts/               # 云端环境、SFT、MAPPO/PPO式训练与评估脚本
│  ├─ evaluation/                  # PlantUML结构化解析与评价脚本
│  ├─ patches/                     # PlantUML PPO-v0补丁
│  └─ experiment_patches/          # Coder-SFT、MAPPO-v1/v2/v3、agent-aware消融补丁
├─ analysis_docs/                  # 技术路线、算法融合和实验总结文档
└─ external_assets_placeholder/    # 数据集和模型参数另行存档说明
```

## 2. 本包不包含的内容

按照课程提交要求，本代码包已经删除或不包含以下大文件：

- PlantUCD原始数据集、1127条教师轨迹数据、142条测试集；
- Hugging Face/ModelScope基础模型权重；
- LoRA适配器参数、checkpoint、`.safetensors`、`.bin`等模型参数；
- 训练输出目录、推理预测结果、日志和缓存文件；
- FAISS索引、嵌入缓存等可重建大文件。

上述文件应提交到“模型参数和数据集”链接。恢复运行时，将数据和模型参数复制到下面说明的占位目录即可。

## 3. 数据和模型参数放置位置

恢复完整实验时，建议按照以下位置放置外部文件：

```text
trajectory_generation/data_splits/
  plantucd_sft_140.json
  plantucd_rl_1127.json
  plantucd_test_142.json

trajectory_generation/corpus/
  custom_corpus.jsonl 或同等RAG样例库文件

trajectory_generation/emb_corpus/
  可选：已构建的向量索引和嵌入缓存

training_rl/LLaMA-Factory/data/
  rema_plantuml相关SFT/MAPPO训练json或jsonl
  dataset_info.json

training_rl/LLaMA-Factory/saves/
  llama3-8b/lora/rema_plantuml/full_sft_v4/
  llama3-8b/lora/rema_plantuml/coder_sft_v5/
  llama3-8b/lora/rema_plantuml/mappo_v3_constrained_tiny64/
```

基础模型建议单独放在云端缓存目录，例如：

```text
/hy-tmp/cache/modelscope/LLM-Research/Meta-Llama-3-8B-Instruct
```

如路径不同，需要同步修改`training_rl/LLaMA-Factory/examples/train_lora/`下相关yaml配置中的`model_name_or_path`字段。

## 4. 轨迹生成运行方式

进入轨迹生成目录：

```bash
cd trajectory_generation
cp .env.sample .env
```

在`.env`或环境变量中配置API信息：

```bash
export OPENAI_API_KEY="你的API Key"
export OPENAI_BASE_URL="https://apinexus.net/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_TIMEOUT=120
export OPENAI_MAX_RETRIES=3
export STRUCTURED_OUTPUT_ATTEMPTS=3
export PLANTUCD_DATA_PATH=data_splits/plantucd_rl_1127.json
```

运行指定区间样本：

```bash
python -u main.py \
  --exp plan_rag_extract \
  --model gpt4omini \
  --dataset plantucd_rl1127 \
  --start_index 0 \
  --end_index 9 \
  --gpus 0
```

检查生成进度：

```bash
python scripts/check_progress.py \
  --output-dir plan_rag_extract_gpt4omini_plantucd_rl1127 \
  --expected 1127
```

## 5. SFT训练与结构奖励后训练

进入训练目录：

```bash
cd training_rl/LLaMA-Factory
pip install -e .
```

若使用云端4090环境，可参考：

```bash
cd ../
bash cloud_scripts/00_setup_env.sh
bash cloud_scripts/01_download_llama3_modelscope.sh
bash cloud_scripts/02_prepare_full_sft_adapter.sh
bash cloud_scripts/03_patch_llamafactory.sh
```

Full-SFT训练配置位于：

```text
training_rl/LLaMA-Factory/examples/train_lora/rema_plantuml_full_sft_4090.yaml
```

MAPPO/PPO式结构奖励后训练配置和脚本位于：

```text
training_rl/LLaMA-Factory/examples/train_lora/rema_plantuml_mappo_v0_tiny32.yaml
training_rl/cloud_scripts/04_run_mappo_v0_tiny32.sh
training_rl/experiment_patches/apply_mappo_v3_constrained.py
```

典型运行方式：

```bash
cd training_rl
python experiment_patches/apply_mappo_v3_constrained.py
cd LLaMA-Factory
llamafactory-cli train examples/train_lora/rema_plantuml_mappo_v3_constrained_tiny64.yaml
```

## 6. 评价方式

预测结果评估脚本位于：

```text
training_rl/evaluation/evaluate_predictions_normalized.py
training_rl/evaluation/rema/
```

评价脚本会解析预测PlantUML和参考PlantUML，计算以下指标：

- format_score
- syntax_score
- class_f1
- attribute_f1
- method_f1
- relation_pair_f1
- relation_label_f1
- multiplicity_f1
- total

示例：

```bash
python evaluation/evaluate_predictions_normalized.py \
  --input outputs/your_predictions.jsonl \
  --summary outputs/your_summary.json
```

## 7. 注意事项

1. 本代码包用于课程电子材料中的“项目代码.zip”，不包含数据集和模型参数。
2. 若恢复完整实验，请先从单独提交的“模型参数和数据集”压缩包中复制数据、基础模型和LoRA权重。
3. GPT-4o-mini相关实验依赖API，运行前必须设置`OPENAI_API_KEY`。
4. Llama-3-8B-Instruct训练建议使用24GB显存及以上GPU；仅CPU环境可运行部分数据处理和评价脚本，不建议进行训练。
