import os, json, argparse, torch, gc
from pathlib import Path
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from corpus.retrieve import Retriever
# 注意这里我们去掉了 load_dataset，直接使用内置库解析
from src.utils import load_corpus, parse_args, GraphState 
from agents.plan_executor import build_plan_executor
from agents.plan import plan_agent
from langgraph.graph import StateGraph, START
from dotenv import load_dotenv

load_dotenv()

def resolve_bge_model_path() -> str:
    explicit_path = os.getenv("BGE_MODEL_PATH")
    if explicit_path:
        return explicit_path

    snapshot_roots = []
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        snapshot_roots.append(Path(hf_home) / "hub" / "models--BAAI--bge-base-zh-v1.5" / "snapshots")
    snapshot_roots.append(Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-base-zh-v1.5" / "snapshots")

    valid_snapshots = []
    for root in snapshot_roots:
        if not root.exists():
            continue
        for snapshot in root.iterdir():
            if not snapshot.is_dir():
                continue
            config_path = snapshot / "config.json"
            has_safetensors = (snapshot / "model.safetensors").exists()
            has_bin = (snapshot / "pytorch_model.bin").exists()
            has_weights = has_safetensors or has_bin
            has_tokenizer = (snapshot / "tokenizer.json").exists() or (snapshot / "vocab.txt").exists()
            if not (config_path.exists() and has_weights and has_tokenizer):
                continue
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if config.get("model_type") or config.get("architectures"):
                valid_snapshots.append(snapshot)

    if valid_snapshots:
        valid_snapshots.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        if not (valid_snapshots[0] / "model.safetensors").exists() and (valid_snapshots[0] / "pytorch_model.bin").exists():
            os.environ.setdefault("BGE_USE_SAFETENSORS", "0")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        return str(valid_snapshots[0])

    return "BAAI/bge-base-zh-v1.5"

class MARAGRetrieverTool:
    def __init__(self, tokenizer, embedding_model, retrieval_model, corpus, top_k=10):
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.retrieval_model = retrieval_model
        self.corpus = corpus
        self.top_k = top_k
        
    def __call__(self, query: str):
        prefixed_query = f"为这个句子生成表示以用于检索相关文章：{query}"
        inputs = self.tokenizer([prefixed_query], max_length=512, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.embedding_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embs = F.normalize(embeddings, p=2, dim=1)
        
        embs_np = embs.cpu().numpy().astype('float32')
        top_doc = self.retrieval_model.search([0], embs_np, top_k=self.top_k)
        
        list_docs, list_doc_ids = [], []
        for doc_id in top_doc[0].keys():
            tmp = None
            try:
                for att_id in [str(doc_id), str(int(doc_id)+1), str(int(doc_id)-1)]:
                    tmp = self.corpus.get(att_id)
                    if tmp: break
            except: pass
            if tmp:
                list_docs.append(tmp.text)
                list_doc_ids.append(tmp.doc_id)
        
        if not list_docs:
            list_docs = ["No relevant documents found."]
            list_doc_ids = ["none"]
            
        return list_docs, list_doc_ids

def plan_executor_node(state: GraphState):
    input = {"original_question": state["original_question"], "plan": state["plan"], "stop": False}
    output = plan_executor_agent.invoke(input, {"recursion_limit": 50})
    return {"past_exp": [output]}

class LocalCorpusStore:
    def __init__(self, jsonl_path):
        self.store = {}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                class DummyDoc:
                    def __init__(self, doc_id, text):
                        self.doc_id = doc_id
                        self.text = text
                self.store[str(item["doc_id"])] = DummyDoc(str(item["doc_id"]), item["text"])
    
    def get(self, doc_id):
        return self.store.get(str(doc_id))

if __name__ == "__main__":
    args = parse_args()
    
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    model_path = resolve_bge_model_path()
    
    print(f"[*] 正在加载模型: {model_path} ...")
    offline = os.getenv("HF_HUB_OFFLINE", "0") == "1" or Path(model_path).exists()
    use_safetensors = os.getenv("BGE_USE_SAFETENSORS", "1") == "1"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=offline)
    model = AutoModel.from_pretrained(
        model_path,
        use_safetensors=use_safetensors,
        local_files_only=offline,
    )
    model.eval().to(f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu")
    print("[*] 模型加载完成。")
    
    STORAGE_DIR = "emb_corpus/plantuml-gte-ml"
    
    retrieve = Retriever(gpu_ids=[])
    retrieve.init_index_and_add(root_dir=STORAGE_DIR, dataset_name="plantuml_db")
    
    local_corpus = LocalCorpusStore("custom_corpus.jsonl")
    
    retriever_tool = MARAGRetrieverTool(tokenizer, model, retrieve, local_corpus, top_k=10)
    
    global plan_executor_agent
    plan_executor_agent = build_plan_executor(retriever_tool=retriever_tool)

    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("planer_node", plan_agent)
    graph_builder.add_node("plan_executor_node", plan_executor_node)
    graph_builder.add_edge(START, "planer_node")
    graph_builder.add_edge("planer_node", "plan_executor_node")
    graph = graph_builder.compile()

    save_dir = f"{args.exp}_{args.model}_{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)
    
    # ================= 核心修改区域 =================
    # 暴力绕过 utils 的坑，直接读取你截图里的真实原始文件
    data_path = os.getenv("PLANTUCD_DATA_PATH", "data_splits/plantucd_rl_1127.json")
    # 指向我们刚切出来的 10% SFT 热身集
    
    print(f"[*] 正在直接加载原始数据集: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"[*] 成功加载，共 {len(dataset)} 条数据，准备精准提取 HumanLang 和 PlantUML。")

    for id, item in enumerate(dataset):
        if id < args.start_index or id > args.end_index: continue
        
        # 因为真实文件没有自带 id 字段，我们按序生成唯一标识
        question_id = f"plantucd_test_{id}"
        save_file = os.path.join(save_dir, f"{question_id}.json")
        
        # 确保能重新覆盖运行，避免受到旧文件的干扰
        if os.path.exists(save_file):
            print(f"Skip existing output: {save_file}")
            continue
            
        print(f"\n--- 处理第 {id} 题: {question_id} ---")
        
        # 直接抓取你截图里的键名
        q_text = item.get("HumanLang", "")
        inputs = {"original_question": f"{q_text}?"}
        
        try:
            with torch.no_grad():
                output = graph.invoke(inputs, {"recursion_limit": 50})
            
            # 直接抓取你截图里的真实答案
            gold_answer = item.get("PlantUML", "未提取到标准答案")
            output["ground_truth"] = str(gold_answer)
            
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False)
            print("保存成功")
            
            del output
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"❌ 题目 {question_id} 出错跳过, 原因: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            continue
