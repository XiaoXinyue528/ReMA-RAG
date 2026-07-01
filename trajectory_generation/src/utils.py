import ir_datasets
import pandas as pd
import torch
import json
import os
import torch.nn.functional as F
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from typing import NamedTuple
from ir_datasets.util import DownloadConfig, home_path, Cache, ZipExtract, GzipExtract, LocalDownload
from ir_datasets.formats import TsvDocs, TsvQueries, TrecQrels
from typing import Annotated, Sequence, Literal, Optional, Union
from typing_extensions import TypedDict, List
from pydantic import BaseModel, Field
import operator

def load_dataset_ir(name, split):
    nq_dataset = ir_datasets.load(f"{name}/{split}")
    df = pd.DataFrame(nq_dataset.queries_iter())
    nq_q_dict = {k: {"q": q, "a": a} for k, q, a in zip(df['query_id'].to_list(), df['text'].to_list(), df['answers'].to_list())}
    corpus = nq_dataset.docs_store()
    df_meta = pd.DataFrame(nq_dataset.qrels_iter())
    return corpus, nq_q_dict, df_meta 

def load_hf_model_causal_lm(dtype=torch.bfloat16, device_map="cuda:1"):
    model_id = "hf/gemma-2-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, torch_dtype=dtype)
    return model, tokenizer

class RetrieveTopChunk():
    def __init__(self, tokenizer: any = None, embedding_model: any = None, retrieval_model: any = None, corpus: any = None, top_k=10):
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.retrieval_model = retrieval_model
        self.corpus = corpus
        self.top_k = top_k

    def __call__(self, query: str):
        batch_dict = self.tokenizer([query], max_length=8192, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: batch_dict[k].to(self.embedding_model.device) for k in batch_dict}
        with torch.no_grad():
            batch_dict["position_ids"] = (torch.cumsum(batch_dict["attention_mask"], dim=1) - 1).clamp(min=0)
            outputs = self.embedding_model(**batch_dict)
            dimension = 768
            embeddings = outputs.last_hidden_state[:, 0, :dimension]
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        embeddings = embeddings.cpu().numpy().astype('float32')
        list_docs = []
        list_doc_ids = []
        
        top_doc = self.retrieval_model.search([0], embeddings, top_k=self.top_k)
        
        for doc_id in top_doc[0]:
            tmp = None
            try:
                for attempt_id in [str(doc_id), str(int(doc_id)+1), str(int(doc_id)-1)]:
                    tmp = self.corpus.get(attempt_id)
                    if tmp:
                        break
            except:
                pass
            
            if tmp:
                list_docs.append(tmp.text)
                list_doc_ids.append(tmp.doc_id)
        
        if not list_docs:
            list_docs = ["No relevant documents found in the current small index."]
            list_doc_ids = ["none"]
            
        return list_docs, list_doc_ids

class RetrieveTopChunkMedcpt(RetrieveTopChunk):
    pass

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                json_object = json.loads(line)
                data.append(json_object)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {line.strip()}")
    return data

def load_dataset(name):
    """
    涓撻棬閫傞厤鐪熷疄鐨?PlantUCD 鏁版嵁闆?
    """
    import json
    data = []
    # 鐩存帴鎸囧悜鎴戜滑鍒氭墠涓嬭浇瑙ｅ帇鐨勭湡瀹炴暟鎹矾寰?
    file_path = "/hy-tmp/my_code/my_code_backup/raw_data/PlantUCD-dataset-full-main/PlantUCD_dataset_test.jsonl"
    
    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            for i, line in enumerate(f):
                item = json.loads(line.strip())
                # 鏄犲皠涓?main.py 璁よ瘑鐨勬牸寮?
                data.append({
                    "id": f"plantucd_test_{i}",       # 鐢熸垚鍞竴棰樼洰ID
                    "input": item.get("HumanLang", "") # 鍙栧嚭鑷劧璇█闇€姹?
                })
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        
    return data

class GenericDoc(NamedTuple):
    doc_id: str
    d: str
    t: str
    a: str
    m: str
    def default_text(self):
        return self.a

def load_corpus(name="dpr"):
    if name == "dpr":
        # 鉁?鍏抽敭淇敼锛氬皢 dev 鏀逛负 train锛屼笌绱㈠紩鏋勫缓鏃朵娇鐢ㄧ殑璇枡搴撲竴鑷?
        nq_dataset = ir_datasets.load("dpr-w100/natural-questions/train")
        corpus = nq_dataset.docs_store()
    else:
        DL_DOCS = GzipExtract(LocalDownload("/scratch2/f0072r1/rs_rl/pubmed.tsv.gz"))
        tmp = TsvDocs(DL_DOCS, doc_cls=GenericDoc, skip_first_line=True)
        corpus = tmp.docs_store()
    return corpus

class QAAnswerFormat(BaseModel):
    analysis: str = Field(description="Your thoughts, analysis about the question and the context. Think step-by-step")
    answer: str = Field(description="The answer for the question")
    success: Union[str, bool] = Field(description="String 'Yes' or 'No' (or boolean), indicate you can answer or not") 
    rating: int = Field(default=None, description="How confident, from 0 to 10. More evidence, more agreement, more confident")

class QAAnswerState(TypedDict):
    analysis: str
    answer: str
    success: Union[str, bool]
    rating: int

class PlanFormat(BaseModel):
    analysis: str = Field(description="Your analysis. Think step-by-step")
    step: List[str] = Field(description="different steps to follow, should be in sorted order")

class PlanState(TypedDict):
    analysis: str
    step: List[str]

class StepTaskFormat(BaseModel):
    type: str = Field(description="Type of task, one of [aggregate, question-answering]")
    task: str = Field(description="The detail task to do in this step")

class StepTaskState(TypedDict):
    type: str
    task: str

class PlanSummaryFormat(BaseModel): 
    output: str = Field(description="your output, follow the format")
    answer: str = Field(description="Final answer for the question")
    score: int = Field(description="Confident score")

class PlanSummaryState(TypedDict):
    output: str
    answer: str
    score: int

class PlanExecState(TypedDict):
    original_question: str
    plan: List[str]
    step_question: Annotated[List[StepTaskState], operator.add]
    step_output: Annotated[List[QAAnswerState], operator.add]
    step_docs_ids: Annotated[List[List[str]], operator.add]
    step_notes: Annotated[List[List[str]], operator.add]
    plan_summary: PlanSummaryState
    stop: bool = False

class RagState(TypedDict):
    question: str
    documents: List[str]
    doc_ids: List[str]
    notes: List[str]
    final_raw_answer: QAAnswerFormat

class GraphState(TypedDict):
    original_question: str
    plan: List[str]
    past_exp: Annotated[List[PlanExecState], operator.add]
    final_answer: str

def parse_args():
    parser = argparse.ArgumentParser(description="sample argument parser")
    parser.add_argument("--model", choices=['gpt4omini', 'llama3-70B', 'llama3-8B', 'llama3-70B-0', 'mix01', 'mix02', 'mix03'])
    parser.add_argument("--dataset", choices=['nq', 'hotpotqa', 'triviaqa', "2wiki", "fever", "medmcqa", "simpleqa", "plantucd_rl1127", "plantucd_test142"])
    parser.add_argument("--exp", choices=['plan_rag_extract', 'plan_rag', 'llmcot', 'rag_extract', 'llmonly'])
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=1000000)
    parser.add_argument('--gpus', nargs='+', type=int)
    return parser.parse_args()

def make_chat_openai(temperature=0.0, api_key=None, max_retries=None, timeout=None):
    """Create a ChatOpenAI client with project-wide base_url, timeout and retry defaults."""
    retry_count = int(os.getenv("OPENAI_MAX_RETRIES", str(max_retries if max_retries is not None else 2)))
    timeout_seconds = float(os.getenv("OPENAI_TIMEOUT", str(timeout if timeout is not None else 60)))
    return ChatOpenAI(
        model_name=os.getenv("MODEL_NAME"),
        temperature=temperature,
        api_key=api_key or os.getenv("API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        max_retries=retry_count,
        timeout=timeout_seconds,
    )


def invoke_structured_with_retry(chain, inputs, task_name="structured_output", attempts=None):
    """Retry structured-output calls and add a JSON-only reminder after parser failures."""
    max_attempts = int(os.getenv("STRUCTURED_OUTPUT_ATTEMPTS", str(attempts or 3)))
    base_inputs = dict(inputs)
    last_error = None
    reminder = (
        "\n\nFORMAT REMINDER: Your previous response could not be parsed as the required JSON schema. "
        "Return ONLY a raw JSON object with the required keys. Do not return markdown, prose, or 'ok'."
    )
    for attempt in range(max_attempts):
        current_inputs = dict(base_inputs)
        if attempt > 0:
            if "question" in current_inputs:
                current_inputs["question"] = str(current_inputs["question"]) + reminder
            elif "cur_step" in current_inputs:
                current_inputs["cur_step"] = str(current_inputs["cur_step"]) + reminder
            elif "memory" in current_inputs:
                current_inputs["memory"] = str(current_inputs["memory"]) + reminder
        try:
            return chain.invoke(current_inputs)
        except Exception as exc:
            last_error = exc
            print(f"[structured-retry] {task_name} attempt {attempt + 1}/{max_attempts} failed: {exc}")
    raise last_error



