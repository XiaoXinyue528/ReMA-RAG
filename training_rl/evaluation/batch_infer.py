from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def extract_plantuml(text: str) -> str:
    value = (text or "").strip()
    lower = value.lower()
    start = lower.find("@startuml")
    end = lower.find("@enduml", start + 1) if start >= 0 else -1
    if start >= 0 and end >= 0:
        return value[start : end + len("@enduml")].strip()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if args.limit > 0:
        rows = rows[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization,
        device_map="auto",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            requirement = row["HumanLang"]
            prompt = "Combine all step snippets into final PlantUML for requirement:\n" + requirement
            messages = [{"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=768,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            prediction = extract_plantuml(tokenizer.decode(new_tokens, skip_special_tokens=True))
            item = {
                "id": f"plantucd_test_{index - 1}",
                "requirement": requirement,
                "gold_plantuml": row["PlantUML"],
                "prediction": prediction,
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(rows)}] {item['id']}", flush=True)


if __name__ == "__main__":
    main()
