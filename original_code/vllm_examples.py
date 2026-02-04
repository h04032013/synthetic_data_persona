#https://huggingface.co/datasets/proj-persona/PersonaHub 
#https://github.com/tencent-ailab/persona-hub/tree/main/code
#https://github.com/allenai/open-instruct/blob/main/scripts/persona_driven_data_gen/persona_driven_generate_math_code.py

import json
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams
from prompt_templates import math_template, math_solution_template

MODEL = "Qwen/Qwen3-8B"
#MODEL = "allenai/Olmo-3-7B-Think"

QUESTIONS_PATH = Path(
    "/n/holylabs/LABS/dam_lab/Users/hdiaz/synthetic_personas/original_code/examples/500_math_examples_qwen38b.jsonl"
)
SOLUTIONS_PATH = Path(
    "/n/holylabs/LABS/dam_lab/Users/hdiaz/synthetic_personas/original_code/examples/500_math_solutions_qwen38b.jsonl"
)

SYSTEM_PROMPT = "You are a helpful assistant."

def format_chat(tokenizer, user_prompt, system=SYSTEM_PROMPT) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

def load_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def generate_questions(llm, tokenizer, out_path, n):
    ds = load_dataset("proj-persona/PersonaHub", "persona", split="train")
    #line below, make sure to either change this or n param in line 104
    #ds = ds.select(range(500))
    personas = [ds[i]["persona"].strip() for i in range(n)]

    prompts = [format_chat(tokenizer, math_template.format(persona=p)) for p in personas]

    sampling = SamplingParams(
        temperature=0.7,
        max_tokens=4096,
        seed=0,
    )
    outputs = llm.generate(prompts, sampling)

    with out_path.open("w", encoding="utf-8") as f:
        for persona, out in zip(personas, outputs):
            text = out.outputs[0].text.strip()
            record = {
                "persona": persona,
                "problem": text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def generate_solutions(llm, tokenizer, in_path, out_path):
    records = load_jsonl(in_path)

    prompts = []
    for r in records:
        problem = (r.get("problem") or "").strip()
        user_prompt = math_solution_template.format(persona=problem)
        prompts.append(format_chat(tokenizer, user_prompt))

    sampling = SamplingParams(
        temperature=0.7,
        max_tokens=4096,
        seed=0,
    )

    outputs = llm.generate(prompts, sampling)

    with out_path.open("w", encoding="utf-8") as f:
        for r, out in zip(records, outputs):
            solution_text = out.outputs[0].text.strip()
            out_record = {
                "persona": r.get("persona", ""),
                "problem": r.get("problem", ""),
                "solution": solution_text,
            }
            f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

def main():
    llm = LLM(model=MODEL)
    tokenizer = llm.get_tokenizer()

    generate_questions(llm, tokenizer, QUESTIONS_PATH , n=500)
    generate_solutions(llm, tokenizer, QUESTIONS_PATH, SOLUTIONS_PATH)

if __name__ == "__main__":
    main()
