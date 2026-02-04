#https://huggingface.co/datasets/proj-persona/PersonaHub 
#https://github.com/tencent-ailab/persona-hub/tree/main/code
#https://github.com/allenai/open-instruct/blob/main/scripts/persona_driven_data_gen/persona_driven_generate_math_code.py

import json
from vllm import LLM, SamplingParams
from datasets import load_dataset
from prompt_templates import math_template

MODEL = "Qwen/Qwen3-8B"
OUTPUT_PATH = (
    "/n/holylabs/LABS/dam_lab/Users/hdiaz/synthetic_personas/original_code/examples/500_math_examples.jsonl"
)

llm = LLM(model=MODEL)
tokenizer = llm.get_tokenizer()

def format_chat(prompt: str, system: str = "You are a helpful assistant.") -> str:
    """Convert a single user prompt into a chat-templated prompt string."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

sampling = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=4196,
    seed=0
)

ds = load_dataset("proj-persona/PersonaHub", 'persona', split="train")

personas = [ds[i]["persona"].strip() for i in range(500)]

prompts = [
    format_chat(math_template.format(persona=p))
    for p in personas
]

outputs = llm.generate(prompts, sampling)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for persona, out in zip(personas, outputs):
        text = out.outputs[0].text.strip()

        record = {
            "persona": persona,
            "problem": text,
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")