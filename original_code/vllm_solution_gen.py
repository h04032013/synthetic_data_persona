
import json
from pathlib import Path

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-8B"

INPUT_PATH = Path(
    "/n/holylabs/LABS/dam_lab/Users/hdiaz/synthetic_personas/original_code/examples/500_math_examples.jsonl"
)
OUTPUT_PATH = Path(
    "/n/holylabs/LABS/dam_lab/Users/hdiaz/synthetic_personas/original_code/examples/500_math_solutions.jsonl"
)

SYSTEM_PROMPT = "You are a helpful assistant."

ANSWER_TEMPLATE = """Provide solution to the given math problem.
Problem: {generated_math_problem}
Note: Provide your solution step-by-step, and end your solution in a new line in the following format:
Final Answer: The final answer is $final_answer$. I hope it is correct.
"""

def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def format_chat(tokenizer, user_prompt: str, system: str = SYSTEM_PROMPT) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

def main():
    records = load_jsonl(INPUT_PATH)

    llm = LLM(model=MODEL)
    tokenizer = llm.get_tokenizer()

    prompts = []
    for r in records:
        problem = (r.get("problem") or "").strip()
        user_prompt = ANSWER_TEMPLATE.format(generated_math_problem=problem)
        prompts.append(format_chat(tokenizer, user_prompt))

    sampling = SamplingParams(
        temperature=0.7,   # lower temp for more consistent math formatting
        top_p=0.95,
        max_tokens=4096,
        seed=0
    )

    outputs = llm.generate(prompts, sampling)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for r, out in zip(records, outputs):
            solution_text = out.outputs[0].text.strip()
            out_record = {
                "persona": r.get("persona", ""),
                "problem": r.get("problem", ""),
                "solution": solution_text,
            }
            f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
