# chat_qwen3_vllm_single_prompt_formatter.py
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-8B"

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
        enable_thinking=False,  # set True if you want Qwen3 "thinking" outputs
    )

sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

prompt = format_chat("Where is Harvard? Give me a short answer in two sentences.")
out = llm.generate([prompt], sampling)[0].outputs[0].text
print(out)
print("Done!")