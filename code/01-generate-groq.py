# %%
import os
import json
from groq import Groq
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

def get_completion_transformers(prompt):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from taker import Model
    if not hasattr(get_completion_transformers, 'm'):
        get_completion_transformers.m = Model(model_name, dtype="hqq4")
    inpt, output = get_completion_transformers.m.generate(prompt)
    return output

# Initialize Groq client
client = Groq(
    api_key="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",
)

def get_completion_openai(prompt):
    # Assume openai>=1.0.0
    from openai import OpenAI

    # Create an OpenAI client with your deepinfra token and endpoint
    openai = OpenAI(
        api_key=os.getenv("DEEPINFRA_API_KEY"),
        base_url="https://api.deepinfra.com/v1/openai",
    )

    chat_completion = openai.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="Qwen/Qwen2.5-Coder-7B"
        temperature=0.3,
        # max_tokens=cfg.max_new_tokens,
    )

    return chat_completion.choices[0].message.content

# Load prompts from file
# model="llama-3.2-3b-preview"
# model = "llama-3.2-11b-text-preview"
# model = "llama-3.1-8b-instant"
model = "ministral-3b-8192"
folder = "../data/llama8b"
version = "V2"

num_copies = 5
SKIP_TEXTS = 0
with open(f"{folder}/{version}_prompts.txt", "r") as f:
    prompts = f.read().splitlines()

# Function to format input for Gemma 2 9B
def format_input(prompt):
    return f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>assistant\n"

# Generate responses and save to JSONL
with open(f"{folder}/{version}_orig_generation.jsonl", "a") as outfile:
    for prompt in tqdm(prompts[SKIP_TEXTS:], desc="Generating responses"):
        for i in tqdm(range(num_copies)):
            formatted_input = format_input(prompt)

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                # model="gemma2-9b-it",
                # model="llama-3.2-11b-text-preview",
                model=model,
                temperature=0.3,
            )

            response = chat_completion.choices[0].message.content
            formatted_full_text = formatted_input + response

            result = {
                "text": prompt,
                "output": response,
                "formatted_input": formatted_input,
                "formatted_full_text": formatted_full_text
            }

            json.dump(result, outfile)
            outfile.write('\n')

print("Generation complete.")
# %%
