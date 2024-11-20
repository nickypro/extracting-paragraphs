import os
import json
from groq import Groq
from tqdm import tqdm

# Initialize Groq client
client = Groq(
    api_key="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",
)

# Load prompts from file
folder = "../data/llama9b"
version = "V2.1"
num_copies = 10
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
                model="gemma2-9b-it",
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