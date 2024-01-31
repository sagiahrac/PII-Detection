import json
import os
from time import sleep

from openai import OpenAI
from tqdm import tqdm


CHAT_MODEL = "gpt-3.5-turbo"
if "OPENAI_API_KEY" not in os.environ:
    raise Exception("OPENAI_API_KEY environment variable not set.")
client = OpenAI()



def generate_essay(essay_text, debug_mode=False):
    prompt = f"rewrite the following essay:\n\n{essay_text}"

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a student"},
            {"role": "user", "content": prompt},
        ],
    )

    if debug_mode:
        print(f"[DEBUG] OUTPUT: {completion.choices[0].message.content}")

    gpt_output = completion.choices[0].message.content
    return gpt_output


if __name__ == "__main__":
    with open("data/train_shard_renamed.json", "r") as f:
        train_data = json.load(f)
    train_data = sorted(train_data, key=lambda x: x["document"])

    new_essays = []
    for essay in tqdm(train_data):
        new_text = generate_essay(essay["full_text"], debug_mode=True)
        essay["new_text"] = new_text
        new_essays.append(essay)

        with open("data/train_shard_generated.json", "w") as f:
            json.dump(new_essays, f, indent=4)
