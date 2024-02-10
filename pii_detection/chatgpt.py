import os

from openai import OpenAI

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
