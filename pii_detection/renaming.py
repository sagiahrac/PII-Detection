from faker import Faker

fake = Faker()


def rename_token(token: str):
    if token.startswith("B"):
        new_token = fake.unique.first_name()
    elif token.startswith("I"):
        new_token = fake.unique.last_name()
    return new_token


def convert_to_text(tokens, whitespaces):
    text = []
    for token, whitespace in zip(tokens, whitespaces):
        text.append(token)
        if whitespace:
            text.append(" ")
    return "".join(text)


def modify_names_in_essay(essay):
    renamed = {}
    new_tokens = []
    for token, label in zip(essay["tokens"], essay["labels"]):
        if "NAME_STUDENT" in label:
            if token not in renamed:
                renamed[token] = rename_token(label)
            new_tokens.append(renamed[token])
        else:
            new_tokens.append(token)

    new_full_text = convert_to_text(new_tokens, essay["trailing_whitespace"])
    essay["full_text"] = new_full_text
    essay["tokens"] = new_tokens
    return essay


if __name__ == "__main__":
    import json

    with open("data/train_shard.json", "r") as f:
        train_data = json.load(f)

    modified_train_data = []
    for essay in train_data:
        essay = modify_names_in_essay(essay)
        modified_train_data.append(essay)

    modified_train_data = sorted(modified_train_data, key=lambda x: x["document"])
    with open("data/train_shard_renamed.json", "w") as f:
        json.dump(modified_train_data, f, indent=4)
