import pandas as pd
import json


def read_tokens_df(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        train_data = json.load(f)

    df = pd.DataFrame.from_records(
        train_data, columns=["document", "tokens", "trailing_whitespace", "labels"]
    )
    token_loc_in_doc = df["tokens"].apply(lambda x: list(range(len(x))))
    df.insert(1, "loc_in_doc", token_loc_in_doc)
    df = df.explode(["loc_in_doc", "tokens", "trailing_whitespace", "labels"])
    df = df.sort_values(["document", "loc_in_doc"]).reset_index(drop=True)
    return df
