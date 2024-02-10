from typing import Dict, List

from spacy.lang.en import English

from pii_detection.errors import TokenWithMultipleLabelsError

EN_TOKENIZER = English().tokenizer


def tokenize_with_spacy(text: str, tokenizer=EN_TOKENIZER) -> Dict[str, List[str]]:
    tokenized_text = tokenizer(text)
    tokens = [token.text for token in tokenized_text]
    trailing_whitespace = [bool(token.whitespace_) for token in tokenized_text]
    return {"tokens": tokens, "trailing_whitespace": trailing_whitespace}


def convert_tokens_to_text(tokens, whitespaces):
    text = []
    for token, whitespace in zip(tokens, whitespaces):
        text.append(token)
        if whitespace:
            text.append(" ")
    return "".join(text)


def filter_pii_tokens(
    tokens: List[str], labels: List[str], document_id: int = None
) -> Dict[str, str]:
    pii_tokens = {}
    for token, label in zip(tokens, labels):
        if label != "O":
            token = token.lower()
            label = label[2:]

            if token not in pii_tokens:
                pii_tokens[token] = label
            else:
                if pii_tokens[token] != label:
                    raise TokenWithMultipleLabelsError(
                        token, [pii_tokens[token], label], document_id
                    )
    return pii_tokens


def label_tokens(tokens, pii_tokens):
    labels = []
    for token in tokens:
        label = pii_tokens.get(token.lower(), "O")
        labels.append(label)
    return labels
