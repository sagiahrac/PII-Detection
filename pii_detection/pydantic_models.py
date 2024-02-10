from typing import Dict, List

from pydantic import BaseModel, NonNegativeInt, model_validator

from pii_detection.chatgpt import generate_essay as generate_text_with_chatgpt
from pii_detection.gen_data_utils import (
    convert_tokens_to_text,
    filter_pii_tokens,
    label_tokens,
    tokenize_with_spacy,
)


class Essay(BaseModel):
    document: NonNegativeInt
    full_text: str
    tokens: List[str]
    trailing_whitespace: List[bool]
    labels: List[str]
    generated: bool = False

    @model_validator(mode="after")
    def validate_arrays_length(self) -> "Essay":
        n_tokens = len(self.tokens)
        if len(self.trailing_whitespace) != n_tokens:
            raise ValueError("trailing_whitespace must have the same length as tokens")

        if len(self.labels) != n_tokens:
            raise ValueError("labels must have the same length as tokens")
        return self

    @model_validator(mode="after")
    def validate_tokens_with_text(self) -> "Essay":
        converted_text = convert_tokens_to_text(self.tokens, self.trailing_whitespace)
        if converted_text.strip() != self.full_text.strip():
            raise ValueError("tokens and trailing_whitespace do not match full_text")
        return self

    @property
    def pii_tokens(self) -> Dict[str, str]:
        return filter_pii_tokens(self.tokens, self.labels, self.document)

    def rewrite_essay(self, debug=False) -> "Essay":
        new_essay = {}
        new_essay["document"] = self.document
        new_essay["full_text"] = generate_text_with_chatgpt(
            self.full_text, debug_mode=debug
        )

        tokens_and_whitespaces = tokenize_with_spacy(new_essay["full_text"])
        new_essay.update(tokens_and_whitespaces)

        new_essay["labels"] = label_tokens(new_essay["tokens"], self.pii_tokens)
        new_essay["generated"] = True

        return Essay(**new_essay)
