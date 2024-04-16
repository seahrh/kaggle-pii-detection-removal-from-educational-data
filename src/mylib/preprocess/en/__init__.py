from typing import AnyStr, List, Tuple

import scml
import scml.nlp as snlp
from spacy.lang.en import English

from mylib.preprocess import Preprocessor

__all__ = ["BasicPreprocessor"]

log = scml.get_logger(__name__)


class BasicPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()
        self.tokenizer = English().tokenizer

    def __call__(self, s: AnyStr, **kwargs) -> str:
        # do not remove accented chars bec they can appear in names and addresses
        # res: str = snlp.to_ascii(s)
        res: str = snlp.to_str(s)
        res = snlp.collapse_whitespace(res)
        return res

    import re

    def tokenize(self, text: str) -> Tuple[List[str], List[bool]]:
        tokenized = self.tokenizer(text)
        tokens = [token.text for token in tokenized]
        trailing_whitespace = [bool(token.whitespace_) for token in tokenized]
        return tokens, trailing_whitespace
