from typing import AnyStr

import scml
import scml.nlp as snlp

from mylib.preprocess import Preprocessor

__all__ = ["BasicPreprocessor"]

log = scml.get_logger(__name__)


class BasicPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()

    def __call__(self, s: AnyStr, **kwargs) -> str:
        res: str = snlp.to_ascii(s)
        res = snlp.collapse_whitespace(res)
        return res
