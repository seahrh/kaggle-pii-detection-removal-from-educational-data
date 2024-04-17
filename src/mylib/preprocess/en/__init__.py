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
        # do not remove accented chars bec they can appear in names and addresses
        # res: str = snlp.to_ascii(s)
        res: str = snlp.to_str(s)
        res = snlp.collapse_whitespace(res)
        return res
