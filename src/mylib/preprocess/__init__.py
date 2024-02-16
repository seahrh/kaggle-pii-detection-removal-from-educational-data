from typing import AnyStr


class Preprocessor:
    def __call__(self, s: AnyStr, **kwargs) -> str:
        raise NotImplementedError("Implement this method in subclass")
