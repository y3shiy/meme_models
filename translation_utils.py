from typing import Protocol

from pandas import DataFrame


class Translator(Protocol):
    def translate_text(self, text: str,
                       source_lang: str, target_lang: str,
                       provider: str | None = None, **kwargs) -> str:
        ...

    def translate_dataframe(self, df: DataFrame,
                            source_lang: str, target_langs: list[str],
                            provider: str | None = None,
                            raise_on_failed_rows: bool = False,
                            missing_value: str = '', **kwargs) -> DataFrame:
        ...


class DeepL:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def translate_text(self, text: str,
                       source_lang: str, target_lang: str,
                       provider: str | None = None, **kwargs) -> str:
        raise NotImplementedError()

    def translate_dataframe(self, df: DataFrame,
                            source_lang: str, target_langs: list[str],
                            provider: str | None = None,
                            raise_on_failed_rows: bool = False,
                            missing_value: str = '', **kwargs) -> DataFrame:
        raise NotImplementedError()
