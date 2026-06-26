from pandas import DataFrame
from aiohttp import ClientSession

import asyncio
import json
import logging
from typing import Protocol


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
    def __init__(self, api_key: str, is_free_api: bool = False) -> None:
        self.api_key = api_key
        self.is_free_api = is_free_api
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')

    def translate_text(self, text: str,
                       source_lang: str, target_lang: str,
                       provider: str | None = None, **kwargs) -> str:
        coro = self._send_translation_request([text], source_lang, target_lang)
        result = asyncio.run(coro)
        assert len(result) == 1
        return result[0]

    def translate_dataframe(self, df: DataFrame,
                            source_lang: str, target_langs: list[str],
                            provider: str | None = None,
                            raise_on_failed_rows: bool = False,
                            missing_value: str = '', **kwargs) -> DataFrame:
        raise NotImplementedError()

    async def _send_translation_request(self, texts: list[str],
                                        source_lang: str,
                                        target_lang: str) -> list[str]:
        source_lang, target_lang = source_lang.upper(), target_lang.upper()

        if self.is_free_api:
            endpoint = 'https://api-free.deepl.com/v2/translate'
        else:
            endpoint = 'https://api.deepl.com/v2/translate'

        headers = {'Authorization': f'DeepL-Auth-Key {self.api_key}',
                   'Content-Type': 'application/json'}

        if source_lang == 'AUTO':
            payload = {'text': texts,
                       'target_lang': target_lang,}
        else:
            payload = {'text': texts,
                       'source_lang': source_lang,
                       'target_lang': target_lang,}

        async with ClientSession() as session:
            async with session.post(endpoint, headers=headers, json=payload) as resp:
                resp_text = await resp.text()
                match resp.status:
                    case status if 200 <= status < 300:
                        resp_json = json.loads(resp_text)
                        result = [obj['text'] for obj in resp_json['translations']]
                        return result
                    case 400:
                        raise ValueError(f'Bad Request: {resp_text}')
                    case 401 | 403:
                        raise PermissionError(f'Auth Failed: {resp_text}')
                    case 429:
                        raise RuntimeError(f'Rate Limited: {resp_text}')
                    case status if 500 <= status < 600:
                        raise RuntimeError(f'Server Error {status}: {resp_text}')
                    case status:
                        raise RuntimeError(f'HTTP {status}: {resp_text}')
