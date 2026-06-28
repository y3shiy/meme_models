from pandas import DataFrame
from aiohttp import ClientSession

import asyncio
import json
import logging
from typing import Protocol
from dataclasses import dataclass
from pathlib import Path


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


class UnsupportedLanguage(Exception):
    pass


class DeepL:
    def __init__(self, api_key: str, is_free_api: bool = False) -> None:
        self.api_config = DeepLApiConfig(api_key, is_free_api)
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

        endpoint = self.api_config.get_translation_endpoint()

        headers = {'Authorization': f'DeepL-Auth-Key {self.api_config.api_key}',
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

    async def _get_supported_languages(self, force_fetch_new: bool = False) -> frozenset[str]:
        cache_path = Path('cache/deepl_supported_languages.json')
        if not force_fetch_new and cache_path.exists():
            cache_file_text = cache_path.read_text(encoding='utf-8')
            supported_languages_json = json.loads(cache_file_text)

            self.logger.info(f'Supported languages loaded from {cache_path}')
        else:
            supported_languages_text = await self._fetch_supported_languages_raw()

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(supported_languages_text, encoding='utf-8')

            supported_languages_json = json.loads(supported_languages_text)

            self.logger.info(f'Supported languages were fetched successfully and written to {cache_path}')

        supported_languages = []
        for obj in supported_languages_json:
            if obj['usable_as_source'] == False or obj['usable_as_target'] == False:
                continue
            supported_languages.append(obj['lang'])

        return frozenset(supported_languages)

    async def _fetch_supported_languages_raw(self) -> str:
        endpoint = self.api_config.get_supported_languages_query_endpoint()

        self.logger.info(f'Supported languages will be fetched from {endpoint}')

        headers = {'Authorization': f'DeepL-Auth-Key {self.api_config.api_key}'}

        async with ClientSession() as session:
            async with session.get(endpoint, headers=headers) as resp:
                resp_text = await resp.text()
                resp.raise_for_status()
                return resp_text


@dataclass(frozen=True)
class DeepLApiConfig:
    api_key: str = ''
    is_free_api: bool = False

    def get_translation_endpoint(self) -> str:
        if self.is_free_api:
            return 'https://api-free.deepl.com/v2/translate'
        else:
            return 'https://api.deepl.com/v2/translate'

    def get_supported_languages_query_endpoint(self) -> str:
        if self.is_free_api:
            return 'https://api-free.deepl.com/v3/languages?resource=translate_text'
        else:
            return 'https://api.deepl.com/v3/languages?resource=translate_text'
