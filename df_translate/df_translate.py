from aiohttp import ClientTimeout
from aiohttp_retry import ListRetry, RetryClient
from pandas import DataFrame

import asyncio
import json
import logging
from typing import Any, Callable, Protocol
from dataclasses import dataclass
from pathlib import Path


DEEPL_REQUEST_TIMEOUT_SECONDS = 10
DEEPL_RETRY_ATTEMPTS = 3
DEEPL_RETRY_PERIOD_SECONDS = 1
DEEPL_RETRY_STATUSES = {429}


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
    def __init__(self, api_key: str,
                 is_free_api: bool = False,
                 _client_session_factory: Callable[[], Any] | None = None,
                 _supported_languages: frozenset[str] | None = None) -> None:
        if _client_session_factory is None:
            self._client_session_factory = lambda: RetryClient(
                timeout=ClientTimeout(total=DEEPL_REQUEST_TIMEOUT_SECONDS),
                retry_options=ListRetry([DEEPL_RETRY_PERIOD_SECONDS] * DEEPL_RETRY_ATTEMPTS,
                                        statuses=DEEPL_RETRY_STATUSES,
                                        retry_all_server_errors=True)
            )
        else:
            self._client_session_factory = _client_session_factory

        self._api_config = DeepLApiConfig(api_key, is_free_api)
        self._logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')

        if _supported_languages is None:
            try:
                self._supported_languages = self._get_supported_languages()
            except DeepLRequestError as error:
                match error.status:
                    case 401 as status:
                        raise DeepLRequestError(status, 'DeepL API key is not valid or the name is spelt incorrectly')
                    case 403 as status:
                        raise DeepLRequestError(status, 'DeepL API key does not have permission to perform translation')
                    case _:
                        raise
        else:
            self._supported_languages = _supported_languages

    def translate_text(self, text: str,
                       source_lang: str, target_lang: str,
                       provider: str | None = None, **kwargs) -> str:
        try:
            coro = self._send_translation_request([text], source_lang, target_lang)
            result = asyncio.run(coro)
            assert len(result) == 1
            return result[0]
        except DeepLRequestError as error:
            match error.status:
                case 401 as status:
                    raise DeepLRequestError(status, 'DeepL API key is not valid or the name is spelt incorrectly')
                case 403 as status:
                    raise DeepLRequestError(status, 'DeepL API key does not have permission to perform translation')
                case _:
                    raise

    def translate_dataframe(self, df: DataFrame,
                            source_lang: str, target_langs: list[str],
                            provider: str | None = None,
                            raise_on_failed_rows: bool = False,
                            missing_value: str = '', **kwargs) -> DataFrame:
        raise NotImplementedError()

    def is_supported_language(self, lang: str) -> bool:
        lang = lang.upper()
        return (lang == 'AUTO') or (lang in self._supported_languages)

    def _raise_for_status(self, status: int, resp_text: str) -> None:
        if 200 <= status < 300:
            return
        if status == 400:
            raise DeepLRequestError(status, f'Bad Request: {resp_text}')
        if status in (401, 403):
            raise DeepLRequestError(status, f'Auth Failed: {resp_text}')
        if status == 429:
            raise DeepLRequestError(status, f'Rate Limited: {resp_text}')
        if 500 <= status < 600:
            raise DeepLRequestError(status, f'Server Error {status}: {resp_text}')

        raise DeepLRequestError(status, f'HTTP {status}: {resp_text}')

    async def _send_translation_request(self, texts: list[str],
                                        source_lang: str,
                                        target_lang: str) -> list[str]:
        if not self.is_supported_language(source_lang):
            raise UnsupportedLanguage(f'Source language is not supported or the name is spelt incorrectly: "{source_lang}"')
        if not self.is_supported_language(target_lang):
            raise UnsupportedLanguage(f'Target language is not supported or the name is spelt incorrectly: "{target_lang}"')

        source_lang, target_lang = source_lang.upper(), target_lang.upper()

        endpoint = self._api_config.get_translation_endpoint()

        headers = {'Authorization': f'DeepL-Auth-Key {self._api_config.api_key}',
                   'Content-Type': 'application/json'}

        if source_lang == 'AUTO':
            payload = {'text': texts,
                       'target_lang': target_lang,}
        else:
            payload = {'text': texts,
                       'source_lang': source_lang,
                       'target_lang': target_lang,}

        async with self._client_session_factory() as session:
            async with session.post(endpoint, headers=headers, json=payload) as resp:
                resp_text = await resp.text()
                self._raise_for_status(resp.status, resp_text)
                resp_json = json.loads(resp_text)
                result = [obj['text'] for obj in resp_json['translations']]
                return result

    def _get_supported_languages(self, force_fetch_new: bool = False) -> frozenset[str]:
        cache_path = Path('cache/deepl_supported_languages.json')
        if not force_fetch_new and cache_path.exists():
            cache_file_text = cache_path.read_text(encoding='utf-8')
            supported_languages_json = json.loads(cache_file_text)

            self._logger.info(f'Supported languages loaded from {cache_path}')
        else:
            supported_languages_text = asyncio.run(self._fetch_supported_languages_raw())

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(supported_languages_text, encoding='utf-8')

            supported_languages_json = json.loads(supported_languages_text)

            self._logger.info(f'Supported languages were fetched successfully and written to {cache_path}')

        supported_languages = []
        for obj in supported_languages_json:
            if obj['usable_as_source'] == False or obj['usable_as_target'] == False:
                continue
            supported_languages.append(obj['lang'].upper())

        return frozenset(supported_languages)

    async def _fetch_supported_languages_raw(self) -> str:
        endpoint = self._api_config.get_supported_languages_query_endpoint()

        self._logger.info(f'Supported languages will be fetched from {endpoint}')

        headers = {'Authorization': f'DeepL-Auth-Key {self._api_config.api_key}'}

        async with self._client_session_factory() as session:
            async with session.get(endpoint, headers=headers) as resp:
                resp_text = await resp.text()
                self._raise_for_status(resp.status, resp_text)
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


class DeepLRequestError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(message)
