from aiohttp import ClientTimeout
from aiohttp_retry import ExponentialRetry, RetryClient
from beartype import beartype
from pandas import DataFrame

import asyncio
import itertools
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class Translator(Protocol):
    def translate_text(self, text: str,
                       source_lang: str, target_lang: str,
                       provider: str | None = None, **kwargs: Any) -> str:
        ...

    def translate_dataframe(self, df: DataFrame,
                            source_lang: str, target_langs: list[str],
                            provider: str | None = None,
                            raise_on_failed_rows: bool = False,
                            missing_value: str = '', **kwargs: Any) -> DataFrame:
        ...


class UnsupportedLanguage(Exception):
    pass


class DeepL:
    @beartype
    def __init__(self, api_key: str,
                 is_free_api: bool = False,
                 _client_session_factory: Callable[[], Any] | None = None,
                 _supported_languages: frozenset[str] | None = None) -> None:
        self._requests_config = DeepLRequestsConfig()

        if _client_session_factory is None:
            self._client_session_factory = lambda: RetryClient(
                timeout=ClientTimeout(total=self._requests_config.timeout_ms / 1000),
                retry_options=ExponentialRetry(
                    attempts=self._requests_config.max_retries + 1,
                    start_timeout=self._requests_config.retry_interval_ms / 1000,
                    factor=1.0,
                    statuses=self._requests_config.retry_statuses,
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

    @beartype
    def use_batch_size(self, batch_size: int) -> 'DeepL':
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        self._requests_config.batch_size = batch_size
        return self

    @beartype
    def use_texts_per_request(self, texts_per_request: int) -> 'DeepL':
        if texts_per_request <= 0:
            raise ValueError('batch_size must be positive')
        if texts_per_request > 50:
            raise ValueError('texts_per_request is capped at 50 according to DeepL API docs')
        self._requests_config.texts_per_request = texts_per_request
        return self

    @beartype
    def use_max_retries(self, max_retries: int) -> 'DeepL':
        if max_retries < 0:
            raise ValueError('max_retries cannot be negative')
        self._requests_config.max_retries = max_retries
        return self

    @beartype
    def use_retry_interval_ms(self, retry_interval_ms: int) -> 'DeepL':
        if retry_interval_ms < 0:
            raise ValueError('retry_interval_ms cannot be negative')
        self._requests_config.retry_interval_ms = retry_interval_ms
        return self

    @beartype
    def use_timeout_ms(self, timeout_ms: int) -> 'DeepL':
        if timeout_ms <= 0:
            raise ValueError('timeout_ms must be positive')
        self._requests_config.timeout_ms = timeout_ms
        return self

    @beartype
    def translate_text(self, text: str,
                       source_lang: str, target_lang: str,
                       provider: str | None = None, **kwargs: Any) -> str:
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

    @beartype
    def translate_dataframe(self, df: DataFrame,
                            source_lang: str, target_langs: list[str],
                            provider: str | None = None,
                            raise_on_failed_rows: bool = False,
                            missing_value: str = '', **kwargs: Any) -> DataFrame:
        if df.shape[0] <= 0 or len(target_langs) <= 0:
            return DataFrame()

        df_rows: int = df.shape[0]
        result = [[missing_value] * len(target_langs) for _ in range(df_rows)]
        assert result != []

        batch_size = self._requests_config.batch_size
        for i, target_lang in enumerate(target_langs):
            total_translated = 0
            for batch in itertools.batched(df.iloc[:, 0], batch_size):
                batch_result = asyncio.run(self._translate_dataframe_range(
                    df, source_lang, target_lang,
                    raise_on_failed_rows=raise_on_failed_rows,
                    missing_value=missing_value,
                    from_index=total_translated,
                    count=len(batch)
                ))
                for j, translation in enumerate(batch_result):
                    result[total_translated + j][i] = translation
                total_translated += len(batch)

        columns = [f'{df.columns[0]}_{lang}' for lang in target_langs]
        return DataFrame(result, columns=columns)


    @beartype
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

    async def _translate_dataframe_range(self, df: DataFrame,
                                         source_lang: str, target_lang: str,
                                         raise_on_failed_rows: bool,
                                         missing_value: str,
                                         from_index: int, count: int) -> list[str]:
        assert from_index >= 0
        assert count > 0

        if from_index + count > df.shape[0]:
            raise ValueError()

        coroutines = [] # list of coroutines
        batch_lengths = []
        texts_per_request = self._requests_config.texts_per_request
        for batch in itertools.batched(df.iloc[from_index:from_index + count, 0],
                                       texts_per_request):
            texts = list(batch)
            batch_lengths.append(len(texts))
            coro = self._send_translation_request(texts, source_lang, target_lang)
            coroutines.append(coro)
        
        translated: list[list[str]] = await asyncio.gather(
                *coroutines, return_exceptions=not raise_on_failed_rows
        )

        result: list[str] = []
        for i, translations in enumerate(translated):
            if isinstance(translations, Exception):
                result.extend([missing_value] * batch_lengths[i])
            else:
                result.extend(translations)

        return result


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


@dataclass
class DeepLRequestsConfig:
    batch_size: int = 100
    texts_per_request: int = 50
    max_retries: int = 3
    retry_interval_ms: int = 1000
    retry_statuses: frozenset[int] = frozenset({429})
    timeout_ms: int = 10000


class DeepLRequestError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(message)
