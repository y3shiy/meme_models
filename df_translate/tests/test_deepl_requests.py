import json

import pytest
import pandas as pd
from pytest import MonkeyPatch
from pandas.testing import assert_frame_equal

import df_translate as df_translate_module
from df_translate import DeepL, DeepLRequestError


class FakeResponse:
    def __init__(self, status: int, data: dict) -> None:
        self.status = status
        self._data = data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    async def text(self) -> str:
        return json.dumps(self._data)

    def raise_for_status(self) -> None:
        return None


class FakeClientSession:
    def __init__(self) -> None:
        self.post_args = None
        self.post_kwargs = None
        self.get_args = None
        self.get_kwargs = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    def post(self, *args, **kwargs) -> FakeResponse:
        self.post_args = args
        self.post_kwargs = kwargs

        return FakeResponse(200, {'translations': [{'text': 'Bonjour le monde!'}]})

    def get(self, *args, **kwargs) -> FakeResponse:
        self.get_args = args
        self.get_kwargs = kwargs
        return FakeResponse(200, [{'lang': 'en',
                                   'usable_as_source': True,
                                   'usable_as_target': True},
                                  {'lang': 'fr',
                                   'usable_as_source': True,
                                   'usable_as_target': True}])


@pytest.fixture
def fake_session() -> FakeClientSession:
    return FakeClientSession()


@pytest.mark.parametrize('is_free_api,expected_endpoint',
                         [(False, 'https://api.deepl.com/v2/translate'),
                          (True, 'https://api-free.deepl.com/v2/translate')])
def test_translation_valid_endpoint(fake_session: FakeClientSession,
                                    is_free_api: bool,
                                    expected_endpoint: str):
    deepl = DeepL('DUMMY_API_KEY', is_free_api,
                  _client_session_factory=lambda: fake_session,
                  _supported_languages=frozenset(['EN', 'FR']))

    text = deepl.translate_text('Hello, world!', 'en', 'fr')

    assert text == 'Bonjour le monde!'
    assert fake_session.post_args == (expected_endpoint,)
    assert fake_session.post_kwargs == {
        'headers': {'Authorization': 'DeepL-Auth-Key DUMMY_API_KEY',
                    'Content-Type': 'application/json'},
        'json': {'text': ['Hello, world!'],
                 'source_lang': 'EN',
                 'target_lang': 'FR'}
    }


def test_translation_auto_source_lang(fake_session: FakeClientSession):
    deepl = DeepL('DUMMY_API_KEY', True,
                  _client_session_factory=lambda: fake_session,
                  _supported_languages=frozenset(['EN', 'FR']))

    text = deepl.translate_text('Hello, world!', 'auto', 'fr')

    assert text == 'Bonjour le monde!'
    assert fake_session.post_args == ('https://api-free.deepl.com/v2/translate',)
    assert fake_session.post_kwargs == {
        'headers': {'Authorization': 'DeepL-Auth-Key DUMMY_API_KEY',
                    'Content-Type': 'application/json'},
        'json': {'text': ['Hello, world!'],
                 'target_lang': 'FR'}
    }


@pytest.mark.parametrize(
    'is_free_api,expected_endpoint',
    [(False, 'https://api.deepl.com/v3/languages?resource=translate_text'),
     (True, 'https://api-free.deepl.com/v3/languages?resource=translate_text')]
)
def test_supported_languages_fetches_new(fake_session: FakeClientSession,
                                         monkeypatch: MonkeyPatch,
                                         tmp_path,
                                         is_free_api: bool,
                                         expected_endpoint: str):
    monkeypatch.chdir(tmp_path)

    deepl = DeepL('DUMMY_API_KEY', is_free_api,
                  _client_session_factory=lambda: fake_session)

    assert deepl.is_supported_language('en')
    assert deepl.is_supported_language('fr')
    assert fake_session.get_args == (expected_endpoint,)
    assert fake_session.get_kwargs == {
        'headers': {'Authorization': 'DeepL-Auth-Key DUMMY_API_KEY'}
    }

    cache_path = tmp_path / 'cache' / 'deepl_supported_languages.json'

    assert cache_path.exists()
    assert json.loads(cache_path.read_text(encoding='utf-8')) == [
        {'lang': 'en',
         'usable_as_source': True,
         'usable_as_target': True},
        {'lang': 'fr',
         'usable_as_source': True,
         'usable_as_target': True}
    ]


def test_default_session_factory_uses_retry_policy(monkeypatch: MonkeyPatch):
    captured_timeout = {}
    captured_retry_options = {}
    captured_retry_client = {}

    class FakeClientTimeout:
        def __init__(self, **kwargs) -> None:
            captured_timeout.update(kwargs)

    class FakeExponentialRetry:
        def __init__(self, **kwargs) -> None:
            captured_retry_options.update(kwargs)

    class FakeRetryClient:
        def __init__(self, **kwargs) -> None:
            captured_retry_client.update(kwargs)

    monkeypatch.setattr(df_translate_module, 'ClientTimeout', FakeClientTimeout)
    monkeypatch.setattr(df_translate_module, 'ExponentialRetry', FakeExponentialRetry)
    monkeypatch.setattr(df_translate_module, 'RetryClient', FakeRetryClient)

    deepl = DeepL('DUMMY_API_KEY',
                  _supported_languages=frozenset(['EN', 'FR']))
    configured_deepl = deepl \
        .use_batch_size(50) \
        .use_max_retries(3) \
        .use_retry_interval_ms(300) \
        .use_timeout_ms(2000)

    session = deepl._client_session_factory()

    assert configured_deepl is deepl
    assert deepl._requests_config.batch_size == 50
    assert deepl._requests_config.max_retries == 3
    assert deepl._requests_config.retry_interval_ms == 300
    assert deepl._requests_config.timeout_ms == 2000
    assert isinstance(session, FakeRetryClient)
    assert captured_timeout == {'total': 2.0}
    assert isinstance(captured_retry_client['timeout'], FakeClientTimeout)
    assert isinstance(captured_retry_client['retry_options'], FakeExponentialRetry)
    expected_retry_options = {'attempts': 4,
                              'start_timeout': 0.3,
                              'factor': 1.0,
                              'statuses': frozenset({429}),
                              'retry_all_server_errors': True}
    assert captured_retry_options == expected_retry_options


def test_translate_dataframe_single_column(monkeypatch: MonkeyPatch):
    calls = []

    async def fake_send_translation_request(texts: list[str],
                                            source_lang: str,
                                            target_lang: str) -> list[str]:
        calls.append((texts, source_lang, target_lang))
        return [f'{texts[0]}_{target_lang.lower()}']

    deepl = DeepL('DUMMY_API_KEY',
                  _supported_languages=frozenset(['EN', 'DE', 'FR']))
    monkeypatch.setattr(deepl, '_send_translation_request', fake_send_translation_request)

    df = pd.DataFrame({'foo': ['hello', 'bye'],
                       'bar': ['ignored 1', 'ignored 2']})

    translated_df = deepl.translate_dataframe(df[['foo']], 'en', ['fr', 'de'])

    expected_df = pd.DataFrame({'foo_fr': ['hello_fr', 'bye_fr'],
                                'foo_de': ['hello_de', 'bye_de']})
    assert_frame_equal(translated_df, expected_df)
    assert sorted(calls) == sorted([(['hello'], 'en', 'fr'),
                                    (['bye'], 'en', 'fr'),
                                    (['hello'], 'en', 'de'),
                                    (['bye'], 'en', 'de')])


def test_translate_dataframe_first_column(monkeypatch: MonkeyPatch):
    calls = []

    async def fake_send_translation_request(texts: list[str],
                                            source_lang: str,
                                            target_lang: str) -> list[str]:
        calls.append((texts, source_lang, target_lang))
        return [f'{texts[0]}_{target_lang.lower()}']

    deepl = DeepL('DUMMY_API_KEY',
                  _supported_languages=frozenset(['EN', 'DE', 'FR']))
    monkeypatch.setattr(deepl, '_send_translation_request', fake_send_translation_request)

    df = pd.DataFrame({'foo': ['hello', 'bye'],
                       'bar': ['ignored 1', 'ignored 2']})

    translated_df = deepl.translate_dataframe(df, 'en', ['fr'])

    expected_df = pd.DataFrame({'foo_fr': ['hello_fr', 'bye_fr']})
    assert_frame_equal(translated_df, expected_df)
    assert sorted(calls) == sorted([(['hello'], 'en', 'fr'),
                                    (['bye'], 'en', 'fr')])


def test_translate_dataframe_failed_rows(monkeypatch: MonkeyPatch):
    async def fake_send_translation_request(texts: list[str],
                                            source_lang: str,
                                            target_lang: str) -> list[str]:
        if texts == ['bye'] and target_lang == 'fr':
            raise DeepLRequestError(429, 'Rate Limited')
        return [f'{texts[0]}_{target_lang.lower()}']

    deepl = DeepL('DUMMY_API_KEY',
                  _supported_languages=frozenset(['EN', 'DE', 'FR']))
    monkeypatch.setattr(deepl, '_send_translation_request', fake_send_translation_request)

    df = pd.DataFrame({'foo': ['hello', 'bye']})

    translated_df = deepl.translate_dataframe(df, 'en', ['fr', 'de'],
                                              raise_on_failed_rows=False,
                                              missing_value='MISSING')

    expected_df = pd.DataFrame({'foo_fr': ['hello_fr', 'MISSING'],
                                'foo_de': ['hello_de', 'bye_de']})
    assert_frame_equal(translated_df, expected_df)


def test_translate_dataframe_empty_inputs():
    deepl = DeepL('DUMMY_API_KEY',
                  _supported_languages=frozenset(['EN', 'DE', 'FR']))

    assert_frame_equal(deepl.translate_dataframe(pd.DataFrame({'foo': []}), 'en', ['fr']),
                       pd.DataFrame())
    assert_frame_equal(deepl.translate_dataframe(pd.DataFrame({'foo': ['hello']}), 'en', []),
                       pd.DataFrame())


def test_translate_dataframe_raise_mode(monkeypatch: MonkeyPatch):
    async def fake_send_translation_request(texts: list[str],
                                            source_lang: str,
                                            target_lang: str) -> list[str]:
        if texts == ['bye'] and target_lang == 'fr':
            raise DeepLRequestError(429, 'Rate Limited')
        return [f'{texts[0]}_{target_lang.lower()}']

    deepl = DeepL('DUMMY_API_KEY',
                  _supported_languages=frozenset(['EN', 'DE', 'FR']))
    monkeypatch.setattr(deepl, '_send_translation_request', fake_send_translation_request)

    df = pd.DataFrame({'foo': ['hello', 'bye']})

    with pytest.raises(DeepLRequestError) as exc_info:
        deepl.translate_dataframe(df, 'en', ['fr'],
                                  raise_on_failed_rows=True)

    assert exc_info.value.status == 429


def test_translate_dataframe_target_langs_list_required():
    deepl = DeepL('DUMMY_API_KEY',
                  _supported_languages=frozenset(['EN', 'DE', 'FR']))
    df = pd.DataFrame({'foo': ['hello']})

    with pytest.raises(ValueError):
        deepl.translate_dataframe(df, 'en', 'fr')
