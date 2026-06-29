import json

import pytest

from df_translate import DeepL


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
                                         monkeypatch: pytest.MonkeyPatch,
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
