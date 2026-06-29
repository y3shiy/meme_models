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

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    def post(self, *args, **kwargs) -> FakeResponse:
        self.post_args = args
        self.post_kwargs = kwargs

        return FakeResponse(200, {'translations': [{'text': 'Bonjour le monde!'}]})

    def get(self, *args, **kwargs) -> FakeResponse:
        return FakeResponse(200, [])


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
