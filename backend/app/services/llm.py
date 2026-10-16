import logging
from functools import lru_cache

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from app.core.config import get_settings
from app.core.retry import call_with_retry

log = logging.getLogger(__name__)


class LLMError(Exception):
    '''Raised when the language model fails or returns unusable output.'''


class LLMAnswer(BaseModel):
    answerable: bool
    answer: str
    sources: list[str]


class GeminiLLM:
    def __init__(self) -> None:
        s = get_settings()
        if not s.gemini_api_key:
            raise RuntimeError('GEMINI_API_KEY is not set')
        self._client = genai.Client(api_key=s.gemini_api_key)
        self._model = s.gemini_model
        self._temperature = s.llm_temperature
        self._max_tokens = s.llm_max_output_tokens

    def generate_answer(self, system_prompt: str, user_prompt: str) -> LLMAnswer:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
            response_mime_type='application/json',
            response_schema=LLMAnswer,
        )
        try:
            response = call_with_retry(
                lambda: self._client.models.generate_content(
                    model=self._model, contents=user_prompt, config=config
                )
            )
        except errors.APIError as exc:
            code = getattr(exc, 'code', '?')
            raise LLMError(f'Gemini request failed ({code}): {exc}') from exc

        text = response.text
        if not text:
            raise LLMError('Gemini returned an empty response (possibly blocked by safety filters).')
        try:
            return LLMAnswer.model_validate_json(text)
        except ValidationError as exc:
            raise LLMError(f'Gemini returned malformed JSON: {exc}') from exc


@lru_cache
def get_llm() -> GeminiLLM:
    return GeminiLLM()
