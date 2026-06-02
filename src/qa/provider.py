"""Provider layer for QA generation and judging."""

from __future__ import annotations

import json
import time
from typing import Any, Dict

import requests


class BaseJSONProvider:
    """Shared OpenAI-compatible chat-completions client returning parsed JSON."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        rpm_limit: int,
        base_url: str,
        timeout: int,
        extra_headers: Dict[str, str] | None = None,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.delay = 60 / rpm_limit if rpm_limit > 0 else 0.0
        self.session = requests.Session()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)
        self.session.headers.update(headers)

    def _post_chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            raw_text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if not raw_text:
                raise ValueError("empty_response_content")
            return json.loads(raw_text)
        finally:
            if self.delay:
                time.sleep(self.delay)


class DeepSeekJSONProvider(BaseJSONProvider):
    """DeepSeek provider for JSON chat completions."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "deepseek-v4-flash",
        rpm_limit: int = 120,
        base_url: str = "https://api.deepseek.com",
        timeout: int = 120,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            rpm_limit=rpm_limit,
            base_url=base_url,
            timeout=timeout,
        )

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "temperature": 0.2,
            "stream": False,
        }
        return self._post_chat_completion(payload)


class OpenRouterJSONProvider(BaseJSONProvider):
    """OpenRouter provider for JSON chat completions."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "google/gemini-3-flash-preview",
        rpm_limit: int = 120,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 120,
        service_tier: str | None = None,
        http_referer: str | None = None,
        app_title: str | None = "wikiqa-data-pipeline",
    ) -> None:
        self.service_tier = service_tier
        extra_headers: Dict[str, str] = {}
        if http_referer:
            extra_headers["HTTP-Referer"] = http_referer
        if app_title:
            extra_headers["X-OpenRouter-Title"] = app_title
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            rpm_limit=rpm_limit,
            base_url=base_url,
            timeout=timeout,
            extra_headers=extra_headers,
        )

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "stream": False,
        }
        if self.service_tier:
            payload["service_tier"] = self.service_tier
        return self._post_chat_completion(payload)
