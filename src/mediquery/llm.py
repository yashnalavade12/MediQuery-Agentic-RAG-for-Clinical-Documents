"""LLM interface for MediQuery.

OllamaClient talks to a local Ollama server (e.g. Gemma 2B) -- fully
offline, no API keys, no data leaves the machine. When Ollama is not
available the agent loop uses deterministic extractive synthesis instead,
so the project runs anywhere with just the standard library.
"""

import json
import urllib.request


class OllamaClient:
    def __init__(
        self,
        model="gemma2:2b",
        base_url="http://localhost:11434",
        timeout=180,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def available(self):
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/api/tags", timeout=5
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    def generate(self, prompt):
        payload = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": False}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body.get("response", "")
