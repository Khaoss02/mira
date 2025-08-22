from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, AsyncIterator

from .stateless_llm_interface import StatelessLLMInterface

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


class VLLMStatelessLLM(StatelessLLMInterface):
    """
    Stateless LLM wrapper for vLLM (e.g. Qwen2.5-Omni-7B).
    Can run locally or via an API if configured.
    """

    def __init__(self,
                 model: Optional[str] = None,
                 api_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 **kwargs: Any) -> None:
        self.api_url = api_url or os.getenv("LLM_API_URL")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self.kwargs = kwargs  # Store extra kwargs like temperature

        if self.api_url and self.api_key:
            # API mode
            self.client = None
            print(f"[VLLMStatelessLLM] Using API endpoint: {self.api_url}")
        else:
            if not _HAS_VLLM:
                raise RuntimeError("vLLM not installed. Install with `pip install vllm`.")
            self.client = LLM(model=self.model, **kwargs)
            print(f"[VLLMStatelessLLM] Loaded local vLLM model: {self.model}")

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """
        Generates a chat completion asynchronously and yields chunks.
        Concatenates messages into a single prompt for vLLM.
        Tools are ignored for now (extend if needed).
        """
        # Concatenate system and messages into a single prompt
        prompt = ""
        if system:
            prompt += f"System: {system}\n\n"
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"{role.capitalize()}: {content}\n"
        prompt += "Assistant:"

        if self.api_url and self.api_key:
            # API call mode (OpenAI-compatible /v1/chat/completions with stream)
            import httpx
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                **self.kwargs,  # e.g., temperature
            }
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", f"{self.api_url}/v1/chat/completions",
                                         headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes():
                        # Parse chunks (assuming SSE format)
                        for line in chunk.decode().splitlines():
                            if line.startswith("data: "):
                                data = line[6:]
                                if data == "[DONE]":
                                    return
                                try:
                                    import json
                                    json_data = json.loads(data)
                                    delta = json_data["choices"][0]["delta"].get("content", "")
                                    if delta:
                                        yield delta
                                except json.JSONDecodeError:
                                    pass
        else:
            # Local vLLM inference with streaming
            params = SamplingParams(
                temperature=self.kwargs.get("temperature", 0.7),
                max_tokens=self.kwargs.get("max_tokens", 512),
            )
            outputs = self.client.generate(prompt, params, use_tqdm=False)
            for output in outputs:
                for token in output.outputs[0].text:  # Yield char-by-char or token-by-token
                    yield token