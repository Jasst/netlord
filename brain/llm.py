# brain/llm.py
import openai
from openai import OpenAI
from typing import Optional, Iterator, List, Dict
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LLMInterface:
    """
    ИСПРАВЛЕНО: generate()/generate_stream() раньше отправляли ТОЛЬКО одно user-сообщение —
    без system prompt и без истории диалога. Для модели это выглядело как каждый раз новый,
    ничем не связанный вопрос без "личности"/инструкций и без памяти о недавних репликах —
    это частая причина шаблонных, generic-ответов у локальных 7B-моделей. Теперь можно
    передать system и history, а CognitiveBrain.step() их использует.
    """
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct",
                 use_openai_api: bool = False, api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        self.model_name = model_name
        self.use_openai_api = use_openai_api
        if use_openai_api:
            openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.client = openai.OpenAI(api_key=openai.api_key)
        else:
            if base_url is not None:
                self.client = OpenAI(base_url=base_url, api_key="not-needed")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
                self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
                self.client = None

    def _build_messages(self, prompt: str, system: Optional[str] = None,
                         history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, system: Optional[str] = None,
                 history: Optional[List[Dict[str, str]]] = None,
                 max_tokens: int = 256, temperature: float = 0.7) -> str:
        if self.use_openai_api or self.client is not None:
            messages = self._build_messages(prompt, system, history)
            response = self.client.chat.completions.create(
                model=self.model_name if self.use_openai_api else "local-model",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        else:
            full_prompt = prompt
            if system:
                full_prompt = f"{system}\n\n{prompt}"
            result = self.pipeline(full_prompt, max_new_tokens=max_tokens, temperature=temperature, do_sample=True)
            return result[0]["generated_text"][len(full_prompt):].strip()

    def generate_stream(self, prompt: str, system: Optional[str] = None,
                         history: Optional[List[Dict[str, str]]] = None,
                         max_tokens: int = 256, temperature: float = 0.7) -> Iterator[str]:
        if self.use_openai_api or self.client is not None:
            messages = self._build_messages(prompt, system, history)
            stream = self.client.chat.completions.create(
                model=self.model_name if self.use_openai_api else "local-model",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        else:
            full = self.generate(prompt, system, history, max_tokens, temperature)
            for word in full.split():
                yield word + " "
