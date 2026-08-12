# brain/llm.py
import openai
from openai import OpenAI
from typing import Optional, Iterator, List, Dict
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LLMInterface:
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

    def _sampling_kwargs(self, temperature: float, top_p: Optional[float],
                          repetition_penalty: Optional[float], top_k: Optional[int],
                          presence_penalty: Optional[float], enable_thinking: Optional[bool]) -> Dict:
        kwargs = {"temperature": temperature}
        if top_p is not None:
            kwargs["top_p"] = top_p
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty

        extra_body = {}
        if not self.use_openai_api:
            if repetition_penalty is not None:
                extra_body["repeat_penalty"] = repetition_penalty
            if top_k is not None:
                extra_body["top_k"] = top_k
        if enable_thinking is not None:
            extra_body["enable_thinking"] = enable_thinking
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def generate(self, prompt: str, system: Optional[str] = None,
                 history: Optional[List[Dict[str, str]]] = None,
                 max_tokens: int = 256, temperature: float = 0.7,
                 top_p: Optional[float] = None, repetition_penalty: Optional[float] = None,
                 top_k: Optional[int] = None, presence_penalty: Optional[float] = None,
                 enable_thinking: Optional[bool] = None) -> str:
        if self.use_openai_api or self.client is not None:
            messages = self._build_messages(prompt, system, history)
            response = self.client.chat.completions.create(
                model=self.model_name if self.use_openai_api else "local-model",
                messages=messages,
                max_tokens=max_tokens,
                **self._sampling_kwargs(temperature, top_p, repetition_penalty, top_k,
                                        presence_penalty, enable_thinking),
            )
            return response.choices[0].message.content.strip()
        else:
            full_prompt = prompt
            if system:
                full_prompt = f"{system}\n\n{prompt}"
            gen_kwargs = dict(max_new_tokens=max_tokens, temperature=temperature, do_sample=True)
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
            if repetition_penalty is not None:
                gen_kwargs["repetition_penalty"] = repetition_penalty
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
            # Для локального HF-пайплайна enable_thinking не поддерживается, игнорируем
            result = self.pipeline(full_prompt, **gen_kwargs)
            return result[0]["generated_text"][len(full_prompt):].strip()

    def generate_stream(self, prompt: str, system: Optional[str] = None,
                         history: Optional[List[Dict[str, str]]] = None,
                         max_tokens: int = 256, temperature: float = 0.7,
                         top_p: Optional[float] = None, repetition_penalty: Optional[float] = None,
                         top_k: Optional[int] = None, presence_penalty: Optional[float] = None,
                         enable_thinking: Optional[bool] = None) -> Iterator[str]:
        if self.use_openai_api or self.client is not None:
            messages = self._build_messages(prompt, system, history)
            stream = self.client.chat.completions.create(
                model=self.model_name if self.use_openai_api else "local-model",
                messages=messages,
                max_tokens=max_tokens,
                stream=True,
                **self._sampling_kwargs(temperature, top_p, repetition_penalty, top_k,
                                        presence_penalty, enable_thinking),
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        else:
            full = self.generate(prompt, system, history, max_tokens, temperature,
                                  top_p=top_p, repetition_penalty=repetition_penalty,
                                  top_k=top_k, presence_penalty=presence_penalty,
                                  enable_thinking=enable_thinking)
            for word in full.split():
                yield word + " "