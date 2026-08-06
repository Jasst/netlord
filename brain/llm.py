# brain/llm.py
import openai
from openai import OpenAI
from typing import Optional, Iterator
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

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        if self.use_openai_api or self.client is not None:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model=self.model_name if self.use_openai_api else "local-model",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        else:
            result = self.pipeline(prompt, max_new_tokens=max_tokens, temperature=temperature, do_sample=True)
            return result[0]["generated_text"][len(prompt):].strip()

    def generate_stream(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> Iterator[str]:
        if self.use_openai_api or self.client is not None:
            messages = [{"role": "user", "content": prompt}]
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
            full = self.generate(prompt, max_tokens, temperature)
            for word in full.split():
                yield word + " "