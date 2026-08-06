# app.py
import asyncio
import json
import os
import time
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from typing import Optional, List
import atexit
import signal
import sys

from brain import CognitiveBrain, BrainConfig

torch.set_default_dtype(torch.float32)

LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")

config = BrainConfig(
    dim_embedding=384,
    gnn_hidden_dim=256,
    gnn_num_heads=4,
    max_neurons=100000,
    max_synapses=500000,
    working_memory_size=10,
    episodic_capacity=50000,
    model_dir="brain_model_v9",
    learning_rate=1e-4,
    checkpoint_every=50,
    embedding_model="intfloat/e5-large-v2",
    llm_model="Qwen/Qwen2-7B-Instruct",
    use_openai_api=False,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    enable_curiosity=True,
    enable_planning=True,
    enable_reflection=True,
    enable_ewc=True,
)
brain = CognitiveBrain(config)
brain.load()
brain.load_dialog_history()

app = FastAPI(title="Smart Brain v9")
app.mount("/static", StaticFiles(directory="static"), name="static")

class AskRequest(BaseModel):
    question: str
    temperature: float = 0.7
    use_search: bool = False
    allow_clarifying: bool = True

class LearnRequest(BaseModel):
    question: str
    answer: str

class TrainTopicRequest(BaseModel):
    topic: str
    num_pairs: int = 30
    negative_ratio: float = 0.2
    epochs: int = 1

@app.post("/ask")
async def ask(req: AskRequest):
    try:
        result = await asyncio.to_thread(brain.step, req.question, use_search=req.use_search)
        answer = result["answer"]
        brain.dialog_memory.append({"user": req.question, "assistant": answer, "time": time.time()})
        await asyncio.to_thread(brain.save_dialog_history)
        return {
            "question": req.question,
            "answer": answer,
            "facts": result.get("memory_results", []),
            "known": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask_stream")
async def ask_stream(req: AskRequest):
    try:
        gen = await asyncio.to_thread(brain.step_stream, req.question, use_search=req.use_search)
        async def generate():
            full = ""
            for token in gen:
                full += token
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True, 'full_answer': full})}\n\n"
            brain.dialog_memory.append({"user": req.question, "assistant": full, "time": time.time()})
            await asyncio.to_thread(brain.save_dialog_history)
        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learn")
async def learn(req: LearnRequest):
    try:
        await asyncio.to_thread(brain.learn_pair, req.question, req.answer)
        await asyncio.to_thread(brain.save)
        return {"status": "learned"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_topic")
async def train_topic(req: TrainTopicRequest):
    try:
        pairs = await asyncio.to_thread(_generate_training_pairs, req.topic, req.num_pairs)
        for q, a in pairs:
            brain.learn_pair(q, a, epochs=req.epochs)
        brain.sleep()
        brain.save()
        return {"status": "ok", "pairs_learned": len(pairs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _generate_training_pairs(topic: str, num_pairs: int) -> List[tuple]:
    prompt = f"Сгенерируй {num_pairs} пар 'вопрос|ответ' по теме '{topic}'. Формат: каждая пара на новой строке, разделённая '|'."
    response = brain.llm.generate(prompt, max_tokens=num_pairs*30, temperature=0.9)
    pairs = []
    for line in response.splitlines():
        if '|' in line:
            q, a = line.split('|', 1)
            pairs.append((q.strip(), a.strip()))
    return pairs[:num_pairs]

@app.get("/stats")
async def stats():
    return brain.get_stats()

@app.post("/sleep")
async def sleep_brain():
    await asyncio.to_thread(brain.sleep)
    return {"status": "sleep_done"}

@app.post("/chat/clear")
async def clear_chat():
    brain.dialog_memory.clear()
    await asyncio.to_thread(brain.save_dialog_history)
    return {"status": "cleared"}

@app.get("/chat/messages")
async def get_messages(limit: int = 50):
    return {"messages": brain.dialog_memory[-limit:]}

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

def save_brain():
    print("\n💾 Сохраняем модель...")
    brain.save()
    brain.save_dialog_history()

atexit.register(save_brain)
signal.signal(signal.SIGINT, lambda s, f: (save_brain(), sys.exit(0)))
signal.signal(signal.SIGTERM, lambda s, f: (save_brain(), sys.exit(0)))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)