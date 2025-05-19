'''
Created on May 17, 2025

@author: Yihao Fang
'''
"""A server that provides OpenAI-compatible RESTful APIs.

It current only supports Chat Completions: https://platform.openai.com/docs/api-reference/chat)
"""

import logging
import os
import time
from collections import defaultdict
from typing import AsyncGenerator, Dict, List, Literal, Optional, Union

import fastapi
import shortuuid
import uvicorn
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from openai import AsyncOpenAI
import copy
import asyncio
from node import serve_node
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

with open("config/config.json","r") as f:
    config = json.load(f)

router_port = config.get("router_port", None)
vllm_ports = config.get("vllm_ports", None)
model = config.get("model", None)
vllm_devices = config.get("vllm_devices", None)


count = defaultdict(lambda: defaultdict(int))


@asynccontextmanager
async def lifespan(app):
    yield


app = fastapi.FastAPI(lifespan=lifespan)


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    # OpenAI fields: https://platform.openai.com/docs/api-reference/chat/create
    model: str
    messages: Union[
        str,
        List[Dict[str, str]],
        List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]],
    ]
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[int, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[Dict[str, str]] = (
        None  # { "type": "json_object" } for json mode
    )
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    tools: Optional[List[Dict[str, Union[str, int, float]]]] = None
    tool_choice: Optional[str] = None
    user: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


async def stream_response(response) -> AsyncGenerator:
    async for chunk in response:
        yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # The model name field contains the parameters for routing.
    # Model name uses format router-[router name]-[threshold] e.g. router-bert-0.7
    # The router type and threshold is used for routing that specific request.
    logging.info(f"Received request: {request}")
    
    async def make_request(port, req_dict_per_c):
        client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{port}/v1",
        )
        res = await client.chat.completions.create(**req_dict_per_c)
        return res
    
    try:
        req_dict = request.model_dump(exclude_none=True)
        n = req_dict["n"]
        l = len(vllm_ports)
        n_partition = [round(n/l)]*(l-1)+ [n-(l-1)*(round(n/l))]
        req_dicts = []
        for n_per_c in n_partition:
            req_dict_per_c = copy.deepcopy(req_dict)
            req_dict_per_c["n"] = n_per_c
            req_dicts.append(req_dict_per_c)
            
        tasks = [asyncio.create_task(make_request(port, req_dict_per_c)) for port, req_dict_per_c in zip(vllm_ports, req_dicts)]    
        results = await asyncio.gather(*tasks)
        
        resp = results[0]
        
        for r in results[1:]:
            resp.choices.extend(r.choices)
        
    except Exception as e:
        return JSONResponse(
            ErrorResponse(message=str(e)).model_dump(),
            status_code=400,
        )
    
    if request.stream:
        return StreamingResponse(
            content=stream_response(resp), media_type="text/event-stream"
        )
    else:
        return JSONResponse(content=resp.model_dump())


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={"status": "online"})


if __name__ == "__main__":
    serve_node(model = model, ports = vllm_ports, devices = vllm_devices)
    uvicorn.run(
        "router:app",
        port=router_port,
        host="0.0.0.0",
        workers=0,
    )
