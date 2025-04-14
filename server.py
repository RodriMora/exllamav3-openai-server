import argparse
import os
import sys
import json
from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import torch
import uvicorn
import uuid
from threading import Lock
import time

# Import ExLLaMA v3 components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Generator, Job, model_init
from chat_templates import prompt_formats

# Models and generators storage
loaded_models = {}
generators = {}
model_lock = Lock()

# API Models (Pydantic schemas)
class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 1677610602
    owned_by: str = "exllama"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 1000
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    max_total_tokens: Optional[int] = None  # Added for compatibility

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = 1677610602
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Dict[str, int]

class StreamChoice(BaseModel):
    index: int
    delta: Dict[str, Optional[str]]
    finish_reason: Optional[str] = None

class StreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = 1677610602
    model: str
    choices: List[StreamChoice]

# Config options for the server
class ServerConfig:
    def __init__(self, args):
        self.args = args
        self.model_name = args.model_name or "local-model"

        # Figure out the appropriate prompt format
        self.mode = args.mode
        self.user_name = args.user_name
        self.bot_name = args.bot_name
        self.system_prompt = args.system_prompt
        self.prompt_format = prompt_formats[self.mode](self.user_name, self.bot_name)

        if not self.system_prompt:
            self.system_prompt = self.prompt_format.default_system_prompt()

        self.add_bos = self.prompt_format.add_bos()

# Initialize the model and generator
def initialize_model(config, model_path=None):
    with model_lock:
        if model_path in loaded_models:
            return loaded_models[model_path], generators[model_path]

        # Initialize the model
        args = config.args

        # If model_path is specified, update the args
        if model_path:
            args.model_dir = model_path

        model, model_config, cache, tokenizer = model_init.init(args)

        # Create generator
        generator = Generator(
            model=model,
            cache=cache,
            tokenizer=tokenizer,
        )

        # Store the model and generator
        loaded_models[model_path] = (model, model_config, cache, tokenizer)
        generators[model_path] = generator

        return loaded_models[model_path], generators[model_path]

# Start-up context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the default model on startup
    if config.args.model_dir:
        initialize_model(config, config.args.model_dir)
    yield
    # Clean up on shutdown
    # Nothing to do for now

# Create the FastAPI app
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/v1/models")
async def list_models():
    """List available models"""
    models = []

    # Add the default model
    if config.args.model_dir:
        model_name = config.model_name
        models.append(ModelCard(id=model_name))

    # Include additional model directories if specified
    if config.args.additional_models:
        for model_path in config.args.additional_models:
            base_name = os.path.basename(os.path.normpath(model_path))
            models.append(ModelCard(id=base_name))

    return ModelList(data=models)

async def generate_stream_response(generator, job, response_id, model_name):
    """Generate streaming response for chat completions"""
    generator.enqueue(job)
    completion_text = ""
    finish_reason = "stop"

    # First chunk with assistant role
    yield f"data: {json.dumps(StreamResponse(id=response_id, model=model_name, choices=[StreamChoice(index=0, delta={'role': 'assistant'})]).dict())}\n\n"

    while generator.num_remaining_jobs():
        for r in generator.iterate():
            chunk = r.get("text", "")
            completion_text += chunk

            if chunk:
                yield f"data: {json.dumps(StreamResponse(id=response_id, model=model_name, choices=[StreamChoice(index=0, delta={'content': chunk})]).dict())}\n\n"

            if r["eos"]:
                if r["eos_reason"] == "max_new_tokens":
                    finish_reason = "length"
                break

    # Final chunk with finish reason
    yield f"data: {json.dumps(StreamResponse(id=response_id, model=model_name, choices=[StreamChoice(index=0, delta={}, finish_reason=finish_reason)]).dict())}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Process a chat completion request"""
    # Determine the model to use
    model_path = config.args.model_dir
    if request.model != config.model_name and config.args.additional_models:
        # Check if the requested model is in additional_models
        matching_models = [m for m in config.args.additional_models if os.path.basename(os.path.normpath(m)) == request.model]
        if matching_models:
            model_path = matching_models[0]
        else:
            raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

    # Initialize the model if not already loaded
    (model, model_config, cache, tokenizer), generator = initialize_model(config, model_path)

    # Prepare the messages
    context = []
    system_prompt = config.system_prompt

    # Extract system message if present
    for msg in request.messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user" or msg.role == "assistant":
            if len(context) > 0 and context[-1][0] is None and msg.role == "assistant":
                # Add assistant response to the last user message
                context[-1] = (context[-1][0], msg.content)
            elif msg.role == "user":
                context.append((msg.content, None))
            elif msg.role == "assistant":
                context.append((None, msg.content))

    # Ensure the last message has a response if it's from a user
    if context and context[-1][1] is None:
        # Format the prompt
        prompt_format = config.prompt_format
        frm_context = prompt_format.format(system_prompt, context)

        # Tokenize
        input_ids = tokenizer.encode(frm_context, add_bos=config.add_bos, encode_special_tokens=True)

        # Get context length from cache
        model_context_length = cache.max_num_tokens

        # Determine max tokens for generation
        max_tokens = request.max_tokens or 1000

        # Handle max_total_tokens if specified
        current_tokens = input_ids.shape[-1]

        if request.max_total_tokens:
            if request.max_total_tokens > model_context_length:
                raise HTTPException(
                    status_code=400,
                    detail=f"max_total_tokens ({request.max_total_tokens}) exceeds model's context window ({model_context_length})"
                )
            # Adjust max_tokens to respect max_total_tokens
            max_tokens = min(max_tokens, request.max_total_tokens - current_tokens)
        else:
            # Make sure we don't exceed the model's context length
            max_tokens = min(max_tokens, model_context_length - current_tokens)

        if max_tokens <= 0:
            raise HTTPException(
                status_code=400,
                detail=f"Input context is too long. Current tokens: {current_tokens}, available: {model_context_length - current_tokens}"
            )

        # Prepare stop conditions
        stop_conditions = prompt_format.stop_conditions(tokenizer)
        if request.stop:
            if isinstance(request.stop, str):
                stop_conditions.append(request.stop)
            else:
                stop_conditions.extend(request.stop)

        # Create job with limited max_new_tokens to respect context limits
        job = Job(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop_conditions=stop_conditions,
        )

        if request.seed is not None:
            torch.manual_seed(request.seed)

        # Process the request
        response_id = f"chatcmpl-{str(uuid.uuid4())}"

        if request.stream:
            # Streaming mode - use StreamingResponse
            return StreamingResponse(
                generate_stream_response(generator, job, response_id, request.model),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming mode
            generator.enqueue(job)
            completion_text = ""
            finish_reason = "stop"

            while generator.num_remaining_jobs():
                for r in generator.iterate():
                    chunk = r.get("text", "")
                    completion_text += chunk

                    if r["eos"]:
                        if r["eos_reason"] == "max_new_tokens":
                            finish_reason = "length"
                        break

            # Calculate token usage
            input_tokens = input_ids.shape[-1]
            output_tokens = len(tokenizer.encode(completion_text, add_bos=False))

            # Return the response
            return ChatCompletionResponse(
                id=response_id,
                model=request.model,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=Message(role="assistant", content=completion_text),
                        finish_reason=finish_reason
                    )
                ],
                usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }
            )
    else:
        # If there's no user message to respond to, return an error
        raise HTTPException(status_code=400, detail="Invalid request: No user message to respond to")

def parse_args():
    parser = argparse.ArgumentParser(description="ExLLaMA v3 OpenAI-compatible API Server")

    # ExLLaMA model arguments
    model_init.add_args(parser, cache=True)

    # API server arguments
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--model-name", type=str, default=None, help="Name to use for the model in API responses")
    parser.add_argument("--additional-models", type=str, nargs="+", help="Additional model directories to serve")

    # Chat formatting arguments
    parser.add_argument("-mode", "--mode", type=str, default="raw", help="Prompt mode")
    parser.add_argument("-un", "--user-name", type=str, default="User", help="User name")
    parser.add_argument("-bn", "--bot-name", type=str, default="Assistant", help="Bot name")
    parser.add_argument("-sp", "--system-prompt", type=str, help="Use custom system prompt")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = ServerConfig(args)

    # If no model name is provided, use the basename of the model directory
    if config.model_name == "local-model" and args.model_dir:
        config.model_name = os.path.basename(os.path.normpath(args.model_dir))

    print(f"Starting ExLLaMA v3 OpenAI-compatible API on {args.host}:{args.port}")
    print(f"Serving model: {config.model_name}")

    if args.additional_models:
        print("Additional models:")
        for model_path in args.additional_models:
            print(f" - {os.path.basename(os.path.normpath(model_path))}")

    uvicorn.run(app, host=args.host, port=args.port)
