from typing import Literal

from fastapi.responses import PlainTextResponse
from openai import AsyncOpenAI
from sse_starlette.sse import EventSourceResponse

from src import create_app
from src.lib.functions import (
    use_chat,
    use_chat_stream,
    use_instruction,
    use_tts,
    use_vision,
)
from src.lib.responses import AudioResponse
from src.schemas import similarity_search, upsert_vector

app = create_app()
ai = AsyncOpenAI()


@app.get("/api/chat")
async def chat(text: str, namespace: str):
    context = await similarity_search(text=text, namespace=namespace)

    async def generator():
        string: str = ""
        async for response in use_chat_stream(
            text=text,
            ai=ai,
            context=context,
            temperature=0.2,
            max_tokens=512,
            model="gpt-3.5-turbo-1106",
        ):
            if isinstance(response, str):
                string += response
            else:
                string += response["data"]

            yield response
        await upsert_vector(text=string, namespace=namespace)
        # The line `yield {"event": "done", "data": string}` is used to send an event to the client
        # indicating that the chat conversation is done and providing the final response data. This is
        # done in the `/api/chat` endpoint.
        yield {"event": "done", "data": string}

    return EventSourceResponse(generator(), sep="\r\n")


@app.get("/api/utterance")
async def utterance(text: str):
    response = await use_chat(
        text=text,
        ai=ai,
        context=None,
        temperature=0.2,
        max_tokens=512,
        model="gpt-3.5-turbo-1106",
    )
    return PlainTextResponse(response)


@app.get("/api/audio")
async def audio(
    text: str,
    model: Literal["tts-1", "tts-1-hd"] = "tts-1",
    voice: Literal["nova", "alloy", "echo", "fable", "onyx", "shimmer"] = "nova",
    response_format: Literal["mp3", "opus", "aac", "flac"] = "opus",
):
    return AudioResponse(
        use_tts(
            text=text,
            ai=ai,
            model=model,
            voice=voice,
            response_format=response_format,
        ),
        media_type=f"audio/{response_format}",
    )


@app.get("/api/autocomplete/{topic}")
async def autocompletion(text: str, topic: str):
    context = f"Autocomplete the following {topic}:\n\n{text}"

    response = await use_instruction(
        ai=ai,
        text=context,
    )
    return PlainTextResponse(response)


@app.get("/api/vision")
async def vision(
    text: str,
    url: str,
    tokens: int = 1024,
):
    response = await use_vision(
        ai=ai,
        image=url,
        model="gpt-4-vision-preview",
        text=text,
        max_tokens=tokens,
        temperature=0.8,
    )
    return PlainTextResponse(response)
