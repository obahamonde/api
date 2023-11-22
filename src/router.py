"""
API Router for /plan
"""
from typing import Literal, Optional

from fastapi import Depends, File, UploadFile
from sse_starlette.sse import EventSourceResponse

from .schemas import APIProxy, Assistant, FileObject, Run, Stack, Thread, ThreadMessage

app = APIProxy()


@app.post("/files", response_model=FileObject)
async def get_files(file: UploadFile = File(...)):
    """
    List files
    """
    content_type = file.content_type or "application/octet-stream"
    response = await app.push_file(file=file)
    response.status_details = content_type
    return response


@app.delete("/files/{file_id}", response_model=None)
async def delete_files(file_id: str):
    """
    Delete files
    """
    await app.ai.files.delete(file_id=file_id)


@app.get("/messages", response_model=ThreadMessage)
async def get_message(content: str, file_ids: str, thread_id: str):
    """
    Create a message
    """
    return await app.push_message(
        content=content, file_ids=file_ids, thread_id=thread_id
    )


@app.post("/threads", response_model=Thread)
async def get_thread(stack: Stack = Depends(Stack)):
    """
    Create a thread
    """
    return await app.create_thread(stack=stack)


@app.delete("/threads/{thread_id}", response_model=None)
async def delete_thread(thread_id: str):
    """
    Delete a thread
    """
    await app.ai.beta.threads.delete(thread_id=thread_id)


@app.get("/runs", response_model=Run)
async def get_run(assistant_id: str, thread_id: str):
    """
    Create a run
    """
    return await app.create_run(assistant_id=assistant_id, thread_id=thread_id)


@app.delete("/runs/{run_id}", response_model=None)
async def delete_run(run_id: str, thread_id: str):
    """
    Delete a run
    """
    pass


@app.get("/assistants", response_model=Assistant)
async def get_assistant(
    name: str,
    instructions: str,
    file_ids: Optional[str] = None,
    model: Literal["gpt-4-1106-preview", "gpt-3.5-turbo-1106"] = "gpt-4-1106-preview",
):
    return await app.create_assistant(
        model=model, instructions=instructions, file_ids=file_ids, name=name
    )


@app.delete("/assistants/{assistant_id}", response_model=None)
async def delete_assistant(assistant_id: str):
    """
    Delete an assistant
    """
    await app.ai.beta.assistants.delete(assistant_id=assistant_id)


@app.get("/events", response_class=EventSourceResponse)
async def event_stream(thread_id: str, run_id: str):
    """
    Stream events
    """
    return await app.stream_steps(thread_id=thread_id, run_id=run_id)
