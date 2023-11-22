"""
A module that contains custom response classes for FastAPI.
"""
from typing import Any, AsyncGenerator, Optional

from fastapi.responses import StreamingResponse

from .decorators import setup_logging

Event = dict[str, Any]
logger = setup_logging(__name__)


class AudioResponse(StreamingResponse):
    """
    A response class that streams audio to the client.

    Args:
        audio_generator (AsyncGenerator[bytes, bytes]): An asynchronous generator that yields audio to be streamed.
        status_code (int, optional): The HTTP status code to return. Defaults to 200.
        headers (dict[str, str], optional): A dictionary of HTTP headers to include in the response. Defaults to
            {"Cache-Control": "no-cache", "Connection": "keep-alive", "Content-Type": "audio/mpeg"}.
        media_type (str, optional): The media type of the response. Defaults to "audio/mpeg".
    """

    def __init__(
        self,
        audio_generator: AsyncGenerator[bytes, None],
        *,
        status_code: int = 200,
        headers: Optional[dict[str, str]] = None,
        media_type: str = "audio/opus",
    ):
        super().__init__(
            self.generate(audio_generator),
            status_code=status_code,
            headers=headers
            or {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": str(media_type),
            },
            media_type=media_type,
        )

    async def generate(
        self, audio_generator: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[bytes, None]:
        """
        Asynchronously generates a stream of bytes responses from an async generator of audio.

        Args:
            audio_generator (AsyncGenerator[bytes, bytes]): An async generator that yields audio.

        Yields:
            AsyncGenerator[bytes, None]: An async generator that yields bytes responses.
        """
        async for audio in audio_generator:
            yield audio
