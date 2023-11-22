from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import HTTPException
from pydantic import BaseModel  # pylint: disable=E0611

from .vector import AsyncClient, LazyProxy

Method = Literal["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS", "TRACE"]
Json = Union[Dict[str, object], List[object], str, int, float, bool, None]


class APIClient(LazyProxy[AsyncClient], BaseModel):
    """
    Generic Lazy Loading APIClient
    """

    base_url: str
    headers: Dict[str, str]

    def __load__(self):
        return AsyncClient(base_url=self.base_url, headers=self.headers)

    def dict(self, *args: Any, **kwargs: Any):
        return super().dict(*args, exclude={"headers"}, **kwargs)

    async def fetch(
        self,
        *,
        method: Method,
        url: str,
        json: Optional[Json] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        if method in ("GET", "DELETE"):
            if json is not None:
                raise ValueError(f"json must be None for {method} requests")
            if headers is not None:
                headers = {**self.headers, **headers}
            else:
                headers = self.headers
            response = await self.__load__().request(
                method=method, url=url, headers=headers
            )
        else:
            if headers is not None:
                headers = {**self.headers, **headers}
            else:
                headers = self.headers
            response = await self.__load__().request(
                method=method, url=url, headers=headers, json=json
            )
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response

    async def get(
        self,
        *,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ):
        response = await self.fetch(method="GET", url=url, headers=headers)
        return response.json()

    async def post(
        self,
        *,
        url: str,
        json: Optional[Json] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        response = await self.fetch(method="POST", url=url, json=json, headers=headers)
        return response.json()

    async def put(
        self,
        *,
        url: str,
        json: Optional[Json] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        response = await self.fetch(method="PUT", url=url, json=json, headers=headers)
        return response.json()

    async def delete(
        self,
        *,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ):
        response = await self.fetch(method="DELETE", url=url, headers=headers)
        return response.json()

    async def patch(
        self,
        *,
        url: str,
        json: Optional[Json] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        response = await self.fetch(method="PATCH", url=url, json=json, headers=headers)
        return response.json()

    async def head(
        self,
        *,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ):
        response = await self.fetch(method="HEAD", url=url, headers=headers)
        return response.json()

    async def options(
        self,
        *,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ):
        response = await self.fetch(method="OPTIONS", url=url, headers=headers)
        return response.json()

    async def trace(
        self,
        *,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ):
        response = await self.fetch(method="TRACE", url=url, headers=headers)
        return response.json()

    async def text(
        self,
        *,
        url: str,
        method: Method,
        json: Optional[Json] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        response = await self.fetch(method=method, url=url, json=json, headers=headers)
        return response.text

    async def blob(
        self,
        *,
        url: str,
        method: Method,
        json: Optional[Json] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        response = await self.fetch(method=method, url=url, json=json, headers=headers)
        return response.content

    async def stream(
        self,
        *,
        url: str,
        method: Method,
        json: Optional[Json] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        response = await self.fetch(method=method, url=url, json=json, headers=headers)
        async for chunk in response.aiter_bytes():
            yield chunk
