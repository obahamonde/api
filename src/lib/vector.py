from __future__ import annotations

from abc import ABC, abstractmethod
from os import environ
from typing import Any, Dict, Generic, Iterable, List, Literal, TypeVar, Union, cast
from uuid import uuid4

from httpx import AsyncClient
from pydantic import BaseModel, Field  # pylint: disable=E0611

from .decorators import handle

T = TypeVar("T")
Value = Union[str, int, float, bool, List[str]]
Filter = Literal["$eq", "$ne", "$lt", "$lte", "$gt", "$gte", "$in", "$nin"]
AndOr = Literal["$and", "$or"]
Query = Union[
    Dict[str, Union[Value, "Query", List["Query"], List[Value]]],
    Dict[Filter, Value],
    Dict[AndOr, List["Query"]],
]
Vector = List[float]
MetaData = Dict[str, Value]


class LazyProxy(Generic[T], ABC):
    def __init__(self) -> None:
        self.__proxied: T | None = None

    def __getattr__(self, attr: str) -> object:
        return getattr(self.__get_proxied__(), attr)

    def __repr__(self) -> str:
        return repr(self.__get_proxied__())

    def __str__(self) -> str:
        return str(self.__get_proxied__())

    def __dir__(self) -> Iterable[str]:
        return self.__get_proxied__().__dir__()

    def __get_proxied__(self) -> T:
        proxied = self.__proxied
        if proxied is not None:
            return proxied

        self.__proxied = proxied = self.__load__()
        return proxied

    def __set_proxied__(self, value: T) -> None:
        self.__proxied = value

    def __as_proxied__(self) -> T:
        """Helper method that returns the current proxy, typed as the loaded object"""
        return cast(T, self)

    @abstractmethod
    def __load__(self) -> T:
        ...


class QueryBuilder:
    """Query builder for Pinecone Query API with MongoDB-like syntax."""

    def __init__(self, field: str = None, query: Query = None):  # type: ignore
        self.field = field
        self.query = query if query else {}

    def __repr__(self) -> str:
        return f"{self.query}"

    def __call__(self, field_name: str) -> QueryBuilder:
        return QueryBuilder(field_name)

    def __and__(self, other: QueryBuilder) -> QueryBuilder:
        return QueryBuilder(query={"$and": [self.query, other.query]})

    def __or__(self, other: QueryBuilder) -> QueryBuilder:
        return QueryBuilder(query={"$or": [self.query, other.query]})

    def __eq__(self, value: Value) -> QueryBuilder:  # type: ignore
        return QueryBuilder(query={self.field: {"$eq": value}})

    def __ne__(self, value: Value) -> QueryBuilder:  # type: ignore
        return QueryBuilder(query={self.field: {"$ne": value}})

    def __lt__(self, value: Value) -> QueryBuilder:
        return QueryBuilder(query={self.field: {"$lt": value}})

    def __le__(self, value: Value) -> QueryBuilder:
        return QueryBuilder(query={self.field: {"$lte": value}})

    def __gt__(self, value: Value) -> QueryBuilder:
        return QueryBuilder(query={self.field: {"$gt": value}})

    def __ge__(self, value: Value) -> QueryBuilder:
        return QueryBuilder(query={self.field: {"$gte": value}})

    def in_(self, values: List[Value]) -> QueryBuilder:
        """MongoDB-like syntax for $in operator."""
        return QueryBuilder(query={self.field: {"$in": values}})

    def nin_(self, values: List[Value]) -> QueryBuilder:
        """MongoDB-like syntax for $nin operator."""
        return QueryBuilder(query={self.field: {"$nin": values}})


class UpsertRequest(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    values: Vector = Field(...)
    metadata: MetaData = Field(...)


class Embedding(BaseModel):
    values: Vector = Field(...)
    metadata: MetaData = Field(...)


class QueryRequest(BaseModel):
    topK: int = Field(default=10)
    filter: Dict[str, Any] = Field(...)
    includeMetadata: bool = Field(default=True)
    vector: Vector = Field(...)


class QueryMatch(BaseModel):
    id: str = Field(...)
    score: float = Field(...)
    metadata: MetaData = Field(...)


class QueryResponse(BaseModel):
    matches: List[QueryMatch] = Field(...)


class UpsertResponse(BaseModel):
    upsertedCount: int = Field(...)


class VectorClient(LazyProxy[AsyncClient]):
    api_key: str = environ["PINECONE_API_KEY"]
    api_endpoint: str = environ["PINECONE_API_URL"]

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.__set_proxied__(self.__load__())

    def __load__(self):
        return AsyncClient(
            headers={"api-key": self.api_key}, base_url=self.api_endpoint
        )

    @handle
    async def upsert(
        self, vectors: List[Vector], metadata: List[MetaData]
    ) -> UpsertResponse:
        async with self.__load__() as session:
            payload = {
                "vectors": [
                    UpsertRequest(values=vector, metadata=meta).dict()
                    for vector, meta in zip(vectors, metadata)
                ]
            }
            response = await session.post("/vectors/upsert", json=payload)
            print(response.text)
            return UpsertResponse(**response.json())

    @handle
    async def query(
        self, expr: Query, vector: Vector, includeMetadata: bool = True, topK: int = 5
    ) -> List[str]:
        async with self.__load__() as session:
            payload = QueryRequest(
                filter=expr, vector=vector, includeMetadata=includeMetadata, topK=topK  # type: ignore
            ).dict()
            response = await session.post("/query", json=payload)
            print(response.json())
            data = QueryResponse(**response.json())
            return [
                f"{match.metadata['text']}: Score {match.score}"
                for match in data.matches
            ]
