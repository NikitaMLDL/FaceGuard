from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    embedding: list  # Эмбеддинг будет передан как список чисел