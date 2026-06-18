from pydantic import BaseModel, Field
from typing import Any


class APIResponse(BaseModel):
    success: bool = True
    message: str = ""
    data: dict[str, Any] | list | None = None
    errors: list[str] | None = None
