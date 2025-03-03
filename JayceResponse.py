from pydantic import BaseModel, Field

class JayceResponse(BaseModel):
    response: str = Field(..., description="Tu respuesta al jugador")
    book_remembered: bool = Field(..., description="Si el jugador ha dicho el nombre del libro, debe ser True, si no, False")