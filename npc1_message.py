from pydantic import BaseModel, Field

class NPC1Message(BaseModel):
    message: str = Field(..., description="La respuesta del NPC1")
    share_fragment: bool = Field(..., description="Si el NPC1 debe compartir un nuevo fragmento")
