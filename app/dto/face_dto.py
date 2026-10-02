from pydantic import BaseModel, Field
from fastapi import UploadFile

class StaffRegisterDTO(BaseModel):
    staff_id: str = Field(..., max_length=50)
    staff_name: str = Field(..., max_length=100)
    # Image comes as UploadFile in controller, so not part of DTO here

class StaffVerifyDTO(BaseModel):
    # Image as UploadFile, handle separately
    pass
