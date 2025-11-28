from pydantic import BaseModel

class ModelPredictionRequest(BaseModel):
    bath: float | None = None
    prev_sold_date: str | None = None
    state: str | None = None
    status: str | None = None
    city: str | None = None
    zip_code: str | None = None
    house_size: float | None = None
    bed: int | None = None
    brokered_by: str | None = None
    street: str | None = None
    acre_lot: float | None = None


NORMALIZED_COLUMNS = []
