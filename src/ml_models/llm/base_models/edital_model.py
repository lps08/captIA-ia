from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field

class Edital(BaseModel):
    titulo: str = Field(..., description='titulo do edital')
    objetivo: str = Field(..., description='objetivo completo do edital')
    elegibilidade: List[str] = Field(..., description='critérios de elegibilidade do edital')
    submissao: str = Field(..., description='data de submissao do edital')
    financiamento: str = Field(..., description='valor do financiamento do edital')
    areas: List[str] = Field(..., description='areas do edital', min_items=1)
    bolsa: bool = Field(..., description='se o financiamento é bolsa', enum=[True, False])