from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field

class Edital(BaseModel):
    titulo_completo: str = Field(..., description='título principal completo do edital/chamada')
    titulos: List[str] = Field(..., description='Os título do edital/chamada')
    numero: None | str = Field(..., description='número do edital')
    objetivo: str = Field(..., description='objetivo completo do edital')
    elegibilidade: List[str] = Field(..., description='critérios de elegibilidade do edital')
    submissao: str = Field(..., description='data de submissao do edital')
    financiamento: str = Field(..., description='valor do financiamento do edital')
    areas: List[str] = Field(..., description='areas de conhecimento do edital', min_items=1)
    nivel_trl: None | str = Field(..., description='nivel de maturidade tecnológica (TRLs)')