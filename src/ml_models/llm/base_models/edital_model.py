from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field

class Edital(BaseModel):
    titulo_completo: str = Field(..., description='título principal completo do edital/chamada')
    titulos: List[str] = Field(..., description='Os título do edital/chamada')
    numero: None | str = Field(..., description='número do edital')
    objetivo: str = Field(..., description='objetivo completo do edital')
    elegibilidade: List[str] = Field(..., description='critérios de elegibilidade do edital')
    inicio: None | str = Field(..., description='data de inicio do edital')
    submissao: str = Field(..., description='data de submissao do edital')
    financiamento: str = Field(..., description='valor do financiamento do edital')
    areas: List[str] = Field(..., description='areas de conhecimento da chamada, podendo ser uma ou mais áreas')
    nivel_trl: None | str = Field(..., description='nivel de maturidade tecnológica (TRLs)')

class EditalGeneralInfo(BaseModel):
    titulo_completo: str = Field(..., description='título principal completo do edital/chamada')
    titulos: List[str] = Field(..., description='Os título do edital/chamada')
    numero: None | str = Field(..., description='numero do edital')
    objetivo: str = Field(..., description='o objetivo/objeto completo do edital')
    nivel_trl: None | str = Field(..., description='nivel de maturidade tecnológica (TRLs)')

class EditalValuesInfo(BaseModel):
    date_lanc: str = Field(..., description='data de lançamento do edital')
    date_sub: str = Field(..., description='data de submissao do edital')
    financiamento: str = Field(..., description='valor do financiamento do edital')

class EditalDomainRequirements(BaseModel):
    areas: List[str] = Field(..., description='areas de conhecimento da chamada podendo ser uma ou mais áreas', enum = ['Ciências Naturais', 'Ciências Sociais', 'Humanidades', 'Tecnologia da Informação', 'Matemática e Estatística', 'Saúde e Medicina', 'Educação', 'Negócios e Economia', 'Meio Ambiente e Sustentabilidade', 'Engenharia e Tecnologia'], min_items=1)
    elegibilidade: List[str] = Field(..., description='criterios de elegibilidade do edital')