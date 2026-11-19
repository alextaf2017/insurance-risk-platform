"""Contratti di ingresso e uscita dell'API."""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "driv_age": 35, "veh_age": 4, "bonus_malus": 60, "veh_power": 6, "density": 1200.0,
        "veh_gas": "Regular", "veh_brand": "B12", "region": "R24", "exposure": 1.0}})
    driv_age: int = Field(ge=18, le=110, description="Eta' del conducente")
    veh_age: int = Field(ge=0, le=100, description="Eta' del veicolo")
    bonus_malus: int = Field(ge=50, le=350, description="Bonus-Malus: obbligatorio, e' la variabile piu' critica")
    veh_power: int = Field(ge=1, le=30)
    density: float = Field(gt=0, description="Abitanti per km2 del comune")
    veh_gas: str = Field(min_length=1, max_length=20)
    veh_brand: str = Field(min_length=1, max_length=10)
    region: str = Field(min_length=1, max_length=10)
    exposure: float = Field(default=1.0, gt=0, le=1, description="Anni-polizza della copertura")


class PolicyBatch(BaseModel):
    policies: list[Policy] = Field(min_length=1, max_length=1000)


class PremiumOut(BaseModel):
    request_id: str
    expected_frequency: float = Field(description="Sinistri pagati attesi per anno-polizza")
    expected_claims: float = Field(description="Sinistri attesi sull'esposizione richiesta")
    severity_capped_mean: float
    tail_load: float
    pure_premium_per_year: float
    pure_premium: float = Field(description="Premio puro sull'esposizione richiesta")
    warnings: list[str]
    model_version: str


class BatchOut(BaseModel):
    request_id: str
    results: list[PremiumOut]


class TriageIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=20000, description="Narrativa del reclamo")


class TriageOut(BaseModel):
    request_id: str
    stage1_score: float
    is_candidate: bool
    stage2_score: Optional[float] = None
    score: Optional[float] = Field(default=None, description="Punteggio finale (solo per i candidati)")
    review: Optional[bool] = Field(default=None, description="True se supera la soglia di revisione (5% dei reclami)")
    route: str = Field(description="not_candidate | cascade | stage1_only")
    n_words: int
    warnings: list[str]
    model_version: str
