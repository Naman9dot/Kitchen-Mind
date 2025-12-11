from __future__ import annotations
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Ingredient:
    name: str
    quantity: float
    unit: str

    def scaled(self, factor: float) -> "Ingredient":
        return Ingredient(name=self.name, quantity=round(self.quantity * factor, 3), unit=self.unit)

@dataclass
class Recipe:
    id: str
    title: str
    ingredients: List[Ingredient]
    steps: List[str]
    servings: int  # baseline servings
    metadata: Dict[str, Any] = field(default_factory=dict)
    ratings: List[float] = field(default_factory=list)
    validator_confidence: float = 0.0
    popularity: int = 0
    approved: bool = False

    def scale(self, target_servings: int) -> "Recipe":
        if self.servings <= 0:
            raise ValueError("Recipe baseline servings must be > 0")
        factor = target_servings / self.servings
        scaled_ings = [ing.scaled(factor) for ing in self.ingredients]
        return Recipe(
            id=self.id,
            title=self.title,
            ingredients=scaled_ings,
            steps=self.steps,
            servings=target_servings,
            metadata={**self.metadata, "scaled_from": self.servings},
            ratings=self.ratings.copy(),
            validator_confidence=self.validator_confidence,
            popularity=self.popularity,
            approved=self.approved,
        )

    def avg_rating(self) -> float:
        return statistics.mean(self.ratings) if self.ratings else 0.0

@dataclass
class User:
    id: str
    username: str
    role: str = "user"  # user, trainer, validator, admin
    rmdt_balance: float = 0.0

    def credit(self, amount: float):
        self.rmdt_balance += amount

    def debit(self, amount: float):
        if amount > self.rmdt_balance:
            raise ValueError("Insufficient RMDT balance")
        self.rmdt_balance -= amount
