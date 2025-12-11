"""
Recipe repository for the KitchenMind system.
Simple in-memory storage. In production, replace with persistent DB.
"""

from typing import Dict, Optional, List
from .models import Recipe


class RecipeRepository:
    """Simple in-memory repository. In production, replace with persistent DB."""
    def __init__(self):
        self.recipes: Dict[str, Recipe] = {}

    def add(self, recipe: Recipe):
        self.recipes[recipe.id] = recipe

    def get(self, recipe_id: str) -> Optional[Recipe]:
        return self.recipes.get(recipe_id)

    def find_by_title(self, title: str) -> List[Recipe]:
        s = title.lower()
        return [r for r in self.recipes.values() if s in r.title.lower()]

    def pending(self) -> List[Recipe]:
        return [r for r in self.recipes.values() if not r.approved]

    def approved(self) -> List[Recipe]:
        return [r for r in self.recipes.values() if r.approved]