"""
Event planning module for creating menus based on recipes.
"""

from typing import Dict, Any, Optional, List
from .repository import RecipeRepository


class EventPlanner:
    def __init__(self, recipe_repo: RecipeRepository):
        self.recipe_repo = recipe_repo

    def plan_event(self, event_name: str, guest_count: int, budget_per_person: float, dietary: Optional[str] = None) -> Dict[str, Any]:
        candidates = self.recipe_repo.approved()
        if dietary:
            candidates = [r for r in candidates if dietary.lower() in r.title.lower()]
        selected = candidates[:5]
        menu = [{'title': r.title, 'serves': r.servings} for r in selected]
        total_cost_est = guest_count * budget_per_person
        return {
            'event': event_name,
            'guests': guest_count,
            'budget': total_cost_est,
            'menu': menu,
            'notes': 'This is a sample plan. Replace with price/availability integrations.'
        }
