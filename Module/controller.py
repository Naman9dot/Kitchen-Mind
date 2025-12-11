"""
Main controller for the KitchenMind system.
Orchestrates all components: recipes, users, synthesis, validation, etc.
"""

import uuid
from typing import Dict, List, Optional
from .models import User, Recipe, Ingredient
from .repository import RecipeRepository
from .vector_store import MockVectorStore
from .scoring import ScoringEngine
from .synthesizer import Synthesizer
from .token_economy import TokenEconomy
from .event_planner import EventPlanner


class KitchenMind:
    def __init__(self):
        self.recipes = RecipeRepository()
        self.vstore = MockVectorStore()
        self.scorer = ScoringEngine()
        self.synth = Synthesizer()
        self.tokens = TokenEconomy()
        self.users: Dict[str, User] = {}

    def create_user(self, username: str, role: str = 'user') -> User:
        user = User(id=str(uuid.uuid4()), username=username, role=role)
        self.users[user.id] = user
        return user

    def submit_recipe(self, trainer: User, title: str, ingredients: List[Dict], steps: List[str], servings: int) -> Recipe:
        assert trainer.role in ('trainer','admin'), 'Only trainers or admins can submit recipes.'
        recipe = Recipe(
            id=str(uuid.uuid4()),
            title=title,
            ingredients=[Ingredient(**ing) for ing in ingredients],
            steps=steps,
            servings=servings,
            metadata={'submitted_by': trainer.username}
        )
        self.recipes.add(recipe)
        self.vstore.index(recipe)
        self.tokens.reward_trainer_submission(trainer, amount=1.0)
        return recipe

    def validate_recipe(self, validator: User, recipe_id: str, approved: bool, feedback: Optional[str] = None, confidence: float = 0.8):
        assert validator.role in ('validator','admin'), 'Only validators or admins can validate.'
        r = self.recipes.get(recipe_id)
        if r is None:
            raise KeyError('Recipe not found')
        r.ingredients = self.synth.normalize_leavening(r.ingredients)
        r.approved = approved
        r.metadata['validation_feedback'] = feedback
        r.validator_confidence = max(0.0, min(1.0, confidence))
        if approved:
            r.popularity += 1
            self.vstore.index(r)
        self.tokens.reward_validator(validator, amount=0.5)
        return r

    def request_recipe(self, user: User, dish_name: str, servings: int = 2, top_k: int = 10, reorder: bool = True) -> Recipe:
        # prefer explicit title matches first (safer)
        direct = [r for r in self.recipes.find_by_title(dish_name) if r.approved]
        candidates = []
        if direct:
            candidates = direct
        else:
            text = f"{dish_name} for {servings}"
            results = self.vstore.query(text, top_k=top_k)
            candidate_ids = [rid for rid,_ in results]
            candidates = [self.recipes.get(rid) for rid in candidate_ids if self.recipes.get(rid) and self.recipes.get(rid).approved]

        if not candidates:
            raise LookupError('No approved recipes found for this dish')

        # if some candidates contain the dish name in title, prefer those
        named = [r for r in candidates if dish_name.lower() in r.title.lower()]
        if named:
            candidates = named

        scored = [(r, self.scorer.score(r)) for r in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_n = [r for r,_ in scored[:2]]
        synthesized = self.synth.synthesize(top_n, servings, reorder=reorder)
        self.recipes.add(synthesized)
        self.vstore.index(synthesized)
        return synthesized


    def rate_recipe(self, user: User, recipe_id: str, rating: float):
        r = self.recipes.get(recipe_id)
        if not r:
            raise KeyError('Recipe not found')
        r.ratings.append(max(0.0, min(5.0, rating)))
        r.popularity += 1
        return r

    def list_pending(self) -> List[Recipe]:
        return self.recipes.pending()

    def event_plan(self, event_name: str, guest_count: int, budget_per_person: float, dietary: Optional[str] = None):
        planner = EventPlanner(self.recipes)
        return planner.plan_event(event_name, guest_count, budget_per_person, dietary)
