"""
KitchenMind - AI-powered recipe management and synthesis system.

This package provides:
- Recipe storage and retrieval
- User management with role-based access
- Recipe synthesis from multiple sources
- Semantic search for recipes
- Token-based reward system
- Event planning functionality
"""

from .models import Ingredient, Recipe, User
from .repository import RecipeRepository
from .vector_store import MockVectorStore
from .scoring import ScoringEngine
from .synthesizer import Synthesizer
from .token_economy import TokenEconomy
from .event_planner import EventPlanner
from .controller import KitchenMind

__version__ = '1.0.0'
__author__ = 'KitchenMind Team'

__all__ = [
    'Ingredient',
    'Recipe',
    'User',
    'RecipeRepository',
    'MockVectorStore',
    'ScoringEngine',
    'Synthesizer',
    'TokenEconomy',
    'EventPlanner',
    'KitchenMind',
]
