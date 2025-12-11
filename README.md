# KitchenMind - Modular Architecture

This project has been transformed from a single-file implementation into a clean, modular package structure.

## Project Structure

```
Kitchen Mind/
├── Module/                    # Main package directory
│   ├── __init__.py           # Package initialization and exports
│   ├── models.py             # Data models (Ingredient, Recipe, User)
│   ├── repository.py         # Recipe storage and retrieval
│   ├── vector_store.py       # Semantic search functionality
│   ├── scoring.py            # Recipe ranking and scoring
│   ├── synthesizer.py        # Recipe synthesis and merging
│   ├── token_economy.py      # RMDT token rewards system
│   ├── event_planner.py      # Event planning functionality
│   └── controller.py         # Main KitchenMind controller
├── main.py                   # Entry point with example usage
├── one_file.py               # Original single-file version (preserved)
└── README.md                 # This file
```

## Module Descriptions

### `models.py`
Core data classes:
- **Ingredient**: Represents a recipe ingredient with name, quantity, and unit
- **Recipe**: Complete recipe with ingredients, steps, metadata, and ratings
- **User**: User account with role-based permissions and RMDT balance

### `repository.py`
- **RecipeRepository**: In-memory storage for recipes
- Methods for adding, retrieving, and filtering recipes

### `vector_store.py`
- **MockVectorStore**: Simple semantic search implementation
- Uses pseudo-random vectors for similarity matching
- In production, replace with actual embeddings + vector DB

### `scoring.py`
- **ScoringEngine**: Multi-factor recipe ranking system
- Considers user ratings, validator confidence, authenticity, scalability, and popularity

### `synthesizer.py`
- **Synthesizer**: Combines multiple recipes into one optimized recipe
- Merges ingredients intelligently
- Normalizes cooking steps and phases
- Handles ingredient conflicts (e.g., leavening agents)

### `token_economy.py`
- **TokenEconomy**: Manages RMDT token rewards
- Rewards trainers for recipe submissions
- Rewards validators for recipe validation

### `event_planner.py`
- **EventPlanner**: Creates event menus from approved recipes
- Filters by dietary requirements
- Estimates costs and guest counts

### `controller.py`
- **KitchenMind**: Main orchestrator class
- Integrates all modules
- Provides high-level API for:
  - User management
  - Recipe submission and validation
  - Recipe requests and synthesis
  - Rating and feedback
  - Event planning

## Usage

### Basic Example

```python
from Module import KitchenMind

# Initialize system
km = KitchenMind()

# Create users
trainer = km.create_user('alice', role='trainer')
validator = km.create_user('bob', role='validator')
user = km.create_user('charlie', role='user')

# Submit a recipe
recipe = km.submit_recipe(
    trainer,
    title='Idli – Traditional',
    ingredients=[
        {'name': 'Rice', 'quantity': 300, 'unit': 'g'},
        {'name': 'Urad Dal', 'quantity': 100, 'unit': 'g'},
        {'name': 'Water', 'quantity': 350, 'unit': 'ml'},
        {'name': 'Salt', 'quantity': 5, 'unit': 'g'},
    ],
    steps=[
        'Soak rice and urad dal separately for 4 hours.',
        'Grind both into a smooth batter.',
        'Let the batter ferment overnight.',
        'Add salt and steam for 12 minutes.'
    ],
    servings=4
)

# Validate recipe
km.validate_recipe(validator, recipe.id, approved=True, confidence=0.85)

# Request synthesized recipe
result = km.request_recipe(user, 'Idli', servings=5)
```

### Running the Example

```bash
python main.py
```

This will run a complete demonstration including:
- User creation
- Recipe submission
- Validation
- Recipe synthesis
- Event planning
- Token rewards display

## Key Features Preserved

All functionality from the original single file has been preserved:

✅ Role-based user system (user, trainer, validator, admin)  
✅ Recipe submission and validation workflow  
✅ Multi-source recipe synthesis  
✅ Semantic search for recipes  
✅ Weighted scoring system  
✅ Token-based reward economy  
✅ Event planning with menu suggestions  
✅ Ingredient normalization and conflict resolution  

## Benefits of Modular Structure

1. **Maintainability**: Each module has a single, clear responsibility
2. **Testability**: Modules can be tested independently
3. **Reusability**: Components can be imported and used separately
4. **Scalability**: Easy to extend individual modules without affecting others
5. **Readability**: Smaller files are easier to understand and navigate
6. **Collaboration**: Multiple developers can work on different modules simultaneously

## Migration Notes

- The original `one_file.py` has been preserved for reference
- All imports use relative imports (`.models`, `.repository`, etc.)
- The package can be imported as: `from Module import KitchenMind`
- All debug logging from the original code has been preserved
- The simplified synthesizer maintains API compatibility

## Future Enhancements

- Replace MockVectorStore with actual vector database (Pinecone, Weaviate, etc.)
- Add persistent storage (PostgreSQL, MongoDB)
- Implement full LLM integration for synthesis
- Add API endpoints (FastAPI/Flask)
- Create web interface
- Add unit tests for all modules
- Implement caching layer
- Add logging configuration

## License

[Your License Here]

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style
- Each module maintains single responsibility
- Tests are added for new functionality
- Documentation is updated
