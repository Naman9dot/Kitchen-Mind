#!/usr/bin/env python3
# kitchenmind_single.py
"""
Single-file runnable version of your KitchenMind system.
Contains: models, repository, vector store, scoring, synthesizer, token economy,
event planner, controller (KitchenMind) and example_run().

Run:
    python kitchenmind_single.py
"""

from __future__ import annotations
import re
import uuid
import random
import math
import statistics
import pprint
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

# Try to import torch early for environment check (optional)
try:
    import torch
except Exception:
    torch = None

# ----------------------------- Models -----------------------------
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

# ----------------------------- Repository -----------------------------
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

# ----------------------------- Vector Store (Mock) -----------------------------
class MockVectorStore:
    """A toy semantic index. Use actual embeddings + vector DB in prod."""
    def __init__(self):
        # store mapping id -> "embedding" (here a random vector) and metadata
        self.vectors: Dict[str, List[float]] = {}

    def index(self, recipe: Recipe):
        # naive: create a deterministic pseudo-random vector from recipe title
        r = abs(hash(recipe.title)) % (10**8)
        random.seed(r)
        vec = [random.random() for _ in range(64)]
        self.vectors[recipe.id] = vec

    def query(self, text: str, top_k=10) -> List[Tuple[str, float]]:
        # return ids with 'distance' (lower = more similar)
        qh = abs(hash(text)) % (10**8)
        random.seed(qh)
        qvec = [random.random() for _ in range(64)]
        def sim(a,b):
            # cosine similarity
            num = sum(x*y for x,y in zip(a,b))
            lena = math.sqrt(sum(x*x for x in a))
            lenb = math.sqrt(sum(x*x for x in b))
            return num/(lena*lenb+1e-9)
        scores = [(rid, sim(qvec, vec)) for rid,vec in self.vectors.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

# ----------------------------- Scoring Engine -----------------------------
class ScoringEngine:
    """Implements the weighted scoring used to pick top recipes."""
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # default weights (must sum to 1 ideally)
        self.weights = weights or {
            'user_rating': 0.30,
            'validator_confidence': 0.20,
            'ingredient_authenticity': 0.15,
            'serving_scalability': 0.15,
            'popularity': 0.10,
            'ai_confidence': 0.10,
        }

    def ingredient_authenticity_score(self, recipe: Recipe) -> float:
        # mock heuristic: penalize unusual units or missing quantities
        score = 1.0
        for ing in recipe.ingredients:
            if not ing.unit or ing.quantity <= 0:
                score -= 0.2
        return max(0.0, score)

    def serving_scalability_score(self, recipe: Recipe) -> float:
        # mock: recipes with reasonable serving numbers get higher score
        if 1 <= recipe.servings <= 12:
            return 1.0
        elif recipe.servings <= 50:
            return 0.8
        else:
            return 0.5

    def popularity_score(self, recipe: Recipe) -> float:
        # popularity normalized (0..1) assuming max popularity ~1000
        return min(1.0, recipe.popularity / 1000.0)

    def ai_confidence_score(self, recipe: Recipe) -> float:
        # placeholder: read from metadata
        return recipe.metadata.get('ai_confidence', 0.5)

    def normalize(self, x: float, max_val: float = 5.0) -> float:
        return max(0.0, min(1.0, x / max_val))

    def score(self, recipe: Recipe) -> float:
        parts = {}
        parts['user_rating'] = self.normalize(recipe.avg_rating(), max_val=5.0)
        parts['validator_confidence'] = recipe.validator_confidence
        parts['ingredient_authenticity'] = self.ingredient_authenticity_score(recipe)
        parts['serving_scalability'] = self.serving_scalability_score(recipe)
        parts['popularity'] = self.popularity_score(recipe)
        parts['ai_confidence'] = self.ai_confidence_score(recipe)

        total = sum(self.weights[k] * parts[k] for k in parts)
        return total

# ----------------------------- Synthesizer -----------------------------
class Synthesizer:
    CANONICAL_NAMES = {
        'curd': 'yogurt',
        'dahi': 'yogurt',
        'yoghurt': 'yogurt',
        'yogurt': 'yogurt',
    }

    PHASE_KEYWORDS = {
        'prep': ['chop', 'slice', 'dice', 'peel', 'grate', 'measure', 'prepare', 'trim', 'wash', 'soak'],
        'mix': ['mix', 'whisk', 'combine', 'stir', 'fold', 'beat', 'blend', 'whip'],
        'rest': ['rest', 'let sit', 'prove', 'proof', 'stand', 'marinate'],
        'cook': ['steam', 'bake', 'fry', 'saute', 'simmer', 'cook', 'boil', 'roast', 'grill', 'heat', 'pressure'],
        'finish': ['garnish', 'serve', 'drizzle', 'sprinkle', 'plate']
    }

    BATTER_KEYWORDS = [
        r"\bwhisk\b", r"\bmix\b", r"\bstir\b", r"\bcombine\b", r"\bfold\b",
        r"\badd\b", r"\bblend\b", r"\bgrind\b", r"\bmake.*batter\b",
    ]

    BATTER_INGREDIENT_HINTS = [
        "flour", "atta", "maida", "besan", "gram flour", "rice flour",
        "yogurt", "curd", "buttermilk", "milk", "water", "eggs",
        "semolina", "suji", "cornflour",
    ]

    LEAVENING_HINTS = [
        "eno", "baking soda", "baking powder", "yeast",
    ]

    COOKING_FINALIZATION_HINTS = [
        "steam", "fry", "bake", "rest", "ferment",
    ]

    @staticmethod
    def _normalize_step_text(s: str) -> str:
        return ' '.join(s.strip().split())

    @classmethod
    def canonical_name(cls, name: str) -> str:
        k = name.strip().lower()
        if k.endswith('s') and k[:-1] in cls.CANONICAL_NAMES:
            k = k[:-1]
        return cls.CANONICAL_NAMES.get(k, name.strip())

    @staticmethod
    def is_batter_step(step: str) -> bool:
        s = step.lower()
        if "batter" in s:
            return True
        if any(k in s for k in Synthesizer.BATTER_INGREDIENT_HINTS):
            if any(v in s for v in ["mix", "combine", "whisk", "blend", "stir", "make"]):
                return True
        if any(re.search(k, s) for k in Synthesizer.BATTER_KEYWORDS):
            return True
        return False

    @staticmethod
    def normalize_batter_steps(steps: List[str]) -> List[str]:
        batter_steps = [s for s in steps if Synthesizer.is_batter_step(s)]
        if not batter_steps:
            return steps
        combined = " ".join(batter_steps).lower()
        output = []
        if any(f in combined for f in ["flour", "gram", "rice", "maida", "semolina", "suji"]):
            output.append("Whisk the flour and liquids together, adding water gradually to form a smooth batter.")
        if any(k in combined for k in Synthesizer.LEAVENING_HINTS):
            output.append("Add the leavening agent (Eno, baking soda, or similar).")
        if "sugar" in combined or "salt" in combined or "spice" in combined:
            output.append("Add sugar, salt, and spices as required.")
        output.append("Mix gently until just combined.")
        final = []
        if any(k in combined for k in ["steam"]):
            final.append("Steam for 15 minutes.")
        elif any(k in combined for k in ["fry"]):
            final.append("Fry until golden.")
        elif any(k in combined for k in ["bake"]):
            final.append("Bake as required.")
        elif any(k in combined for k in ["rest", "ferment"]):
            final.append("Allow the batter to rest or ferment as required.")
        output.extend(final)
        return output

    def _ingredient_tokens(self, name: str) -> List[str]:
        """
        Normalize an ingredient name into tokens we can look for in step text.
        E.g., "Gram Flour" -> ['gram', 'flour']
        """
        s = re.sub(r'[^a-z\s]', ' ', name.lower())
        toks = [t for t in s.split() if len(t) > 1]
        return toks

    def ensure_ingredient_coverage(self, out_lines: List[str], merged_ings: List[Ingredient]) -> List[str]:
        """
        Improved ensure_ingredient_coverage.

        Behavior:
        - Build tokens for merged ingredients.
        - Detect which ingredients are *missing* from out_lines.
        - Prefer removing short "add|mix|combine" lines that mention *missing* ingredients (NOT any merged ingredient).
        - Never remove lines classified as cook/rest/finish or that contain time/temp.
        - Insert the combined step at the earliest removed index, or before the first cook/rest/finish/time line,
          or append if neither exists.
        - Debug prints included to trace decisions.
        """
        if not merged_ings:
            print("DEBUG: no merged_ings -> returning unchanged out_lines")
            return out_lines

        print("DEBUG: START ensure_ingredient_coverage")
        all_text = " ".join(out_lines).lower()
        missing = []
        toks_by_name = {}
        toks_all_by_name = {}

        # Tokenize all merged ingredients, and mark which are missing
        for ing in merged_ings:
            name = ing.name.strip()
            toks = self._ingredient_tokens(name)
            if not toks:
                print(f"DEBUG: ingredient {repr(name)} produced no tokens; skipping")
                continue
            toks_all_by_name[name] = toks
            present = any(tok in all_text for tok in toks)
            if not present:
                missing.append((name, ing.unit.strip().lower()))
                toks_by_name[name] = toks

        # Debug: show tokenization for all merged ingredients
        print("DEBUG: merged ingredient tokens:")
        for n, toks in toks_all_by_name.items():
            print("  -", repr(n), "->", toks)
        if missing:
            print("DEBUG: missing ingredients detected:")
            for name, unit in missing:
                print("  -", repr(name), "unit=", repr(unit), "tokens=", toks_by_name.get(name))
        else:
            print("DEBUG: no missing ingredients -> returning unchanged out_lines")
            return out_lines

        # Identify candidate lines to remove — prefer short add/mix/combine lines,
        # but DO NOT remove cook/rest/finish/time lines.
        indices_to_remove = set()
        add_line_pattern = re.compile(r'^\s*(add|mix|combine)\b.*$', flags=re.I)

        protected_indices = set()
        for i, s in enumerate(out_lines):
            if self.classify_phase(s) in ('cook', 'rest', 'finish') or self.has_time_or_temp(s):
                protected_indices.add(i)

        print("DEBUG: protected_indices (cook/rest/finish/time):", protected_indices)

        for i, s in enumerate(out_lines):
            if i in protected_indices:
                # never remove protected lines
                continue
            low = s.lower()
            # Prefer removing short add/mix/combine lines that mention any *missing* token
            if add_line_pattern.match(s):
                for toks in toks_by_name.values():  # NOTE: only missing ingredient tokens
                    if any(re.search(r'\b' + re.escape(tok) + r'\b', low) for tok in toks):
                        # Safety: don't remove if the line itself contains time/temp or is cook/rest/finish (extra guard)
                        if self.classify_phase(s) in ('cook', 'rest', 'finish') or self.has_time_or_temp(s):
                            break
                        indices_to_remove.add(i)
                        break
            else:
                # Consider removing only if it's short-ish and mentions missing tokens
                for toks in toks_by_name.values():
                    if any(re.search(r'\b' + re.escape(tok) + r'\b', low) for tok in toks):
                        # Only remove short lines (<= 6 words) to avoid chopping descriptive instructions
                        if len(low.split()) <= 6:
                            indices_to_remove.add(i)
                        break

        print("DEBUG: candidate indices_to_remove before protection check:", indices_to_remove)
        # Ensure we never remove protected indices accidentally
        indices_to_remove = {i for i in indices_to_remove if i not in protected_indices}
        print("DEBUG: final indices_to_remove (after excluding protected):", indices_to_remove)

        # Debug: list current out_lines
        print("DEBUG: current out_lines:")
        for i, line in enumerate(out_lines):
            print(f"  [{i}] {repr(line)}")

        # Determine insertion index:
        insert_idx = None
        if indices_to_remove:
            insert_idx = min(indices_to_remove)
            print("DEBUG: will insert at earliest removed index:", insert_idx)
        else:
            # insert before the first protected cook/rest/finish line if present
            for i, s in enumerate(out_lines):
                if i in protected_indices:
                    insert_idx = i
                    print("DEBUG: no removals; will insert before first protected index:", insert_idx)
                    break
            if insert_idx is None:
                print("DEBUG: no removals and no protected index -> will append combined step")

        # Remove the selected indices (descending order to keep indices valid)
        for idx in sorted(indices_to_remove, reverse=True):
            try:
                popped = out_lines.pop(idx)
                print(f"DEBUG: popped out_lines[{idx}] = {repr(popped)}")
            except Exception as exc:
                print(f"DEBUG: failed to pop index {idx}: {exc}")

        print("DEBUG: out_lines AFTER removal:")
        for i, line in enumerate(out_lines):
            print(f"  [{i}] {repr(line)}")

        # classify missing into dry vs wet
        liquid_keys = {'water', 'milk', 'buttermilk', 'yogurt', 'oil', 'olive oil', 'lemon juice', 'juice'}
        liquid_units = {'ml', 'l', 'litre', 'liter', 'cup', 'cups', 'tbsp', 'tsp'}
        dry = []
        wet = []
        for name, unit in missing:
            nlow = name.lower()
            if any(k in nlow for k in liquid_keys) or (unit in liquid_units or unit in {'ml', 'l'}):
                wet.append(name)
            else:
                if unit in {'pc', 'pcs', 'piece'} and any(k in nlow for k in ('egg', 'eggs', 'egg')):
                    wet.append(name)
                else:
                    dry.append(name)

        def short_label(full_name: str, keep_warm_for_liquid: bool = True) -> str:
            s = full_name.strip()
            s = re.sub(r'[\(\)\[\]\,]', ' ', s)
            s = s.replace('-', ' ')
            s = re.sub(r'\s+', ' ', s).strip()
            if not s:
                return full_name.title()
            parts = s.split()
            if keep_warm_for_liquid and any(k in s.lower() for k in ('water', 'milk', 'buttermilk', 'yogurt', 'oil', 'juice')):
                if len(parts) == 1:
                    return parts[0].title()
                return " ".join(parts[-2:]).title()
            if parts[-1].lower() in {'flour', 'sugar', 'water', 'yeast', 'salt', 'oil', 'milk', 'yogurt', 'semolina'}:
                return parts[-1].title()
            if len(parts) == 1:
                return parts[0].title()
            return " ".join(parts[-2:]).title()

        disp_dry = [short_label(n, keep_warm_for_liquid=False) for n in dry]
        disp_wet = [short_label(n, keep_warm_for_liquid=True) for n in wet]

        # Build combined instruction
        if disp_dry and disp_wet:
            dry_txt = ", ".join(disp_dry[:-1]) + " and " + disp_dry[-1] if len(disp_dry) > 1 else disp_dry[0]
            wet_txt = ", ".join(disp_wet[:-1]) + " and " + disp_wet[-1] if len(disp_wet) > 1 else disp_wet[0]
            add_step = f"Combine {dry_txt}. Then add {wet_txt} and mix until just combined."
        elif disp_dry:
            dry_txt = ", ".join(disp_dry[:-1]) + " and " + disp_dry[-1] if len(disp_dry) > 1 else disp_dry[0]
            add_step = f"Combine {dry_txt} and mix as required."
        else:
            wet_txt = ", ".join(disp_wet[:-1]) + " and " + disp_wet[-1] if len(disp_wet) > 1 else disp_wet[0]
            add_step = f"Add {wet_txt} and mix until just combined."

        print("DEBUG: disp_dry =", disp_dry)
        print("DEBUG: disp_wet =", disp_wet)
        print("DEBUG: built add_step =", repr(add_step))

        # Insert combined step at determined index (or append)
        if insert_idx is None or insert_idx > len(out_lines):
            out_lines.append(add_step)
            print("DEBUG: appended combined step at end")
        else:
            out_lines.insert(insert_idx, add_step)
            print("DEBUG: inserted combined step at index", insert_idx)

        print("DEBUG: FINAL out_lines from ensure_ingredient_coverage:")
        for i, line in enumerate(out_lines):
            print(f"  [{i}] {repr(line)}")

        print("DEBUG: END ensure_ingredient_coverage")
        return out_lines




    def _collapse_repeated_words(self, s: str) -> str:
            # replace "Whisk the Whisk" -> "Whisk the"
            return re.sub(r'\b(\w+)(?: \1\b)+', r'\1', s, flags=re.I)

    def _normalize_for_dedupe(self, s: str) -> str:
        """
        Create an order-insensitive fingerprint for a step:
        - lower, remove punctuation and numbers
        - remove common stopwords, action verbs, filler words, and unit words
        - keep likely nouns/ingredient tokens (e.g., flour, yogurt, water, eno, steam)
        - return sorted unique tokens joined by space (so "whisk flour yogurt" and
          "whisk mix gram flour and yogurt" collapse to same fingerprint)
        """
        if not s:
            return ""



        s = self._collapse_repeated_words(s)

        s = s.lower()
        # remove punctuation and digits
        s = re.sub(r'[^a-z\s]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        tokens = s.split()

        # words to ignore (stopwords + common cooking actions/fillers + units/measure words)
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'to', 'for', 'of', 'in', 'on', 'with',
            'then', 'so', 'by', 'at', 'from', 'as', 'into', 'until', 'that'
        }
        action_verbs = {
            # common verbs we don't want to rely on for fingerprint
            'mix', 'mixing', 'whisk', 'whisking', 'stir', 'stirring', 'combine', 'combining',
            'add', 'adding', 'fold', 'folding', 'beat', 'beating', 'blend', 'blending',
            'grind', 'grinding', 'soak', 'soaking', 'steam', 'steaming', 'bake', 'baking',
            'fry', 'frying', 'cook', 'cooking', 'heat', 'press', 'pressing', 'serve', 'serving',
            'let', 'allow', 'rest', 'stand', 'proof', 'prove', 'garnish', 'sprinkle', 'drizzle',
            'make', 'making', 'prepare', 'preparing', 'measure', 'measuring', 'adjust', 'adjusting',
            'together', 'together,', 'together.' , 'gently', 'gradually', 'until', 'into', 'form'
        }
        unit_words = {
            'g', 'gram', 'grams', 'kg', 'ml', 'l', 'cup', 'cups', 'tbsp', 'tsp', 'teaspoon', 'tablespoon',
            'pinch', 'piece', 'pieces', 'slice', 'slices', 'small', 'large', 'medium'
        }
        # any other noisy tokens
        noisy = stopwords | action_verbs | unit_words

        # keep tokens that are likely ingredients / important nouns
        kept = []
        for t in tokens:
            if t in noisy:
                continue
            # short tokens like 'of' filtered already; skip 1-char tokens
            if len(t) <= 1:
                continue
            # drop obvious adjectives that add noise ('smooth', 'golden', 'fresh') — optional
            if t in {'smooth', 'golden', 'fresh', 'warm', 'hot', 'cold'}:
                continue
            kept.append(t)

        if not kept:
            # fallback: use tokens excluding pure punctuation/stopwords
            kept = [t for t in tokens if t not in stopwords]

        # produce order-insensitive fingerprint: unique sorted tokens
        key_tokens = sorted(set(kept))
        return " ".join(key_tokens)


    def _dedupe_steps(self, steps: List[str]) -> List[str]:
        """Preserve first occurrence; remove later steps that normalize-identically."""
        seen = set()
        out = []
        for s in steps:
            key = self._normalize_for_dedupe(s)
            if not key:
                # keep non-empty originals
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out


    def generate_prep_from_ingredients(self, merged_ings: List[Ingredient]) -> List[str]:
        names = {ing.name.strip().lower(): ing for ing in merged_ings}
        prep_lines: List[str] = []

        rice_keys = {'rice', 'idli rice', 'parboiled rice', 'idli rice (parboiled)'}
        urad_keys = {'urad dal', 'urad', 'black gram', 'black-gram'}

        has_rice = any(k in names for k in rice_keys)
        has_urad = any(k in names for k in urad_keys)

        if has_rice and has_urad:
            prep_lines.append("Soak rice and urad dal separately for 4–6 hours, then drain.")
            prep_lines.append("Grind soaked rice and urad dal to a smooth batter and combine; ferment if required.")
            return prep_lines

        if 'semolina' in names or 'rava' in names:
            prep_lines.append("Mix semolina with yogurt and water to make a batter; let it rest for 10–15 minutes if using semolina.")
            return prep_lines

        flour_aliases = {'gram flour', 'besan', 'maida', 'atta', 'flour'}
        yogurt_aliases = {'yogurt', 'curd', 'dahi', 'yoghurt'}
        has_flour = any(k in names for k in flour_aliases)
        has_yogurt = any(k in names for k in yogurt_aliases)
        if has_flour and has_yogurt:
            prep_lines.append("Whisk the flour and yogurt together, adding water gradually to form a smooth batter.")
            return prep_lines

        return prep_lines

    def merge_semantic_steps(self, steps: List[str]) -> List[str]:
        norm_steps = []
        seen = set()
        for s in steps:
            if not s:
                continue
            s_norm = self._normalize_step_text(s)
            key = s_norm.lower()
            if key and key not in seen:
                seen.add(key)
                norm_steps.append(s_norm)
        if not norm_steps:
            return []
        flour_pattern = r"(gram flour|besan|semolina|suji|maida|atta|rice|[a-z ]+flour)"
        yogurt_pattern = r"(yogurt|curd|dahi|yoghurt)"
        batter_step = None
        for s in norm_steps:
            low = s.lower()
            if any(v in low for v in ["mix", "whisk", "combine", "stir"]):
                if re.search(flour_pattern, low) and re.search(yogurt_pattern, low):
                    m_flour = re.search(flour_pattern, low)
                    m_yog = re.search(yogurt_pattern, low)
                    flour_txt = (m_flour.group(1) if m_flour else "flour").strip()
                    yog_txt = (m_yog.group(1) if m_yog else "yogurt").strip()
                    flour_txt = flour_txt.title()
                    yog_txt = yog_txt.title()
                    batter_step = (
                        f"Whisk the {flour_txt} and {yog_txt} together, adding water gradually to form a smooth batter."
                    )
                    break
        key_add_names = ["water", "eno", "baking soda", "sugar", "salt"]
        seen_add = []
        for s in norm_steps:
            low = s.lower()
            if "add" in low:
                for name in key_add_names:
                    if name in low and name not in seen_add:
                        seen_add.append(name)
        add_step = None
        if seen_add:
            display_parts = []
            for n in seen_add:
                disp = "Eno" if n == "eno" else n
                display_parts.append(disp)
            if len(display_parts) == 1:
                list_txt = display_parts[0]
            else:
                list_txt = ", ".join(display_parts[:-1]) + " and " + display_parts[-1]
            add_step = f"Add {list_txt}. Mix gently until just combined."
        cook_step = None
        for s in norm_steps:
            low = s.lower()
            if "steam" in low:
                m_time = re.search(r"(\d+)\s*(?:mins?|minutes?)", low)
                if m_time:
                    cook_step = f"Steam for {m_time.group(1)} minutes."
                else:
                    cook_step = "Steam until cooked through."
                break
        if not cook_step:
            if any("steam" in s.lower() for s in norm_steps):
                cook_step = "Steam until cooked through."
        merged = []
        if batter_step:
            # skip if an existing normalized step already contains most of the phrase
            low_b = batter_step.lower()
            duplicate_found = any(
                (low_b in s.lower()) or (s.lower() in low_b) or (self._normalize_for_dedupe(s) == self._normalize_for_dedupe(batter_step))
                for s in norm_steps
            )
            if not duplicate_found:
                merged.append(self._normalize_step_text(batter_step))
        if add_step:
            merged.append(self._normalize_step_text(add_step))
        if cook_step:
            merged.append(self._normalize_step_text(cook_step))
        if not merged:
            return norm_steps
        return merged

    def remove_invalid_leavening_from_steps(self, steps: List[str], ingredients: List[Ingredient]) -> List[str]:
        has_eno = any(i.name.lower() == "eno" for i in ingredients)
        has_soda = any(i.name.lower() in ["baking soda", "soda"] for i in ingredients)
        if has_eno and not has_soda:
            cleaned = []
            for s in steps:
                s2 = s
                s2 = re.sub(r'\b(baking soda|soda)\b', '', s2, flags=re.I)
                s2 = re.sub(r'\band\s+and\b', 'and', s2, flags=re.I)
                s2 = re.sub(r'\b(and)\s*(?=[\.,;:])', '', s2, flags=re.I)
                s2 = re.sub(r'\band\s*$', '', s2, flags=re.I)
                s2 = re.sub(r'\s+', ' ', s2).strip()
                if s2:
                    cleaned.append(s2)
            return cleaned
        return steps

    def canonicalize_step_text(self, text: str) -> str:
        out = text
        for alias, canon in self.CANONICAL_NAMES.items():
            pattern = r'\b' + re.escape(alias) + r'\b'
            out = re.sub(pattern, canon.title(), out, flags=re.I)
        return out

    def normalize_leavening(self, ingredients: List[Ingredient]) -> List[Ingredient]:
        has_eno = any(i.name.lower() == "eno" for i in ingredients)
        has_soda = any(i.name.lower() in ["baking soda", "soda"] for i in ingredients)
        if has_eno and has_soda:
            ingredients = [i for i in ingredients if i.name.lower() not in ["baking soda", "soda"]]
        return ingredients

    def merge_ingredients(self, recipes: List[Recipe], requested_servings: int) -> List[Ingredient]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for r in recipes:
            for ing in r.ingredients:
                cname = self.canonical_name(ing.name)
                key = cname.strip().lower()
                if key not in grouped:
                    grouped[key] = {"name": cname.strip(), "per_serving": [], "units": []}
                if r.servings <= 0:
                    raise ValueError("Source recipe has invalid servings")
                grouped[key]["per_serving"].append(ing.quantity / r.servings)
                grouped[key]["units"].append(ing.unit)
        merged: List[Ingredient] = []
        for key, data in grouped.items():
            avg_per_serving = sum(data["per_serving"]) / len(data["per_serving"])
            final_qty = round(avg_per_serving * requested_servings, 3)
            unit = max(set(data["units"]), key=data["units"].count) if data["units"] else ""
            merged.append(Ingredient(name=data["name"].title(), quantity=final_qty, unit=unit))
        merged = self.normalize_leavening(merged)
        return merged

    class FreeOpenLLM:
        """
        Adapter to call a local HuggingFace transformers pipeline for text2text-generation.

        Behavior:
          - If torch & CUDA available: attempt GPU-friendly loads (8-bit + device_map="auto").
          - If that fails: try auto device map with low_cpu_mem_usage/offloading.
          - If no GPU or all fails: create CPU pipeline.
          - If any of above fails, `_pipe` remains None and caller will use fallback logic.
        """
        def __init__(self, model_name: str = 'lmsys/fastchat-t5-3b-v1.0'):
            self.model_name = model_name
            self._pipe = None
            self._init_error = None

            # Print a short environment check
            try:
                import transformers  # quick presence check
                has_transformers = True
            except Exception as e:
                has_transformers = False
                self._init_error = e
                return

            use_cuda = False
            try:
                import torch as _torch
                use_cuda = _torch.cuda.is_available()
            except Exception:
                use_cuda = False

            # Try several strategies in order of preference
            # 1) If CUDA available, try 8-bit + device_map="auto" (requires bitsandbytes)
            if use_cuda:
                try:
                    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
                    # attempt 8-bit (fast, memory efficient) first
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        load_in_8bit=True,       # requires bitsandbytes
                    )
                    self._pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)
                    return
                except Exception as e:
                    # keep trying next strategies
                    self._init_error = e

                # 2) Try device_map="auto" with offloading / low memory usage
                try:
                    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        offload_folder="offload",
                        low_cpu_mem_usage=True,
                    )
                    # pipeline will use device_map placements
                    self._pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device_map="auto")
                    return
                except Exception as e:
                    self._init_error = e

            # 3) CPU fallback — load on CPU (low memory usage)
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map={"": "cpu"}, low_cpu_mem_usage=True)
                self._pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
                return
            except Exception as e:
                self._init_error = e
                self._pipe = None

        def available(self) -> bool:
            return self._pipe is not None

        def generate(self, prompt: str, **gen_kwargs) -> str:
            if not self.available():
                raise RuntimeError(f"LLM pipeline for {self.model_name} is not available. Init error: {getattr(self,'_init_error',None)}")
            out = self._pipe(prompt, **gen_kwargs)
            if isinstance(out, list) and out:
                first = out[0]
                if isinstance(first, dict):
                    # common HF pipeline return shape: [{"generated_text": "..."}]
                    return first.get('generated_text', str(first))
                return str(first)
            return str(out)

    @classmethod
    def classify_phase(cls, step: str) -> str:
        low = step.lower()
        for phase in ['prep', 'mix', 'rest', 'cook', 'finish']:
            keywords = cls.PHASE_KEYWORDS.get(phase, [])
            for kw in keywords:
                if kw in low:
                    return phase
        if re.search(r'\b(min|minute|minutes|hr|hour|°c|°f|degrees|°)\b', low):
            return 'cook'
        return 'mix'

    def reorder_steps(self, steps: List[str]) -> List[str]:
        buckets: Dict[str, List[Tuple[int,str]]] = {'prep': [], 'mix': [], 'rest': [], 'cook': [], 'finish': []}
        for i, s in enumerate(steps):
            phase = self.classify_phase(s)
            buckets.setdefault(phase, []).append((i, s))
        ordered = []
        for phase in ['prep', 'mix', 'rest', 'cook', 'finish']:
            items = sorted(buckets.get(phase, []), key=lambda x: x[0])
            ordered.extend([s for _, s in items])
        return ordered if ordered else steps

    @staticmethod
    def has_time_or_temp(text: str) -> bool:
        return bool(re.search(r'\b(\d+\s?(mins?|minutes?|hrs?|hours?|°\s?[CF]|°C|°F|degrees))\b', text, flags=re.I))

    def compute_ai_confidence(self, num_sources: int, steps: List[str], generated_text: str) -> float:
        base = 0.45
        src_bonus = min(0.25, 0.08 * num_sources)
        step_bonus = min(0.2, 0.02 * len(steps))
        time_bonus = 0.15 if any(self.has_time_or_temp(s) for s in steps) else 0.0
        length_penalty = 0.0
        if len(generated_text.split()) < 30:
            length_penalty = 0.1
        conf = base + src_bonus + step_bonus + time_bonus - length_penalty
        return round(max(0.0, min(0.99, conf)), 3)

    def synthesize(self, top_recipes: List[Recipe], requested_servings: int,
                   llm_model: str = 'lmsys/fastchat-t5-3b-v1.0', reorder: bool = True) -> Recipe:
        if not top_recipes:
            raise ValueError("No recipes provided for synthesis")

        merged_ings = self.merge_ingredients(top_recipes, requested_servings)
        prep_from_ings = self.generate_prep_from_ingredients(merged_ings)

        raw_steps = []
        for r in top_recipes:
            for s in r.steps:
                s_norm = self._normalize_step_text(s)
                s_norm = self.canonicalize_step_text(s_norm)
                raw_steps.append(s_norm)

        raw_steps = prep_from_ings + raw_steps
        src = "\n".join(f"- {s}" for s in raw_steps)

        prompt = (
            f"Combine the following cooking actions into one clear, merged recipe for {requested_servings} servings.\n\n"
            f"Write 4–8 numbered steps. Keep steps short (one sentence each). Do NOT add new ingredients or quantities.\n"
            f"Try to include times/temperatures when they are present in the source actions.\n\n"
            f"Source actions:\n{src}\n\n"
            f"Output (begin with '1. '):\n1. "
        )

        llm = self.FreeOpenLLM(model_name=llm_model)
        if not llm.available():
            fallback_steps = []
            seen = set()
            for s in raw_steps:
                s_clean = re.sub(r'\s+', ' ', s).strip()
                if s_clean.lower() not in seen:
                    seen.add(s_clean.lower())
                    fallback_steps.append(s_clean)
            out_lines = fallback_steps[:6] if fallback_steps else ["Combine ingredients and cook as directed."]
            if reorder:
                out_lines = self.reorder_steps(out_lines)
            out_lines = self.merge_semantic_steps(out_lines)
            out_lines = self.remove_invalid_leavening_from_steps(out_lines, merged_ings)
            # ensure prep lines survive
            if prep_from_ings:
                # prepend in original order
                prep_normed = [self._normalize_step_text(p) for p in prep_from_ings]
                for p in prep_normed[::-1]:
                    out_lines.insert(0, p)
            out_lines = [self._collapse_repeated_words(s).strip() for s in out_lines]
            out_lines = self.ensure_ingredient_coverage(out_lines, merged_ings)
            # finally dedupe aggressively but preserve readable originals
            out_lines = self._dedupe_steps(out_lines)

            generated_text = "\n".join(out_lines)
            ai_conf = self.compute_ai_confidence(len(top_recipes), out_lines, generated_text)
            validator_conf = round(min(1.0, ai_conf * 0.8), 3)
            title_base = top_recipes[0].title.split(':')[0].strip()
            title = f"Synthesized — {title_base} (for {requested_servings} servings)"
            meta = {
                "sources": [r.id for r in top_recipes],
                "ai_confidence": ai_conf,
                "synthesis_method": f"fallback:no-llm"
            }
            return Recipe(
                id=str(uuid.uuid4()),
                title=title,
                ingredients=merged_ings,
                steps=out_lines,
                servings=requested_servings,
                metadata=meta,
                validator_confidence=validator_conf,
                approved=True
            )

        gen_kwargs = {
            "max_new_tokens": 180,
            "do_sample": True,
            "temperature": 0.35,
            "top_p": 0.9,
        }

        generated = llm.generate(prompt, **gen_kwargs)

        pattern = r'^\s*(\d+)\.\s*(.+?)(?=\n\s*\d+\.|\Z)'
        matches = re.findall(pattern, generated, flags=re.S | re.M)

        out_lines = []
        for _, text in matches:
            text = text.strip()
            text = re.sub(r'\s+', ' ', text)
            if not text.endswith(('.', '!', '?')):
                text = text + '.'
            if len(text.split()) >= 3:
                out_lines.append(text)

        if not out_lines:
            gen_clean = re.sub(r'\s+', ' ', generated).strip()
            sentences = re.split(r'(?<=[\.\?\!])\s+', gen_clean)
            short_sentences = [s.strip().rstrip('.') + '.' for s in sentences if len(s.split()) >= 3]
            out_lines = short_sentences[:8]

        if not out_lines:
            raise RuntimeError("Model failed to produce any usable steps.")

        out_lines = [' '.join(s.split()) for s in out_lines]
        out_lines = [self.canonicalize_step_text(s) for s in out_lines]
        if reorder:
            out_lines = self.reorder_steps(out_lines)
        out_lines = self.merge_semantic_steps(out_lines)
        out_lines = self.remove_invalid_leavening_from_steps(out_lines, merged_ings)
        # ensure prep lines survive

        if prep_from_ings:
            # prepend in original order
            prep_normed = [self._normalize_step_text(p) for p in prep_from_ings]
            for p in prep_normed[::-1]:
                out_lines.insert(0, p)

        out_lines = [self._collapse_repeated_words(s).strip() for s in out_lines]
        out_lines = self.ensure_ingredient_coverage(out_lines, merged_ings)
        # finally dedupe aggressively but preserve readable originals
        out_lines = self._dedupe_steps(out_lines)


        generated_text = generated if isinstance(generated, str) else str(generated)
        ai_conf = self.compute_ai_confidence(len(top_recipes), out_lines, generated_text)
        validator_conf = round(min(1.0, ai_conf * 0.8), 3)

        base_title = top_recipes[0].title.split(':')[0].strip()
        title = f"Synthesized — {base_title} (for {requested_servings} servings)"

        meta = {
            "sources": [r.id for r in top_recipes],
            "ai_confidence": ai_conf,
            "synthesis_method": f"llm:{llm_model}"
        }

        merged_ings = self.normalize_leavening(merged_ings)

        return Recipe(
            id=str(uuid.uuid4()),
            title=title,
            ingredients=merged_ings,
            steps=out_lines,
            servings=requested_servings,
            metadata=meta,
            validator_confidence=validator_conf,
            approved=True
        )

# ----------------------------- Token Economy -----------------------------
class TokenEconomy:
    def __init__(self):
        self.ledger: Dict[str, float] = {}

    def reward_trainer_submission(self, trainer: User, amount: float = 1.0):
        trainer.credit(amount)
        self.ledger.setdefault(trainer.id, 0.0)
        self.ledger[trainer.id] += amount

    def reward_validator(self, validator: User, amount: float = 0.5):
        validator.credit(amount)
        self.ledger.setdefault(validator.id, 0.0)
        self.ledger[validator.id] += amount

# ----------------------------- Event Planner -----------------------------
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

# ----------------------------- KitchenMind Controller -----------------------------
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

# ----------------------------- Example Usage -----------------------------
def example_run():
    # show a short GPU/torch check
    try:
        import torch as _torch
        print("Torch:", _torch.__version__, "| CUDA available:", _torch.cuda.is_available())
        if _torch.cuda.is_available():
            try:
                print("CUDA device name:", _torch.cuda.get_device_name(0))
            except Exception:
                pass
    except Exception:
        print("Torch not available (will use CPU fallback).")

    km = KitchenMind()
    # create users
    t = km.create_user('alice_trainer', role='trainer')
    v = km.create_user('bob_validator', role='validator')
    u = km.create_user('charlie_user', role='user')

    # trainer submits two versions of a dish
    r1 = km.submit_recipe(
        t,
        title='Idli – Traditional South Indian Steamed Rice Cakes',
        ingredients=[
            {'name':'Rice', 'quantity':300, 'unit':'g'},
            {'name':'Urad Dal', 'quantity':100, 'unit':'g'},
            {'name':'Water', 'quantity':350, 'unit':'ml'},
            {'name':'Salt', 'quantity':5, 'unit':'g'},
        ],
        steps=['Soak rice and urad dal separately for 4 hours.', 'Grind both into a smooth batter.', 'Let the batter ferment overnight.', 'Add salt and steam for 12 minutes.'],
        servings=4
    )

    r2 = km.submit_recipe(
        t,
        title='Rava Idli – Quick Version',
        ingredients=[
            {'name':'Semolina', 'quantity':200, 'unit':'g'},
            {'name':'Yogurt', 'quantity':150, 'unit':'g'},
            {'name':'Water', 'quantity':120, 'unit':'ml'},
            {'name':'Eno', 'quantity':3, 'unit':'g'},
        ],
        steps=['Mix semolina and yogurt to make a batter.', 'Add water gradually.', 'Add Eno and steam the batter.'],
        servings=3
    )

    r3 = km.submit_recipe(
        t,
        title='Besan Chilla (Savory Gram Flour Pancake)',
        ingredients=[
            {'name':'Gram flour', 'quantity':200, 'unit':'g'},
            {'name':'Water', 'quantity':180, 'unit':'ml'},
            {'name':'Onion', 'quantity':1, 'unit':'pc'},
            {'name':'Green chilli', 'quantity':1, 'unit':'pc'},
            {'name':'Salt', 'quantity':4, 'unit':'g'},
        ],
        steps=[
            'Chop onion and green chilli.',
            'Mix gram flour with water to make a pourable batter.',
            'Season with salt and mix well.',
            'Fry ladlefuls of batter until golden on both sides.'
        ],
        servings=4
    )

    # 2) Plain Pancakes (batter + baking powder; tests leavening present)
    r4 = km.submit_recipe(
        t,
        title='American Pancakes',
        ingredients=[
            {'name':'Flour', 'quantity':200, 'unit':'g'},
            {'name':'Milk', 'quantity':250, 'unit':'ml'},
            {'name':'Egg', 'quantity':1, 'unit':'pc'},
            {'name':'Baking powder', 'quantity':8, 'unit':'g'},
            {'name':'Salt', 'quantity':1, 'unit':'g'},
        ],
        steps=[
            'Whisk flour, baking powder and salt.',
            'Add milk and egg and whisk until smooth batter forms.',
            'Heat a pan and cook pancakes for 2 minutes each side.'
        ],
        servings=3
    )

    # 3) Vegetable Stir-fry (no batter; tests cook-phase detection and time parsing)
    r5 = km.submit_recipe(
        t,
        title='Quick Vegetable Stir-Fry',
        ingredients=[
            {'name':'Carrot', 'quantity':150, 'unit':'g'},
            {'name':'Bell pepper', 'quantity':100, 'unit':'g'},
            {'name':'Soy sauce', 'quantity':15, 'unit':'ml'},
            {'name':'Oil', 'quantity':15, 'unit':'ml'},
        ],
        steps=[
            'Slice the vegetables thinly.',
            'Heat oil in a wok and stir-fry vegetables for 5 minutes.',
            'Add soy sauce and toss for 1 minute and serve.'
        ],
        servings=2
    )

    # 4) Simple Bread (yeast present; tests leavening handling with time/rest)
    r6 = km.submit_recipe(
        t,
        title='Quick Yeast Bread',
        ingredients=[
            {'name':'All-purpose flour', 'quantity':500, 'unit':'g'},
            {'name':'Warm water', 'quantity':320, 'unit':'ml'},
            {'name':'Instant yeast', 'quantity':7, 'unit':'g'},
            {'name':'Salt', 'quantity':8, 'unit':'g'},
        ],
        steps=[
            'Combine flour, yeast and salt.',
            'Add warm water and knead the dough for 10 minutes.',
            'Let the dough rest for 1 hour until doubled.',
            'Bake at 220°C for 25 minutes.'
        ],
        servings=8
    )

    # 5) Omelette (short steps; tests very short generated output)
    r7 = km.submit_recipe(
        t,
        title='Masala Omelette',
        ingredients=[
            {'name':'Eggs', 'quantity':3, 'unit':'pc'},
            {'name':'Onion', 'quantity':30, 'unit':'g'},
            {'name':'Salt', 'quantity':2, 'unit':'g'},
            {'name':'Oil', 'quantity':10, 'unit':'ml'},
        ],
        steps=[
            'Beat eggs with chopped onion and salt.',
            'Heat oil and cook the beaten eggs until set.'
        ],
        servings=1
    )

    # 6) Dosa (rice + urad dal again - tests prep lines for rice+urad)
    r8 = km.submit_recipe(
        t,
        title='Dosa – Crispy South Indian Crepe',
        ingredients=[
            {'name':'Rice', 'quantity':400, 'unit':'g'},
            {'name':'Urad Dal', 'quantity':100, 'unit':'g'},
            {'name':'Salt', 'quantity':5, 'unit':'g'},
            {'name':'Oil', 'quantity':20, 'unit':'ml'},
        ],
        steps=[
            'Soak rice and urad dal for 5 hours.',
            'Grind into a smooth batter and ferment overnight.',
            'Spread batter on a hot pan and cook until crisp.'
        ],
        servings=6
    )

    # 7) Khaman variant (conflicting leavening: Eno + baking soda; tests normalization)
    r9 = km.submit_recipe(
        t,
        title='Khaman (variant with mixed leavening)',
        ingredients=[
            {'name':'Gram flour', 'quantity':220, 'unit':'g'},
            {'name':'Yogurt', 'quantity':120, 'unit':'g'},
            {'name':'Eno', 'quantity':4, 'unit':'g'},
            {'name':'Baking soda', 'quantity':2, 'unit':'g'},
        ],
        steps=[
            'Whisk gram flour and yogurt to make a batter.',
            'Add sugar and salt to taste.',
            'Add Eno and baking soda and steam for 12 minutes.'
        ],
        servings=5
    )

    # 8) Lemon Rice (tests simple mix + garnish)
    r10 = km.submit_recipe(
        t,
        title='Lemon Rice',
        ingredients=[
            {'name':'Cooked rice', 'quantity':400, 'unit':'g'},
            {'name':'Lemon juice', 'quantity':30, 'unit':'ml'},
            {'name':'Mustard seeds', 'quantity':2, 'unit':'g'},
            {'name':'Peanuts', 'quantity':30, 'unit':'g'},
            {'name':'Oil', 'quantity':20, 'unit':'ml'},
        ],
        steps=[
            'Heat oil, add mustard seeds and peanuts until aromatic.',
            'Add cooked rice, lemon juice and mix well.',
            'Garnish with coriander and serve.'
        ],
        servings=4
    )

    # 9) Roti (tests recipes with unit-less ingredients in steps)
    r11 = km.submit_recipe(
        t,
        title='Whole Wheat Roti',
        ingredients=[
            {'name':'Whole wheat flour', 'quantity':300, 'unit':'g'},
            {'name':'Water', 'quantity':150, 'unit':'ml'},
            {'name':'Salt', 'quantity':2, 'unit':'g'},
        ],
        steps=[
            'Mix flour and water to form a soft dough.',
            'Divide and roll into discs, then cook on a hot tawa for 1 minute each side.'
        ],
        servings=6
    )

    # 10) Simple Salad (no cooking; tests reorder and short output)
    r12 = km.submit_recipe(
        t,
        title='Cucumber Tomato Salad',
        ingredients=[
            {'name':'Cucumber', 'quantity':150, 'unit':'g'},
            {'name':'Tomato', 'quantity':150, 'unit':'g'},
            {'name':'Olive oil', 'quantity':10, 'unit':'ml'},
            {'name':'Salt', 'quantity':2, 'unit':'g'},
        ],
        steps=[
            'Chop cucumber and tomato.',
            'Toss with olive oil and salt and serve immediately.'
        ],
        servings=2
    )
    # --------------------------------------------------------------------


        # ---------- VALIDATE ALL RECIPES ----------
    for r in km.recipes.recipes.values():
        if r.metadata.get("submitted_by") == "alice_trainer":
            km.validate_recipe(v, r.id, approved=True, feedback="Auto-approved", confidence=0.85)

    # ---------- Request Synthesis ----------
    try:
        synthesized = km.request_recipe(u, 'Quick Yeast Bread', servings=5)
        print('\n--- Synthesized Recipe (for 5) ---')
        pprint.pprint(asdict(synthesized))
    except Exception as e:
        print("Synthesis failed:", str(e))
        synthesized = None

    # ---------- Event Plan ----------
    plan = km.event_plan('Birthday Party', guest_count=20, budget_per_person=5.0)
    print('\n--- Event Plan ---')
    pprint.pprint(plan)

    # ---------- Balances ----------
    print('\n--- User Balances (RMDT) ---')
    for usr in (t, v, u):
        print(f"{usr.username} ({usr.role}): {usr.rmdt_balance} RMDT")

if __name__ == '__main__':
    example_run()
