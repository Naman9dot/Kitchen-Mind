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
        'mix': ['mix', 'whisk', 'combine', 'stir', 'fold', 'beat', 'blend', 'whip', 'knead'],
        'rest': ['rest', 'let sit', 'prove', 'proof', 'stand', 'marinate'],
        'cook': ['steam', 'bake', 'fry', 'saute', 'simmer', 'cook', 'boil', 'roast', 'grill', 'heat', 'pressure', 'stir-fry', 'stir fry'],
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
        print(f"DEBUG: _normalize_step_text input={repr(s)}")
        out = ' '.join(s.strip().split())
        print(f"DEBUG: _normalize_step_text output={repr(out)}")
        return out

    @classmethod
    def canonical_name(cls, name: str) -> str:
        print(f"DEBUG: canonical_name input={repr(name)}")
        k = name.strip().lower()
        if k.endswith('s') and k[:-1] in cls.CANONICAL_NAMES:
            print(f"DEBUG: canonical_name trimming plural: {k} -> {k[:-1]}")
            k = k[:-1]
        canon = cls.CANONICAL_NAMES.get(k, name.strip())
        result = canon.lower() if isinstance(canon, str) else name.strip().lower()
        print(f"DEBUG: canonical_name output={repr(result)}")
        return result


    @staticmethod
    def is_batter_step(step: str) -> bool:
        print(f"DEBUG: is_batter_step checking: {repr(step)}")
        s = step.lower()
        if "batter" in s:
            print("DEBUG: is_batter_step -> True (found 'batter')")
            return True
        if any(k in s for k in Synthesizer.BATTER_INGREDIENT_HINTS):
            if any(v in s for v in ["mix", "combine", "whisk", "blend", "stir", "make"]):
                print("DEBUG: is_batter_step -> True (ingredient hint + mixing verb found)")
                return True
        if any(re.search(k, s) for k in Synthesizer.BATTER_KEYWORDS):
            print("DEBUG: is_batter_step -> True (keyword regex matched)")
            return True
        print("DEBUG: is_batter_step -> False")
        return False

    @staticmethod
    def normalize_batter_steps(steps: List[str]) -> List[str]:
        print(f"DEBUG: normalize_batter_steps called with {len(steps)} steps")
        batter_steps = [s for s in steps if Synthesizer.is_batter_step(s)]
        print(f"DEBUG: normalize_batter_steps detected {len(batter_steps)} batter_steps")
        if not batter_steps:
            print("DEBUG: normalize_batter_steps -> returning original steps (no batter steps found)")
            return steps
        combined = " ".join(batter_steps).lower()
        output = []
        if any(f in combined for f in ["flour", "gram", "rice", "maida", "semolina", "suji"]):
            output.append("Whisk the flour and liquids together, adding water gradually to form a smooth batter.")
            print("DEBUG: normalize_batter_steps -> added flour/liquid whisk instruction")
        if any(k in combined for k in Synthesizer.LEAVENING_HINTS):
            output.append("Add the leavening agent (Eno, baking soda, or similar).")
            print("DEBUG: normalize_batter_steps -> added leavening instruction")
        if "sugar" in combined or "salt" in combined or "spice" in combined:
            output.append("Add sugar, salt, and spices as required.")
            print("DEBUG: normalize_batter_steps -> added seasoning instruction")
        output.append("Mix gently until just combined.")
        print("DEBUG: normalize_batter_steps -> added final mixing instruction")
        final = []
        if any(k in combined for k in ["steam"]):
            final.append("Steam for 15 minutes.")
            print("DEBUG: normalize_batter_steps -> final action: steam")
        elif any(k in combined for k in ["fry"]):
            final.append("Fry until golden.")
            print("DEBUG: normalize_batter_steps -> final action: fry")
        elif any(k in combined for k in ["bake"]):
            final.append("Bake as required.")
            print("DEBUG: normalize_batter_steps -> final action: bake")
        elif any(k in combined for k in ["rest", "ferment"]):
            final.append("Allow the batter to rest or ferment as required.")
            print("DEBUG: normalize_batter_steps -> final action: rest/ferment")
        output.extend(final)
        print(f"DEBUG: normalize_batter_steps output: {output}")
        return output

    def _ingredient_tokens(self, name: str) -> List[str]:
        print(f"DEBUG: _ingredient_tokens input={repr(name)}")
        s = re.sub(r'[^a-z\s]', ' ', name.lower())
        toks = [t for t in s.split() if len(t) > 1]
        print(f"DEBUG: _ingredient_tokens output={toks}")
        return toks



    def ensure_ingredient_coverage(self, out_lines: List[str], merged_ings: List[Ingredient]) -> List[str]:
        """
        Improved ensure_ingredient_coverage. (Rewritten to fix insertion/order/index bugs.)
        """
        import re

        if not merged_ings:
            print("DEBUG: no merged_ings -> returning unchanged out_lines")
            return out_lines

        print("DEBUG: START ensure_ingredient_coverage")
        all_text = " ".join(out_lines).lower()
        print("DEBUG: all_text (truncated 800 chars) =", (all_text[:800] + '...') if len(all_text) > 800 else all_text)
        missing = []
        toks_by_name = {}
        toks_all_by_name = {}

        # Tokenize all merged ingredients, and mark which are missing
        for ing in merged_ings:
            name = ing.name.strip()
            toks = self._ingredient_tokens(name)
            print(f"DEBUG: checking ingredient={repr(name)} tokens={toks}")
            if not toks:
                print(f"DEBUG: ingredient {repr(name)} produced no tokens; skipping")
                continue
            toks_all_by_name[name] = toks

            # Check each token and print presence
            token_present_any = False
            for tok in toks:
                present_tok = (tok in all_text)
                print(f"DEBUG:   token check -> {tok!r}: present? {present_tok}")
                if present_tok:
                    token_present_any = True

            present = token_present_any
            print(f"DEBUG: ingredient {repr(name)} present overall? {present}")

            if not present:
                missing.append((name, ing.unit.strip().lower()))
                toks_by_name[name] = toks

        # Debug: show tokenization for all merged ingredients
        print("DEBUG: merged ingredient tokens:")
        for n, toks in toks_all_by_name.items():
            print("  -", repr(n), "->", toks)

        print("DEBUG: toks_by_name (missing-token map) =", toks_by_name)
        print("DEBUG: missing list =", missing)



        # ensure eggs are considered present when any step mentions "beaten"
        # Handle beaten eggs special case
        print("DEBUG: beaten-egg check: 'beaten' in all_text? ->", ("beaten" in all_text))
        if "beaten" in all_text:
            print("DEBUG: beaten detected in text; ensuring Eggs not wrongly marked missing")

            # If Eggs token was never added, create canonical entry
            eggs_present_in_tokens = any(name.lower() == "eggs" for name in toks_all_by_name)
            print("DEBUG: eggs_present_in_tokens before =", eggs_present_in_tokens)

            if not eggs_present_in_tokens:
                toks_all_by_name["Eggs"] = ["eggs"]
                print("DEBUG: added canonical Eggs -> ['eggs'] to toks_all_by_name")

            # Remove Eggs from missing if it was incorrectly included
            old_missing = missing[:]
            missing = [m for m in missing if m[0].lower() != "eggs"]
            toks_by_name.pop("Eggs", None)

            print("DEBUG: missing BEFORE egg-clean =", old_missing)
            print("DEBUG: missing AFTER  egg-clean =", missing)

        else:
            print("DEBUG: beaten not detected; no egg-cleanup applied")

        # Debug missing-ingredient output
        if missing:
            print("DEBUG: missing ingredients detected:")
            for name, unit in missing:
                print("  -", repr(name),
                      "unit=", repr(unit),
                      "tokens=", toks_by_name.get(name))
        else:
            print("DEBUG: no missing ingredients -> returning unchanged out_lines")
            return out_lines

        # Identify candidate lines to remove
        indices_to_remove = set()
        add_line_pattern = re.compile(r'^\s*(add|mix|combine)\b.*$', flags=re.I)

        # Determine protected indices
        protected_indices = set()
        print("DEBUG: scanning for protected indices...")
        for i, s in enumerate(out_lines):
            phase = self.classify_phase(s)
            time_flag = self.has_time_or_temp(s)
            print(f"DEBUG:   index={i} phase={phase!r} has_time_or_temp={time_flag} text={repr(s)}")

            if phase in ('cook', 'rest', 'finish') or time_flag:
                protected_indices.add(i)
                print("DEBUG:     -> marked as PROTECTED")
            else:
                print("DEBUG:     -> not protected")

        print("DEBUG: protected_indices (cook/rest/finish/time):", protected_indices)

        #

        for i, s in enumerate(out_lines):
            low = s.lower()
            print(f"\nDEBUG: scanning index={i} text={repr(s)}")

            if i in protected_indices:
                print("DEBUG:   -> SKIP (protected index)")
                continue

            is_add_line = bool(add_line_pattern.match(s))
            print("DEBUG:   is_add_line?", is_add_line)

            matched_any_token = False

            # Prefer removing add/mix/combine lines if they mention missing tokens
            if is_add_line:
                print("DEBUG:   ADD/MIX/COMBINE line detected, checking missing tokens…")
                for toks in toks_by_name.values():
                    print("DEBUG:     checking toks =", toks)
                    token_hits = [
                        (tok, bool(re.search(r'\b' + re.escape(tok) + r'\b', low)))
                        for tok in toks
                    ]
                    print("DEBUG:       token_hits =", token_hits)

                    if any(hit for tok, hit in token_hits):
                        matched_any_token = True

                        # Safety guard: Do not remove lines with cooking/rest/time info
                        phase = self.classify_phase(s)
                        time_flag = self.has_time_or_temp(s)
                        print(f"DEBUG:       phase={phase} time_flag={time_flag}")

                        if phase in ('cook', 'rest', 'finish') or time_flag:
                            print("DEBUG:       -> SKIP removal (protected by phase/time)")
                            break

                        print("DEBUG:       -> Marking for removal (ADD-line + missing token match)")
                        indices_to_remove.add(i)
                        break
                if not matched_any_token:
                    print("DEBUG:   -> no token matched; keep line")

            else:
                # Non-add lines: only consider removal if short-ish and mentions missing tokens
                print("DEBUG:   non-add line; checking short-line removal logic…")
                for toks in toks_by_name.values():
                    print("DEBUG:     checking toks =", toks)
                    token_hits = [
                        (tok, bool(re.search(r'\b' + re.escape(tok) + r'\b', low)))
                        for tok in toks
                    ]
                    print("DEBUG:       token_hits =", token_hits)

                    if any(hit for tok, hit in token_hits):
                        matched_any_token = True
                        word_count = len(low.split())
                        print("DEBUG:       missing-token match; word_count=", word_count)

                        if word_count <= 6:
                            print("DEBUG:       -> Marking for removal (short line + missing token)")
                            indices_to_remove.add(i)
                        else:
                            print("DEBUG:       -> NOT removed because word_count > 6")
                        break

                if not matched_any_token:
                    print("DEBUG:   -> no token matched; keep line")

        print("\nDEBUG: candidate indices_to_remove before protection check:", indices_to_remove)

        # Remove protected indices (safety)
        indices_to_remove = {i for i in indices_to_remove if i not in protected_indices}
        print("DEBUG: final indices_to_remove (after excluding protected):", indices_to_remove)


        # Debug: list current out_lines
        print("DEBUG: current out_lines:")
        for i, line in enumerate(out_lines):
            print(f"  [{i}] {repr(line)}")

        # Find first heat / cook occurrences and print detailed matches
        heat_idx = None
        cook_idx = None
        heat_matches = []
        cook_matches = []
        for i, line in enumerate(out_lines):
            low = line.lower()
            m_heat = re.search(r'\b(heat|preheat)\b', low)
            m_cook = (
                re.search(r'\bcook\b', low)
                or re.search(r'\bbake\b', low)
                or re.search(r'\bfry\b', low)
                or re.search(r'\bsimmer\b', low)
                or re.search(r'\bsteam\b', low)
            )
            if m_heat:
                heat_matches.append((i, line))
                if heat_idx is None:
                    heat_idx = i
                    print(f"DEBUG: found first HEAT match at index {i}: {repr(line)}")
                else:
                    print(f"DEBUG: found additional HEAT match at index {i}: {repr(line)}")
            if m_cook:
                cook_matches.append((i, line))
                if cook_idx is None:
                    cook_idx = i
                    print(f"DEBUG: found first COOK-like match at index {i}: {repr(line)}")
                else:
                    print(f"DEBUG: found additional COOK-like match at index {i}: {repr(line)}")

        if not heat_matches:
            print("DEBUG: no HEAT/preheat matches found")
        if not cook_matches:
            print("DEBUG: no COOK-like matches found (cook/bake/fry/simmer/steam)")

        # if there is a 'beaten' mention earlier than both, prefer the first occurrence of beaten/cook/heat
        beaten_idx = None
        beaten_lines = []
        for i, line in enumerate(out_lines):
            low = line.lower()
            if 'beaten' in low or re.search(r'\bbeat', low):
                beaten_lines.append((i, line))
                if beaten_idx is None:
                    beaten_idx = i

        if beaten_lines:
            print("DEBUG: beaten lines detected:")
            for i, line in beaten_lines:
                print(f"  beaten at index {i}: {repr(line)}")
        else:
            print("DEBUG: no beaten lines detected")

        # Choose the earliest relevant index for where to insert combination steps
        if heat_idx is not None and cook_idx is not None:
            if heat_idx < cook_idx:
                first_cook_idx = heat_idx
                print(f"DEBUG: selecting HEAT index as first_cook_idx -> {first_cook_idx}")
            else:
                first_cook_idx = cook_idx
                print(f"DEBUG: selecting COOK-like index as first_cook_idx -> {first_cook_idx}")
        elif heat_idx is not None:
            first_cook_idx = heat_idx
            print(f"DEBUG: selecting HEAT index as first_cook_idx -> {first_cook_idx}")
        elif cook_idx is not None:
            first_cook_idx = cook_idx
            print(f"DEBUG: selecting COOK-like index as first_cook_idx -> {first_cook_idx}")
        elif beaten_idx is not None:
            first_cook_idx = beaten_idx
            print(f"DEBUG: no heat/cook found; selecting BEATEN index as first_cook_idx -> {first_cook_idx}")
        else:
            first_cook_idx = None
            print("DEBUG: no heat/cook/beaten indices found; first_cook_idx = None")

        print(
            "DEBUG: debug heat_idx =", heat_idx,
            "cook_idx =", cook_idx,
            "beaten_idx =", beaten_idx,
            "chosen first_cook_idx =", first_cook_idx
        )


        # Identify wet-add candidate (use merged_ings info)
        liquid_keys = {'water', 'milk', 'buttermilk', 'yogurt', 'curd', 'oil', 'olive oil', 'lemon juice', 'juice'}
        liquid_units = {'ml', 'l', 'litre', 'liter', 'cup', 'cups', 'tbsp', 'tsp'}
        wet_tokens_all = set()

        print("\nDEBUG: BEGIN scanning merged_ings for liquid/wet tokens")
        print("DEBUG: liquid_keys =", liquid_keys)
        print("DEBUG: liquid_units =", liquid_units)

        # detect if any existing step actually mixes oil into batter
        print("DEBUG: checking if oil is explicitly mixed into batter...")
        is_oil_in_batter = any(
            re.search(r'\b(oil)\b', s.lower()) and
            re.search(r'\b(mix|combine|whisk|add|stir|fold)\b', s.lower())
            for s in out_lines
        )
        print("DEBUG: is_oil_in_batter =", is_oil_in_batter)

        for ing in merged_ings:
            nlow = ing.name.strip().lower()
            unit = (ing.unit or "").strip().lower()
            toks = self._ingredient_tokens(ing.name)

            print(f"\nDEBUG: ingredient={repr(ing.name)}, unit={repr(unit)}, tokens={toks}")

            # compute liquid detection
            liquid_name_match = any(k in nlow for k in liquid_keys)
            liquid_unit_match = unit in liquid_units
            is_liquid = liquid_name_match or liquid_unit_match

            print("DEBUG:   liquid_name_match =", liquid_name_match)
            print("DEBUG:   liquid_unit_match =", liquid_unit_match)
            print("DEBUG:   -> is_liquid =", is_liquid)

            # special handling for oil
            if nlow == 'oil':
                print("DEBUG:   special-case: oil detected")
                print("DEBUG:   is_oil_in_batter =", is_oil_in_batter)
                if is_liquid and not is_oil_in_batter:
                    print("DEBUG:   -> oil treated as PAN GREASE, SKIPPING adding oil tokens to wet_tokens_all")
                    continue
                else:
                    print("DEBUG:   -> oil treated as actual wet ingredient (added to wet_tokens_all)")

            if is_liquid:
                print("DEBUG:   adding tokens to wet_tokens_all:", toks)
                for t in toks:
                    wet_tokens_all.add(t)
            else:
                print("DEBUG:   not liquid; skip token add")

        print("DEBUG: FINAL wet_tokens_all =", wet_tokens_all)
        print("DEBUG: END scanning merged_ings for liquid/wet tokens\n")


        wet_add_index = None
        print("\nDEBUG: BEGIN scanning out_lines for a wet-add candidate")
        print("DEBUG: wet_tokens_all =", wet_tokens_all)
        print("DEBUG: protected_indices =", protected_indices)
        print("DEBUG: add_line_pattern =", add_line_pattern.pattern)

        if wet_tokens_all:
            for i, s in enumerate(out_lines):
                print(f"\nDEBUG: checking line[{i}] = {repr(s)}")
                if i in protected_indices:
                    print(f"DEBUG:  -> SKIP (index {i} is protected)")
                    continue
                if add_line_pattern.match(s):
                    print("DEBUG:  -> matches add_line_pattern")
                    low = s.lower()
                    # check direct liquid word presence
                    liquid_word_found = any(w in low for w in liquid_keys)
                    # check for wet token matches as word boundaries
                    wet_token_found = any(re.search(r'\b' + re.escape(tok) + r'\b', low) for tok in wet_tokens_all)
                    print("DEBUG:    liquid_word_found =", liquid_word_found)
                    print("DEBUG:    wet_token_found =", wet_token_found)
                    if liquid_word_found or wet_token_found:
                        wet_add_index = i
                        print("DEBUG: found wet-add candidate at index", i, "line:", repr(s))
                        break
                    else:
                        print("DEBUG:  -> No wet word/token match in this 'add' line")
                else:
                    print("DEBUG:  -> Not an add/mix/combine line (skipping)")

        print("\nDEBUG: wet_add_index final value =", wet_add_index)

        # If wet_add_index exists prefer removing it (and we'll insert before it)
        if wet_add_index is not None:
            print("DEBUG: adding wet_add_index to indices_to_remove:", wet_add_index)
            indices_to_remove.add(wet_add_index)
        else:
            print("DEBUG: wet_add_index is None — not adding to indices_to_remove here")

        print("DEBUG: current indices_to_remove (pre-insert-decide) =", indices_to_remove)
        insert_idx = None
        if wet_add_index is not None:
            insert_idx = wet_add_index
            print("DEBUG: prefer insert at wet_add_index:", insert_idx)
        elif indices_to_remove:
            insert_idx = min(indices_to_remove)
            print("DEBUG: will insert at earliest removed index:", insert_idx)
        elif first_cook_idx is not None:
            # For ingredient combination steps, insert AFTER heat but BEFORE active cooking
            # Look for stir/fry/simmer (active cooking) not just heat
            active_cook_idx = None
            for i, s in enumerate(out_lines):
                if i <= first_cook_idx:
                    continue
                if re.search(r'\b(stir|fry|simmer|sauté|saute)\b', s.lower()):
                    active_cook_idx = i
                    break

            if active_cook_idx is not None:
                insert_idx = active_cook_idx
                print(f"DEBUG: found active cooking at index {active_cook_idx}; will insert before it: {insert_idx}")
            else:
                first_cook_line = out_lines[first_cook_idx].lower() if first_cook_idx < len(out_lines) else ""
                has_active_cooking = bool(re.search(r'\b(stir|fry|simmer|sauté|saute)\b', first_cook_line))

                if has_active_cooking:
                    # Line like "Heat oil and stir-fry" - insert AFTER it
                    insert_idx = first_cook_idx + 1
                    print(f"DEBUG: first cook line has active cooking (stir/fry); will insert after it at: {insert_idx}")
                else:
                    # Line like "Heat oil" only - insert at it (before heating)
                    insert_idx = first_cook_idx
                    print("DEBUG: will insert before first cook/rest/finish/time index:", insert_idx)
        else:
            insert_idx = len(out_lines)
            print("DEBUG: no cook/rest/finish/time line found; inserting at end:", insert_idx)

        print("DEBUG: insertion decision complete; insert_idx =", insert_idx)
        for i, line in enumerate(out_lines):
            print(f"  [{i}] {repr(line)}")

        removed_indices_sorted = sorted(indices_to_remove, reverse=True)
        removed_count_before = 0
        print("DEBUG: removal order (descending) =", removed_indices_sorted)

        for idx in removed_indices_sorted:
            try:
                # defensive check
                if idx < 0 or idx >= len(out_lines):
                    print(f"DEBUG: skip pop index {idx} (out of range for current out_lines length {len(out_lines)})")
                    continue
                popped = out_lines.pop(idx)
                removed_count_before += 1
                print(f"DEBUG: popped out_lines[{idx}] = {repr(popped)}")
            except Exception as exc:
                print(f"DEBUG: failed to pop index {idx}: {exc}")

        print("DEBUG: removal pass complete, removed_count_before =", removed_count_before)
        print("DEBUG: out_lines AFTER removal:")
        for i, line in enumerate(out_lines):
            print(f"  [{i}] {repr(line)}")

        # compute post-removal cook/heat indices
        cook_idx_post = None
        heat_idx_post = None
        print("\nDEBUG: scanning post-removal out_lines for cook/heat indices")
        for i, line in enumerate(out_lines):
            low = line.lower()
            is_cook = bool(re.search(r'\b(cook|bake|fry|simmer|steam)\b', low))
            is_heat = bool(re.search(r'\b(heat|preheat)\b', low))
            if cook_idx_post is None and is_cook:
                cook_idx_post = i
                print(f"DEBUG: found cook at index {i}: {repr(line)}")
            if heat_idx_post is None and is_heat:
                heat_idx_post = i
                print(f"DEBUG: found heat at index {i}: {repr(line)}")
            # continue scanning so we show first occurrences for each

        print("DEBUG: cook_idx_post =", cook_idx_post, "heat_idx_post =", heat_idx_post)

        #
        if cook_idx_post is not None and heat_idx_post is not None and cook_idx_post < heat_idx_post:
            print(f"DEBUG: condition met -> cook_idx_post={cook_idx_post} < heat_idx_post={heat_idx_post}; will attempt move")
            try:
                # defensive bounds check
                if not (0 <= heat_idx_post < len(out_lines)):
                    print(f"DEBUG: abort move - heat_idx_post {heat_idx_post} out of range (len={len(out_lines)})")
                elif not (0 <= cook_idx_post <= len(out_lines)):
                    print(f"DEBUG: abort move - cook_idx_post {cook_idx_post} out of range (len={len(out_lines)})")
                else:
                    heat_line = out_lines.pop(heat_idx_post)
                    # If popping an earlier index shifts cook_idx_post, adjust target:
                    # If heat_idx_post < cook_idx_post then after pop the cook_idx_post decreases by 1
                    adjusted_target = cook_idx_post if heat_idx_post > cook_idx_post else max(0, cook_idx_post - 1)
                    out_lines.insert(adjusted_target, heat_line)
                    print(f"DEBUG: moved heat line from index {heat_idx_post} to {adjusted_target} to ensure heating before cooking")
                    print(f"DEBUG: moved line content: {repr(heat_line)}")
                    # reflect on how the removal changed indices
                    print("DEBUG: out_lines snapshot after move:")
                    for i, line in enumerate(out_lines):
                        print(f"  [{i}] {repr(line)}")

                    # adjust insert_idx if we moved a line that affects it (original semantics)
                    if insert_idx is not None:
                        print("DEBUG: adjusting insert_idx due to move (original insert_idx =", insert_idx, ")")
                        # heat was removed before insert point and inserted after — adjust accordingly
                        if heat_idx_post < insert_idx and adjusted_target >= insert_idx:
                            insert_idx -= 1
                            print("DEBUG: case A -> decreased insert_idx by 1")
                        # heat was removed after insert point and inserted before or at insert point
                        elif heat_idx_post > insert_idx and adjusted_target <= insert_idx:
                            insert_idx += 1
                            print("DEBUG: case B -> increased insert_idx by 1")
                        else:
                            print("DEBUG: case None -> insert_idx unchanged")
                        print("DEBUG: post-move insert_idx =", insert_idx)
            except Exception as exc:
                print("DEBUG: failed to reorder heat/cook lines:", exc)
        else:
            print("DEBUG: no reordering required (either cook/heat not both found or already ordered)")

        # After removals and possible reordering, adjust insert_idx to account for how many removed indices were < original insert_idx
        if insert_idx is not None:
            num_removed_before = sum(1 for r in indices_to_remove if r < insert_idx)
            new_insert_idx = max(0, insert_idx - num_removed_before)
            print(f"DEBUG: adjusted insert_idx from {insert_idx} -> {new_insert_idx} (removed_before={num_removed_before})")
            insert_idx = new_insert_idx

        # classify missing into dry vs wet (based on missing list)
        dry = []
        wet = []
        for name, unit in missing:
            nlow = name.lower()
            print(f"\nDEBUG: checking missing ingredient: {name!r}, unit={unit!r}")
            print("DEBUG:  nlow =", nlow)

            is_liquid_name = any(k in nlow for k in liquid_keys)
            is_liquid_unit = (unit in liquid_units) or (unit in {'ml', 'l'})
            print("DEBUG:  is_liquid_name =", is_liquid_name, "| is_liquid_unit =", is_liquid_unit)

            if is_liquid_name or is_liquid_unit:
                print("DEBUG:  -> classified as WET (liquid keyword or unit)")
                wet.append(name)
            else:
                # special egg rule
                is_egg = any(k in nlow for k in ('egg', 'eggs'))
                is_piece_unit = unit in {'pc', 'pcs', 'piece'}
                print("DEBUG:  is_egg =", is_egg, "| is_piece_unit =", is_piece_unit)

                if is_egg and is_piece_unit:
                    print("DEBUG:  -> classified as WET (egg mixture via piece-unit override)")
                    wet.append(name)
                else:
                    print("DEBUG:  -> classified as DRY")
                    dry.append(name)

        print("\nDEBUG: initial dry list =", dry)
        print("DEBUG: initial wet list =", wet)

        # Remove ingredients from dry/wet lists if they are already explicitly mentioned in steps
        # Use the original all_text (pre-removal) so we do not re-add already-covered ingredients
        existing_text = all_text

        def _already_covered(name: str) -> bool:
            toks = self._ingredient_tokens(name)
            if not toks:
                return False
            # Prefer strict word-boundary token coverage, fall back to substring check for multi-word names
            if all(re.search(r"\\b" + re.escape(t) + r"\\b", existing_text) for t in toks):
                return True
            return name.strip().lower() in existing_text

        dry = [n for n in dry if not _already_covered(n)]
        wet = [n for n in wet if not _already_covered(n)]

        print("DEBUG: dry after coverage-filter =", dry)
        print("DEBUG: wet after coverage-filter =", wet)

        # Also include wet display names from merged ingredients (even if not missing)
        merged_wet_names = []
        print("\nDEBUG: scanning merged_ings for implicit wet ingredients")
        for ing in merged_ings:
            nlow = ing.name.strip().lower()
            unit = (ing.unit or "").strip().lower()
            is_liquid_name = any(k in nlow for k in liquid_keys)
            is_liquid_unit = unit in liquid_units
            print(f"DEBUG:  merged ingredient {ing.name!r}, unit={unit!r} -> liquid_name={is_liquid_name}, liquid_unit={is_liquid_unit}")

            if is_liquid_name or is_liquid_unit:
                merged_wet_names.append(ing.name)

        print("DEBUG: merged_wet_names =", merged_wet_names)

        # Special-case: water is implicit in soak/grind batter flows (idli-style)
        has_soak_line = any(re.search(r'\bsoak\b', s, flags=re.I) for s in out_lines)
        has_grind_line = any('grind' in s.lower() for s in out_lines)
        only_missing_water = len(wet) == 1 and wet[0].strip().lower() == 'water'
        skip_autogen_water = False
        if has_soak_line and has_grind_line and only_missing_water:
            print("DEBUG: implicit water detected via soak+grind; suppressing auto-gen water add step")
            wet = []  # treat water as already covered
            skip_autogen_water = True
            # If no grind line mentions water, fold a gentle hint into the first grind line
            if not any('water' in s.lower() for s in out_lines):
                for idx, line in enumerate(out_lines):
                    if 'grind' in line.lower():
                        out_lines[idx] = line.rstrip(' .;') + '; add water as needed to grind.'
                        break

        # If water was implicit, skip the auto-generated add_step entirely
        if skip_autogen_water:
            print("DEBUG: skip_autogen_water=True -> returning out_lines without insertion")
            return out_lines

        # If no wet missing but dry exists, pick an existing wet signal from merged ingredients
        if not wet and merged_wet_names and dry:
            print("DEBUG: no wet missing ingredients, but dry exists — selecting fallback wet from merged ingredients")

            preferred = None
            for w in merged_wet_names:
                print(f"DEBUG:  checking merged wet candidate {w!r}")
                if 'warm' in w.lower():
                    preferred = w
                    print("DEBUG:   -> chosen because contains 'warm'")
                    break

            if not preferred:
                preferred = merged_wet_names[0]
                print(f"DEBUG: fallback preferred={preferred!r} (first merged wet name)")

            if preferred not in wet:
                print(f"DEBUG: adding preferred fallback wet ingredient: {preferred!r}")
                wet.append(preferred)


        #
        def short_label(full_name: str, keep_warm_for_liquid: bool = True) -> str:
            print(f"DEBUG: short_label input: {full_name!r}, keep_warm_for_liquid={keep_warm_for_liquid}")
            s = full_name.strip()
            s = re.sub(r'[\(\)\[\]\,]', ' ', s)
            s = s.replace('-', ' ')
            s = re.sub(r'\s+', ' ', s).strip()
            print(f"DEBUG: short_label cleaned -> {s!r}")
            if not s:
                print("DEBUG: short_label -> fallback to title() of full_name")
                return full_name.title()
            parts = s.split()
            print(f"DEBUG: short_label parts = {parts}")
            if keep_warm_for_liquid and any(k in s.lower() for k in ('water', 'milk', 'buttermilk', 'yogurt', 'oil', 'juice', 'curd')):
                if len(parts) == 1:
                    outp = parts[0].title()
                    print(f"DEBUG: short_label -> single-word liquid -> {outp!r}")
                    return outp
                outp = " ".join(parts[-2:]).title()
                print(f"DEBUG: short_label -> keep warm for liquid -> {outp!r}")
                return outp
            if parts[-1].lower() in {'flour', 'sugar', 'water', 'yeast', 'salt', 'oil', 'milk', 'yogurt', 'semolina'}:
                outp = parts[-1].title()
                print(f"DEBUG: short_label -> last token matches well-known -> {outp!r}")
                return outp
            if len(parts) == 1:
                outp = parts[0].title()
                print(f"DEBUG: short_label -> single-part fallback -> {outp!r}")
                return outp
            outp = " ".join(parts[-2:]).title()
            print(f"DEBUG: short_label -> default two-word label -> {outp!r}")
            return outp

        print("DEBUG: creating display labels for dry/wet")
        print("DEBUG: dry =", dry)
        print("DEBUG: wet =", wet)

        disp_dry = []
        for n in dry:
            try:
                label = short_label(n, keep_warm_for_liquid=False)
            except Exception as e:
                print(f"DEBUG: short_label raised for dry {n!r}: {e!r}; falling back to title()")
                label = n.title()
            print(f"DEBUG: disp_dry item: {n!r} -> {label!r}")
            disp_dry.append(label)

        disp_wet = []
        for n in wet:
            try:
                label = short_label(n, keep_warm_for_liquid=True)
            except Exception as e:
                print(f"DEBUG: short_label raised for wet {n!r}: {e!r}; falling back to title()")
                label = n.title()
            print(f"DEBUG: disp_wet item: {n!r} -> {label!r}")
            disp_wet.append(label)

        print("DEBUG: disp_dry =", disp_dry)
        print("DEBUG: disp_wet =", disp_wet)

        # When building add_step, also include ALL dry/wet ingredients from merged_ings
        # (not just missing ones) to ensure complete recipe instructions
        all_dry_from_merged = []
        all_wet_from_merged = []
        for ing in merged_ings:
            nlow = ing.name.strip().lower()
            unit = (ing.unit or "").strip().lower()
            is_liquid_name = any(k in nlow for k in liquid_keys)
            is_liquid_unit = unit in liquid_units
            if is_liquid_name or is_liquid_unit:
                all_wet_from_merged.append(ing.name)
            else:
                all_dry_from_merged.append(ing.name)
        
        # Combine missing dry with all dry from merged (avoiding duplicates)
        full_dry_list = list(dict.fromkeys(dry + all_dry_from_merged))
        # Combine missing wet with all wet from merged (avoiding duplicates)
        full_wet_list = list(dict.fromkeys(wet + all_wet_from_merged))
        
        print(f"DEBUG: full_dry_list (missing + all merged) = {full_dry_list}")
        print(f"DEBUG: full_wet_list (missing + all merged) = {full_wet_list}")
        
        # Create display labels for the full lists
        disp_full_dry = []
        for n in full_dry_list:
            try:
                label = short_label(n, keep_warm_for_liquid=False)
            except Exception as e:
                label = n.title()
            disp_full_dry.append(label)
        
        disp_full_wet = []
        for n in full_wet_list:
            try:
                label = short_label(n, keep_warm_for_liquid=True)
            except Exception as e:
                label = n.title()
            disp_full_wet.append(label)
        
        print(f"DEBUG: disp_full_dry = {disp_full_dry}")
        print(f"DEBUG: disp_full_wet = {disp_full_wet}")

        # Build combined instruction (this is your add_step) using full lists
        if disp_full_dry and disp_full_wet:
            if len(disp_full_dry) > 1:
                dry_txt = ", ".join(disp_full_dry[:-1]) + " and " + disp_full_dry[-1]
            else:
                dry_txt = disp_full_dry[0]
            if len(disp_full_wet) > 1:
                wet_txt = ", ".join(disp_full_wet[:-1]) + " and " + disp_full_wet[-1]
            else:
                wet_txt = disp_full_wet[0]
            add_step = f"Combine {dry_txt}. Then add {wet_txt} and mix until just combined."
            # Mark this as autogenerated so it won't be affected by soy sauce deduping later
            add_step = f"[AUTO-GEN] {add_step}"
            print("DEBUG: built add_step (dry+wet) ->", add_step)
        elif disp_full_dry:
            if len(disp_full_dry) > 1:
                dry_txt = ", ".join(disp_full_dry[:-1]) + " and " + disp_full_dry[-1]
            else:
                dry_txt = disp_full_dry[0]
            add_step = f"Combine {dry_txt} and mix as required."
            print("DEBUG: built add_step (dry only) ->", add_step)
        else:
            if len(disp_full_wet) > 1:
                wet_txt = ", ".join(disp_full_wet[:-1]) + " and " + disp_full_wet[-1]
            else:
                wet_txt = disp_full_wet[0] if disp_full_wet else ""
            add_step = f"Add {wet_txt} and mix until just combined."
            print("DEBUG: built add_step (wet only) ->", add_step)

        print("DEBUG: final add_step =", add_step)


        # Insert combined step at determined index (or append)
        if insert_idx is None:
            print("DEBUG: insert_idx is None -> will append at end")
            out_lines.append(add_step)
            did_insert_at = len(out_lines) - 1
        elif insert_idx > len(out_lines):
            print(f"DEBUG: insert_idx {insert_idx} > len(out_lines) {len(out_lines)} -> appending at end")
            out_lines.append(add_step)
            did_insert_at = len(out_lines) - 1
        else:
            print(f"DEBUG: inserting combined step at index {insert_idx}")

            inserted_flags = [False] * len(out_lines)
            #
            # --- improved insertion logic: prefer after last GRIND before FERMENT ---
            # find indices of relevant phases
            ferment_idxs = [i for i, s in enumerate(out_lines) if re.search(r'\bferment(?:ed|ing)?\b', s, flags=re.I)]
            grind_idxs = [i for i, s in enumerate(out_lines) if re.search(r'\bgrind\b', s, flags=re.I)]

            # existing fallback candidate (existing code computed something like `first_cook_idx` or `insert_idx`)
            # keep the variable name `insert_idx` consistent with your code below; if not present yet, compute it:
            # find first cook/rest/finish/time index (your current logic that you already have)
            try:
                _current_insert_idx = insert_idx  # if already computed above
            except NameError:
                # replicate the original fallback: insert before first cook/rest/finish/time index
                _current_insert_idx = next(
                    (i for i, s in enumerate(out_lines)
                    if any(k in s.lower() for k in ('steam', 'cook', 'bake', 'fry', 'grill', 'roast', 'simmer'))
                    or self.has_time_or_temp(s)
                    ), len(out_lines)
                )

            # Now prefer placing after last grind that occurs before fermentation (ideal for batters)
            chosen_idx = None
            if ferment_idxs:
                first_ferment = min(ferment_idxs)
                # find last grind strictly before the first ferment
                prior_grinds = [g for g in grind_idxs if g < first_ferment]
                if prior_grinds:
                    # insert *after* the last grind (i.e., at index last_grind + 1)
                    chosen_idx = prior_grinds[-1] + 1
                else:
                    # no prior grind — insert just before the earliest ferment
                    chosen_idx = first_ferment
            else:
                # no ferment found — keep the original logic (before first cook/rest/time)
                chosen_idx = _current_insert_idx

            # Ensure chosen_idx is within bounds and not before index 0
            chosen_idx = max(0, min(chosen_idx, len(out_lines)))

            # Avoid inserting before a protected index (like an explicitly protected soak). If chosen spot is protected,
            # attempt to move forward to the nearest non-protected index that is <= first cook/time index.
            if 'protected_indices' in locals() and chosen_idx in protected_indices:
                # move forward until non-protected or end
                advance_idx = chosen_idx
                while advance_idx in protected_indices and advance_idx < len(out_lines):
                    advance_idx += 1
                # if we've stepped past end, fallback to original insert point
                if advance_idx < len(out_lines):
                    chosen_idx = advance_idx
                else:
                    # fallback: use previously computed _current_insert_idx (safe fallback)
                    chosen_idx = _current_insert_idx

            # finally set insert_idx for use where you call out_lines.insert(insert_idx, add_step)
            insert_idx = chosen_idx

            # --- replace the "will insert before first cook/rest/finish/time index" logic with this ---
            # `first_cook_idx` already computed earlier (or None)

            # Find candidate indices that are mix/grind/combine lines (and not protected), before cook/time
            mix_like_pattern = re.compile(r'\b(grind|mix|combine|whisk|beat|fold|blend|stir)\b', flags=re.I)

            last_mix_idx = None
            for idx, line in enumerate(out_lines):
                if idx in protected_indices:
                    continue
                # stop scanning when we reach the cook index (we don't want to insert after cook)
                if first_cook_idx is not None and idx >= first_cook_idx:
                    break
                if mix_like_pattern.search(line):
                    last_mix_idx = idx

            if last_mix_idx is not None:
                # place the add-step immediately AFTER the last mix/grind line
                insert_idx = last_mix_idx + 1
            else:
                # fallback: keep original behavior (insert before the first cook/rest/finish/time index if present)
                insert_idx = first_cook_idx if first_cook_idx is not None else len(out_lines)

            # perform insertion (existing creation of add_step remains the same)
            out_lines.insert(insert_idx, add_step)

            # insert the new step
            out_lines.insert(insert_idx, add_step)
            inserted_flags.insert(insert_idx, True)
            # run dedupe and re-generate flags mapping (simple approach: dedupe returns new list)
            out_lines = self._dedupe_steps(out_lines)

            did_insert_at = insert_idx

        print("DEBUG: insertion performed at index", did_insert_at)

        # Show after insertion
        print("\nDEBUG: out_lines AFTER insertion:")
        for i, line in enumerate(out_lines):
            print(f"  [{i}] {repr(line)}")

        print("DEBUG: FINAL out_lines from ensure_ingredient_coverage:")
        for i, line in enumerate(out_lines):
            print(f"  [{i}] {repr(line)}")

        print("DEBUG: END ensure_ingredient_coverage\n")
        return out_lines


    def _collapse_repeated_words(self, s: str) -> str:
        print(f"DEBUG: _collapse_repeated_words INPUT = {repr(s)}")
        out = re.sub(r'\b(\w+)(?: \1\b)+', r'\1', s, flags=re.I)
        print(f"DEBUG: _collapse_repeated_words OUTPUT = {repr(out)}")
        return out

    def _token_set(self, s: str) -> set:
        """Return normalized token set used for fuzzy matching, with debug."""
        print("\nDEBUG: _token_set START")
        print("DEBUG:   raw input =", repr(s))

        try:
            k = self._normalize_for_dedupe(s)
            print("DEBUG:   normalized key =", repr(k))
        except Exception as exc:
            print("DEBUG:   ERROR in _normalize_for_dedupe:", exc)
            print("DEBUG:   -> returning empty set")
            return set()

        if not k:
            print("DEBUG:   normalized key is empty -> returning empty set")
            print("DEBUG: _token_set END\n")
            return set()

        toks = set(k.split())
        print("DEBUG:   token set =", toks)
        print("DEBUG: _token_set END\n")
        return toks




    def _normalize_for_dedupe(self, s: str) -> str:
        print(f"\nDEBUG: _normalize_for_dedupe START input={repr(s)}")

        if not s:
            print("DEBUG: _normalize_for_dedupe early exit: empty string")
            return ""

        # collapse repeated words
        before_collapse = s
        s = self._collapse_repeated_words(s)
        print(f"DEBUG: After collapse: {repr(before_collapse)} -> {repr(s)}")

        # lowercase
        s = s.lower()
        print(f"DEBUG: Lowercased: {repr(s)}")

        # -------------------------------------------------------------
        # NEW NORMALIZATION RULES
        # -------------------------------------------------------------
        # soak → unify variants
        before = s
        s = re.sub(r'\bsoaked\b', 'soak', s)
        print(f"DEBUG: soak normalization: {repr(before)} -> {repr(s)}")

        # beaten eggs → unify variants
        before = s
        s = re.sub(r'\b(beaten|beaten eggs|egg mixture)\b', 'beaten_eggs', s)
        print(f"DEBUG: beaten normalization: {repr(before)} -> {repr(s)}")

        # grind / grinding / ground → grind
        before = s
        s = re.sub(r'\b(grind(?:ing)?|ground)\b', 'grind', s)
        print(f"DEBUG: grind normalization: {repr(before)} -> {repr(s)}")

        # -------------------------------------------------------------
        # time & temperature normalization
        # -------------------------------------------------------------
        before = s
        s = re.sub(r'\b\d+\s*(?:[-–]\s*\d+)?\s*(?:hours?|hrs?|minutes?|mins?)\b',
                    ' time ', s, flags=re.I)
        print(f"DEBUG: time normalization: {repr(before)} -> {repr(s)}")

        before = s
        s = re.sub(r'\b\d+\s*(?:°\s?[cf]|°c|°f)\b', ' temp ', s, flags=re.I)
        print(f"DEBUG: temp normalization: {repr(before)} -> {repr(s)}")

        # -------------------------------------------------------------
        # remove punctuation & digits
        # -------------------------------------------------------------
        before_clean = s
        s = re.sub(r'[^a-z\s]', ' ', s)
        print(f"DEBUG: punctuation/digit removal: {repr(before_clean)} -> {repr(s)}")

        # collapse whitespace
        before_strip = s
        s = re.sub(r'\s+', ' ', s).strip()
        print(f"DEBUG: After whitespace normalization: {repr(before_strip)} -> {repr(s)}")

        tokens = s.split()
        print(f"DEBUG: Final tokens = {tokens}")

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
            print(f"DEBUG: Checking token={repr(t)}")
            if t in noisy:
                print(f"DEBUG:  -> SKIP (noisy word)")
                continue
            # short tokens like 'of' filtered already; skip 1-char tokens
            if len(t) <= 1:
                print(f"DEBUG:  -> SKIP (1-char token)")
                continue
            # drop obvious adjectives that add noise ('smooth', 'golden', 'fresh') \u2014 optional
            if t in {'smooth', 'golden', 'fresh', 'warm', 'hot', 'cold'}:
                print(f"DEBUG:  -> SKIP (adjective noise)")
                continue
            kept.append(t)
            print(f"DEBUG:  -> KEEP")

        print("DEBUG: kept tokens before fallback =", kept)

        if not kept:
            # fallback: use tokens excluding pure punctuation/stopwords
            print("DEBUG: kept empty -> fallback path activated")
            kept = [t for t in tokens if t not in stopwords]
            print("DEBUG: kept tokens AFTER fallback =", kept)

        # produce order-insensitive fingerprint: unique sorted tokens
        key_tokens = sorted(set(kept))
        print("DEBUG: final key_tokens =", key_tokens)

        return " ".join(key_tokens)

    #
    def _dedupe_steps(self, steps: List[str]) -> List[str]:
        print("DEBUG: _dedupe_steps START")
        # seen_keys maps normalized_key -> index in out
        seen_keys: Dict[str, int] = {}
        out: List[str] = []

        for idx, s in enumerate(steps):
            print(f"DEBUG:   Step[{idx}] original={repr(s)}")
            key = self._normalize_for_dedupe(s)
            print(f"DEBUG:   Step[{idx}] normalized key={repr(key)}")

            if not key:
                print(f"DEBUG:   -> SKIP (empty key)")
                continue

            new_tokens = set(key.split())
            if not new_tokens:
                print(f"DEBUG:   -> SKIP (no tokens after split)")
                continue

            # Try to find an existing key that is highly overlapping (near-duplicate)
            found_similar = None
            for k_existing in list(seen_keys.keys()):
                print(f"\nDEBUG:   Checking existing key={repr(k_existing)}")

                existing_tokens = set(k_existing.split())
                print("DEBUG:      existing_tokens =", existing_tokens)

                if not existing_tokens:
                    print("DEBUG:      -> SKIP (existing key has empty token set)")
                    continue

                inter = new_tokens & existing_tokens
                union = new_tokens | existing_tokens

                print("DEBUG:      inter =", inter)
                print("DEBUG:      union =", union)

                overlap = (len(inter) / len(union)) if union else 0.0
                print(f"DEBUG:      overlap ratio = {overlap:.4f}")

                if overlap >= 0.40:
                    found_similar = k_existing
                    print(f"DEBUG:   -> found similar existing key={repr(k_existing)} with overlap={overlap:.4f}")
                    break
                else:
                    print("DEBUG:      -> not similar enough (needs >= 0.40)")

            if found_similar:
                # If new key is more informative (more distinct tokens), replace the existing kept step
                existing_tokens = set(found_similar.split())
                existing_idx = seen_keys[found_similar]
                existing_step = out[existing_idx]

                def _has_restish(text: str) -> bool:
                    return bool(re.search(r"\b(rest|rise|proof|prove|ferment)\b", text, flags=re.I))

                # If phases differ (e.g., rest vs knead), keep both even if tokens overlap
                try:
                    phase_existing = self.classify_phase(existing_step)
                    phase_new = self.classify_phase(s)
                except Exception:
                    phase_existing = phase_new = None

                if phase_existing and phase_new and phase_existing != phase_new:
                    print("DEBUG:   -> phase mismatch (", phase_existing, "vs", phase_new, ") -> keeping both")
                    seen_keys[key] = len(out)
                    out.append(s)
                    continue

                # Preserve explicit rest/proof/ferment steps even if another step overlaps more tokens
                if _has_restish(existing_step) and not _has_restish(s):
                    print("DEBUG:   -> SKIP replacement to preserve rest/proof/ferment step")
                    continue

                if len(new_tokens) > len(existing_tokens):
                    replaced_idx = existing_idx
                    print(f"DEBUG:   -> REPLACING less-informative step at index {replaced_idx} (key {repr(found_similar)}) with new step")
                    out[replaced_idx] = s
                    # update mapping: remove old key, add new key at same index
                    del seen_keys[found_similar]
                    seen_keys[key] = replaced_idx
                else:
                    print("DEBUG:   -> SKIP (existing step is more informative or equal)")
                    # keep existing, skip this new (less informative) variant
                    continue
            else:
                # No similar existing key found — ensure we don't keep exact duplicates
                if key in seen_keys:
                    print(f"DEBUG:   -> SKIP (exact duplicate key present)")
                    continue
                seen_keys[key] = len(out)
                out.append(s)
                print(f"DEBUG:   -> KEEP (new key stored at index {seen_keys[key]})")

        print("DEBUG: _dedupe_steps FINAL =", out)
        return out



    def generate_prep_from_ingredients(self, merged_ings: List[Ingredient]) -> List[str]:
        print("DEBUG: generate_prep_from_ingredients START")
        names = {ing.name.strip().lower(): ing for ing in merged_ings}
        print("DEBUG: ingredient names map =", names)

        prep_lines: List[str] = []

        rice_keys = {'rice', 'idli rice', 'parboiled rice', 'idli rice (parboiled)'}
        urad_keys = {'urad dal', 'urad', 'black gram', 'black-gram'}

        has_rice = any(k in names for k in rice_keys)
        has_urad = any(k in names for k in urad_keys)

        print(f"DEBUG: has_rice={has_rice}, has_urad={has_urad}")

        if has_rice and has_urad:
            print("DEBUG: matched rice+urad dal prep rule")
            prep_lines.append("Soak rice and urad dal separately for 4\u20136 hours, then drain.")
            prep_lines.append("Grind soaked rice and urad dal to a smooth batter and combine; ferment if required.")
            print("DEBUG: prep_lines =", prep_lines)
            return prep_lines

        if 'semolina' in names or 'rava' in names:
            print("DEBUG: matched semolina/rava prep rule")
            prep_lines.append("Mix semolina and Yogurt to make a batter.")
            prep_lines.append("Add water gradually.")
            prep_lines.append("Add Eno and steam the batter.")
            print("DEBUG: prep_lines =", prep_lines)
            return prep_lines

        flour_aliases = {'gram flour', 'besan', 'maida', 'atta', 'flour'}
        yogurt_aliases = {'yogurt', 'curd', 'dahi', 'yoghurt'}

        has_flour = any(k in names for k in flour_aliases)
        has_yogurt = any(k in names for k in yogurt_aliases)

        print(f"DEBUG: has_flour={has_flour}, has_yogurt={has_yogurt}")

        if has_flour and has_yogurt:
            print("DEBUG: matched flour+yogurt prep rule")
            prep_lines.append("Whisk the flour and yogurt together, adding water gradually to form a smooth batter.")
            print("DEBUG: prep_lines =", prep_lines)
            return prep_lines

        print("DEBUG: No prep rules matched -> returning empty list")
        return prep_lines






    #
    def merge_semantic_steps(self, steps: List[str]) -> List[str]:

        print("DEBUG: merge_semantic_steps START")
        print("DEBUG: input steps =", steps)

        # Defensive: ensure steps is a list of strings
        if not steps:
            return []
        steps = [s if s is not None else "" for s in steps]

        # If a step contains both "steam" and "salt", remove the salt fragment from that step only.
        cleaned_steps = []
        for s in steps:
            if not isinstance(s, str):
                s = str(s)
            low = s.lower()
            if "steam" in low and "salt" in low:
                # Remove patterns like "add salt and", "add salt,", "add salt" but preserve other salt occurrences
                s = re.sub(r'\badd\s+salt\s*(?:and\s*)?', '', s, flags=re.I)
                s = re.sub(r'\b(salt\s*,\s*and\s*)', '', s, flags=re.I)
                s = re.sub(r'\b(salt)\b(?=\s*(for|until|when|to|and|,|\.|$))', '', s, flags=re.I)
                # cleanup leftover punctuation/whitespace
                s = re.sub(r'\s+', ' ', s).strip()
                s = re.sub(r'\s+,', ',', s)
                if s and not s.endswith('.'):
                    s = s + '.'
                print("DEBUG: cleaned steam+salt step ->", repr(s))
            cleaned_steps.append(s)
        steps = cleaned_steps
        print("DEBUG: steps after per-step steam/salt cleanup =", steps)

        # treat only standalone soak/soaked as a soak step — ignore if other cooking verbs are present
        def is_pure_soak(s: str) -> bool:
            low = s.lower()
            if not re.search(r'\bsoak(?:ed)?\b', low):
                return False
            if re.search(r'\b(grind|mix|combine|spread|cook|fry|whisk|blend|pulse|beat|stir|bake|roast|saute)\b', low):
                return False
            return True

        # helper to split composite step into prep + cook if it contains both
        _prep_verbs = r'\b(beat|whisk|mix|combine|stir|fold|knead|blend|whisked|beaten)\b'
        _cook_verbs = r'\b(heat|cook|fry|sauté|saute|bake|roast|grill|steam|simmer)\b'

        def _split_prep_and_cook(raw_steps):
            out = []
            for idx, s in enumerate(raw_steps):
                low = s.lower()
                if re.search(_prep_verbs, low) and re.search(_cook_verbs, low):
                    parts = re.split(r'\b(?:then|and then|, then|;| and | then )\b', s, flags=re.IGNORECASE)
                    prep = None
                    cook = None
                    for p in parts:
                        p_stripped = p.strip()
                        if prep is None and re.search(_prep_verbs, p_stripped, re.IGNORECASE):
                            prep = p_stripped
                            continue
                        if cook is None and re.search(_cook_verbs, p_stripped, re.IGNORECASE):
                            cook = p_stripped
                    if prep and cook:
                        prep_line = prep if prep.endswith('.') else prep + '.'
                        cook_line = cook if cook.endswith('.') else cook + '.'
                        out.append(prep_line)
                        out.append(cook_line)
                        continue
                out.append(s)
            return out

        # split combined prep+cook lines early
        steps = _split_prep_and_cook(steps)
        print("DEBUG: steps after _split_prep_and_cook =", steps)

        # normalize / dedupe initial input (use your helper if available)
        norm_steps = []
        seen = set()
        for s in steps:
            if not s:
                continue
            try:
                s_norm = self._normalize_step_text(s)
            except Exception:
                # fallback minimal normalizer
                s_norm = re.sub(r'\s+', ' ', s.strip())
                if s_norm and not s_norm.endswith('.'):
                    s_norm += '.'
            key = s_norm.strip().lower()
            if key and key not in seen:
                seen.add(key)
                norm_steps.append(s_norm)
        print("DEBUG: norm_steps after initial normalization =", norm_steps)

        if not norm_steps:
            return []

        # preserve_combine heuristic
        preserve_combine = None
        dry_keep_keywords = ["flour", "gram flour", "besan", "all-purpose", "yeast", "salt", "egg", "eggs"]
        for s in norm_steps:
            low = s.lower()
            if any(k in low for k in ("combine", "whisk", "mix")) and any(dk in low for dk in dry_keep_keywords):
                preserve_combine = s
                break

        # detect batter-like step (flour + yogurt patterns)
        batter_step = None
        flour_pattern = r"(gram flour|besan|semolina|suji|maida|atta|rice|[a-z ]+flour)"
        yogurt_pattern = r"(yogurt|curd|dahi|yoghurt)"
        for s in norm_steps:
            low = s.lower()
            if any(v in low for v in ["mix", "whisk", "combine", "stir"]):
                if re.search(flour_pattern, low) and re.search(yogurt_pattern, low):
                    m_flour = re.search(flour_pattern, low)
                    m_yog = re.search(yogurt_pattern, low)
                    flour_txt = (m_flour.group(1) if m_flour else "flour").strip().title()
                    yog_txt = (m_yog.group(1) if m_yog else "yogurt").strip().title()
                    batter_step = f"Whisk the {flour_txt} and {yog_txt} together, adding water gradually to form a smooth batter."
                    break

        # scan for explicit add ingredients (water/eno/salt etc.)
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
            display_parts = [("Eno" if n == "eno" else n) for n in seen_add]
            if len(display_parts) == 1:
                list_txt = display_parts[0]
            else:
                list_txt = ", ".join(display_parts[:-1]) + " and " + display_parts[-1]
            add_step = f"Add {list_txt}. Mix gently until just combined."

        # detect cook/steam fallback
        cook_step = None
        for s in norm_steps:
            low = s.lower()
            if "steam" in low:
                m_time = re.search(r"(\d+)\s*(?:mins?|minutes?)", low)
                cook_step = f"Steam for {m_time.group(1)} minutes." if m_time else "Steam until cooked through."
                break
        if not cook_step and any("steam" in s.lower() for s in norm_steps):
            cook_step = "Steam until cooked through."

        # helper fallbacks (use your existing helpers if present)
        def _normalize_step_text_local(s: str) -> str:
            try:
                return self._normalize_step_text(s)
            except Exception:
                s2 = (s or "").strip()
                s2 = re.sub(r'\s+', ' ', s2)
                if s2 and not s2.endswith('.'):
                    s2 += '.'
                return s2

        def _normalize_for_dedupe_local(s: str) -> str:
            try:
                return self._normalize_for_dedupe(s)
            except Exception:
                s2 = s.lower()
                s2 = re.sub(r'[^a-z0-9\s]', ' ', s2)
                s2 = re.sub(r'\s+', ' ', s2).strip()
                return s2

        def classify_phase_local(text: str) -> str:
            try:
                return self.classify_phase(text)
            except Exception:
                t = text.lower()
                cook_keys = ['bake','roast','cook','fry','simmer','saute','steam']
                if any(k in t for k in cook_keys):
                    return 'cook'
                rest_keys = ['rest','rise','proof','ferment','hang','set']
                if any(k in t for k in rest_keys):
                    return 'rest'
                add_keys = ['add','mix','combine','stir','knead','whisk','grind','soak','soaked','beat']
                if any(k in t for k in add_keys):
                    return 'add'
                return 'other'

        def has_time_or_temp_local(text: str) -> bool:
            try:
                return self.has_time_or_temp(text)
            except Exception:
                pattern = r'\b(\d+\s*(?:-|\u2013)?\d*\s*(?:min|mins|minutes|h|hr|hour|hours)|\d+°C|\d+°F|for \d+|overnight)\b'
                return bool(re.search(pattern, text.lower()))

        def extract_hours(text: str):
            t = text.lower()
            if 'overnight' in t:
                return 12.0
            m_range = re.search(r'(\d+(?:\.\d*)?)\s*[-–to]+\s*(\d+(?:\.\d*)?)\s*(?:hours?|hrs?|h)\b', t)
            if m_range:
                return max(float(m_range.group(1)), float(m_range.group(2)))
            m_single = re.search(r'(\d+(?:\.\d*)?)\s*(?:hours?|hrs?|h)\b', t)
            if m_single:
                return float(m_single.group(1))
            m_num = re.search(r'\b(\d+(?:\.\d*)?)\b', t)
            if m_num:
                return float(m_num.group(1))
            return None

        # streaming: buffer 'add' lines
        merged = []
        add_buffer = []

        def flush_add_buffer():
            if not add_buffer:
                return
            joined = " ".join(add_buffer).strip()
            joined = re.sub(r'\s+', ' ', joined)
            if not joined.endswith('.'):
                joined += '.'
            norm_joined = _normalize_step_text_local(joined)
            merged.append(norm_joined)
            add_buffer.clear()

        for idx, s in enumerate(norm_steps):
            try:
                s_norm = _normalize_step_text_local(s)
            except Exception:
                s_norm = (s or "").strip()
                s_norm = re.sub(r'\s+', ' ', s_norm)
                if s_norm and not s_norm.endswith('.'):
                    s_norm += '.'

            phase = classify_phase_local(s_norm)
            try:
                protected = (phase in {'cook', 'rest', 'finish'}) or has_time_or_temp_local(s_norm)
            except Exception:
                protected = (phase in {'cook', 'rest', 'finish'})

            if phase == 'add' and not protected:
                add_buffer.append(s_norm)
                continue
            else:
                if add_buffer:
                    flush_add_buffer()
                merged.append(s_norm)

        if add_buffer:
            flush_add_buffer()

        # append fallback add_step/cook_step if absent
        if not any('add' in x.lower() for x in merged) and add_step:
            merged.append(_normalize_step_text_local(add_step))
        if cook_step and not any('steam' in x.lower() for x in merged):
            merged.append(_normalize_step_text_local(cook_step))

        # ensure soak before grind for key tokens
        ing_tokens = ['rice', 'urad', 'dal', 'semolina', 'besan', 'flour']
        for ing_tok in ing_tokens:
            try:
                soak_idx = next((i for i, s in enumerate(merged) if is_pure_soak(s) and ing_tok in s.lower()), None)
                grind_idx = next((i for i, s in enumerate(merged) if re.search(r'\bgrind\b', s.lower()) and ing_tok in s.lower()), None)
                if soak_idx is not None and grind_idx is not None and soak_idx > grind_idx:
                    moved_line = merged.pop(soak_idx)
                    merged.insert(grind_idx, moved_line)
            except Exception:
                pass

        # remove generic soaks if timed ones exist & collapse duplicates preferring informative
        # (Keep your detailed logic from original — keep minimal here for safety)
        final = []
        seen_keys = {}
        for s in merged:
            key = _normalize_for_dedupe_local(s)
            if key not in seen_keys:
                seen_keys[key] = s
                final.append(s)
            else:
                existing = seen_keys[key]
                score_existing = (1 if has_time_or_temp_local(existing) else 0) + (1 if 'soak' in existing.lower() else 0) + (len(existing) / 200.0)
                score_new = (1 if has_time_or_temp_local(s) else 0) + (1 if 'soak' in s.lower() else 0) + (len(s) / 200.0)
                if score_new > score_existing + 0.01:
                    idx = final.index(existing)
                    final[idx] = s
                    seen_keys[key] = s

        # compact duplicates preserving order
        compacted = []
        seen_keys2 = set()
        for s in final:
            k = _normalize_for_dedupe_local(s)
            if k in seen_keys2:
                continue
            seen_keys2.add(k)
            compacted.append(s)

        # final bucketize & reorder: soak -> grind -> add -> other -> cook
        soak_bucket, grind_bucket, add_bucket, other_bucket, cook_bucket = [], [], [], [], []
        for s in compacted:
            low = s.lower()
            if 'soak' in low:
                soak_bucket.append(s)
            elif 'grind' in low:
                grind_bucket.append(s)
            elif any(k in low for k in ['combine', 'add', 'mix', 'whisk', 'stir']):
                add_bucket.append(s)
            elif any(k in low for k in ['cook', 'fry', 'spread', 'bake', 'steam', 'roast']):
                cook_bucket.append(s)
            else:
                other_bucket.append(s)

        reordered = []
        reordered.extend(soak_bucket)
        grind_soaked = [g for g in grind_bucket if 'soak' in g.lower() or 'soaked' in g.lower()]
        grind_other = [g for g in grind_bucket if g not in grind_soaked]
        reordered.extend(grind_soaked + grind_other)
        reordered.extend(add_bucket)
        reordered.extend(other_bucket)
        reordered.extend(cook_bucket)

        # final dedupe preserving more informative steps
        final_ordered = []
        seen_final = set()
        for s in reordered:
            k = _normalize_for_dedupe_local(s)
            if k not in seen_final:
                seen_final.add(k)
                final_ordered.append(s)

        print("DEBUG: merge_semantic_steps FINAL =", final_ordered)
        return final_ordered





    def remove_invalid_leavening_from_steps(self, steps: List[str], ingredients: List[Ingredient]) -> List[str]:
        print("DEBUG: remove_invalid_leavening_from_steps START")
        print("DEBUG: steps input =", steps)
        print("DEBUG: ingredients =", [(i.name, i.quantity, i.unit) for i in ingredients])

        has_eno = any(i.name.lower() == "eno" for i in ingredients)
        has_soda = any(i.name.lower() in ["baking soda", "soda"] for i in ingredients)

        print(f"DEBUG: has_eno={has_eno}, has_soda={has_soda}")

        if has_eno and not has_soda:
            print("DEBUG: Rule triggered \u2014 ENO present, Baking Soda absent")
            cleaned = []
            for idx, s in enumerate(steps):
                print(f"DEBUG: cleaning step[{idx}] = {repr(s)}")
                s2 = s
                s2 = re.sub(r'\b(baking soda|soda)\b', '', s2, flags=re.I)
                s2 = re.sub(r'\band\s+and\b', 'and', s2, flags=re.I)
                s2 = re.sub(r'\b(and)\s*(?=[\.,;:])', '', s2, flags=re.I)
                s2 = re.sub(r'\band\s*$', '', s2, flags=re.I)
                before_strip = s2
                s2 = re.sub(r'\s+', ' ', s2).strip()
                print(f"DEBUG:   cleaned -> {repr(s2)} (before strip={repr(before_strip)})")
                if s2:
                    cleaned.append(s2)
            print("DEBUG: remove_invalid_leavening_from_steps OUTPUT =", cleaned)
            return cleaned

        print("DEBUG: No change \u2014 returning original steps")
        return steps

    #
    def canonicalize_step_text(self, text: str) -> str:
        print(f"DEBUG: canonicalize_step_text input={repr(text)}")

        out = text

        print("DEBUG: starting alias replacements...")

        for alias, canon in self.CANONICAL_NAMES.items():
            pattern = r'\b' + re.escape(alias) + r'\b'

            print(f"DEBUG:   checking alias '{alias}' with pattern '{pattern}'")

            new_out = re.sub(pattern, canon.title(), out, flags=re.I)

            if new_out != out:
                print(f"DEBUG:     replaced alias '{alias}' > '{canon.title()}'")
                print(f"DEBUG:     updated text: {repr(new_out)}")

            out = new_out

        print(f"DEBUG: canonicalize_step_text output={repr(out)}")
        print("DEBUG: END canonicalize_step_text\n")

        return out



    def normalize_leavening(self, ingredients: List[Ingredient]) -> List[Ingredient]:
        print("DEBUG: normalize_leavening START")
        print("DEBUG: ingredients input =", [(i.name, i.quantity, i.unit) for i in ingredients])

        has_eno = any(i.name.lower() == "eno" for i in ingredients)
        has_soda = any(i.name.lower() in ["baking soda", "soda"] for i in ingredients)

        print(f"DEBUG: has_eno={has_eno}, has_soda={has_soda}")

        if has_eno and has_soda: # Corrected from && to and
            print("DEBUG: removing baking soda/soda (ENO present)")
            ingredients = [i for i in ingredients if i.name.lower() not in ["baking soda", "soda"]]

        print("DEBUG: normalize_leavening output =", [(i.name, i.quantity, i.unit) for i in ingredients])
        return ingredients


    def merge_ingredients(self, recipes: List[Recipe], requested_servings: int) -> List[Ingredient]:
        print("DEBUG: merge_ingredients START")
        print(f"DEBUG: requested_servings={requested_servings}")
        print("DEBUG: recipes count =", len(recipes))

        grouped: Dict[str, Dict[str, Any]] = {}

        for r_idx, r in enumerate(recipes):
            print(f"DEBUG: processing recipe[{r_idx}] servings={r.servings}")
            for ing in r.ingredients:
                print(f"DEBUG:   ingredient={ing.name}, qty={ing.quantity}, unit={ing.unit}")

                cname = self.canonical_name(ing.name)
                key = cname.strip().lower()
                print(f"DEBUG:   canonical_name={cname}, key={key}")

                if key not in grouped:
                    grouped[key] = {"name": cname.strip(), "per_serving": [], "units": []}
                    print(f"DEBUG:   created new group for {key}")

                if r.servings <= 0:
                    raise ValueError("Source recipe has invalid servings")

                per_serving_val = ing.quantity / r.servings
                grouped[key]["per_serving"].append(per_serving_val)
                grouped[key]["units"].append(ing.unit)

                print(f"DEBUG:   added per_serving={per_serving_val}, unit={ing.unit}")

        print("DEBUG: grouped raw =", grouped)

        merged: List[Ingredient] = []
        for key, data in grouped.items():
            avg_per_serving = sum(data["per_serving"]) / len(data["per_serving"])
            final_qty = round(avg_per_serving * requested_servings, 3)
            unit = max(set(data["units"]), key=data["units"].count) if data["units"] else ""

            print(f"DEBUG: merging {key}: avg_per_serving={avg_per_serving}, final_qty={final_qty}, unit={unit}")

            merged.append(Ingredient(
                name=data["name"].title(),
                quantity=final_qty,
                unit=unit
            ))

        print("DEBUG: merged before leavening normalization =", [(i.name, i.quantity, i.unit) for i in merged])
        merged = self.normalize_leavening(merged)
        print("DEBUG: merged final output =", [(i.name, i.quantity, i.unit) for i in merged])

        return merged


    class FreeOpenLLM:
        """Adapter to call a local HuggingFace transformers pipeline for text2text-generation."""

        def __init__(self, model_name: str = 'google/flan-t5-base'):
            self.model_name = model_name
            self._pipe = None
            self._init_error = None
            print(f"DEBUG: FreeOpenLLM.__init__() start, model_name={model_name!r}")
            try:
                from transformers import (
                    pipeline,
                    T5ForConditionalGeneration,
                    T5Tokenizer
                )
                print("DEBUG: transformers imported successfully")

                print("DEBUG: loading tokenizer...")
                tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=False)
                print("DEBUG: tokenizer loaded")

                print("DEBUG: loading model...")
                model = T5ForConditionalGeneration.from_pretrained(model_name)
                print("DEBUG: model loaded")

                print("DEBUG: creating pipeline (device=-1 -> CPU)...")
                self._pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=(0 if (torch is not None and torch.cuda.is_available()) else -1))

                print("DEBUG: pipeline created successfully; _pipe set")
            except Exception as e:
                self._pipe = None
                self._init_error = e
                print("DEBUG: FreeOpenLLM.__init__() failed with exception:", repr(e))



        def available(self) -> bool:
            avail = self._pipe is not None
            print(f"DEBUG: FreeOpenLLM.available() -> {avail}")
            return avail

        def generate(self, prompt: str, **gen_kwargs) -> str:
            if not self.available():
                err = getattr(self, "_init_error", None)
                print("DEBUG: FreeOpenLLM.generate() called but pipe not available; raising RuntimeError. init_error =", repr(err))
                raise RuntimeError(f"LLM pipeline for {self.model_name} is not available. Init error: {err}")
            # Print truncated prompt for debugging (avoid huge dumps)
            try:
                truncated_prompt = (prompt[:1000] + '...') if len(prompt) > 1000 else prompt
            except Exception:
                truncated_prompt = "<unprintable prompt>"
            print("DEBUG: FreeOpenLLM.generate() called. prompt (truncated) =", truncated_prompt.replace("\n", "\\n"))
            print("DEBUG: gen_kwargs =", gen_kwargs)
            out = self._pipe(prompt, **gen_kwargs)
            print("DEBUG: raw pipeline output type:", type(out), "len(out) if list ->", (len(out) if isinstance(out, list) else "n/a"))
            if isinstance(out, list) and out:
                first = out[0]
                print("DEBUG: pipeline first element type:", type(first))
                if isinstance(first, dict):
                    generated_text = first.get('generated_text', str(first))
                    print("DEBUG: pipeline returned generated_text (truncated) =", (generated_text[:1000] + '...') if len(generated_text) > 1000 else generated_text)
                    return generated_text
                generated_str = str(first)
                print("DEBUG: pipeline returned first element as string (truncated) =", (generated_str[:1000] + '...') if len(generated_str) > 1000 else generated_str)
                return generated_str
            out_str = str(out)
            print("DEBUG: pipeline returned non-list output (truncated) =", (out_str[:1000] + '...') if len(out_str) > 1000 else out_str)
            return out_str

    @classmethod
    def classify_phase(cls, step: str) -> str:
        low = step.lower()
        # Treat explicit stir-fry as cooking even though 'stir' alone maps to mix
        if re.search(r'\bstir[- ]?fry\b', low):
            print(f"DEBUG: classify_phase('{step}') -> 'cook' (matched stir-fry pattern)")
            return 'cook'
        # Check cook/rest/finish BEFORE mix/prep to prioritize cooking actions
        # e.g., 'cook the beaten eggs' should be cook, not mix (even though it contains 'beat')
        for phase in ['cook', 'rest', 'finish', 'prep', 'mix']:
            keywords = cls.PHASE_KEYWORDS.get(phase, [])
            for kw in keywords:
                if kw in low:
                    print(f"DEBUG: classify_phase('{step}') -> '{phase}' (matched keyword '{kw}')")
                    return phase
        if re.search(r'\b(min|minute|minutes|hr|hour|°c|°f|degrees|°)\b', low):
            print(f"DEBUG: classify_phase('{step}') -> 'cook' (matched time/temperature pattern)")
            return 'cook'
        print(f"DEBUG: classify_phase('{step}') -> 'mix' (default)")
        return 'mix'
    #
    def reorder_steps(self, steps: List[str]) -> List[str]:
        print("\nDEBUG: START reorder_steps()")
        print("DEBUG: input steps =", steps)

        buckets: Dict[str, List[Tuple[int, str]]] = {'prep': [], 'mix': [], 'rest': [], 'cook': [], 'finish': []}

        for i, s in enumerate(steps):
            phase = self.classify_phase(s)
            print(f"DEBUG:   step[{i}] classified as phase '{phase}': {s}")
            buckets.setdefault(phase, []).append((i, s))

        print("\nDEBUG: buckets after classification:")
        for phase, items in buckets.items():
            print(f"DEBUG:   {phase}: {items}")

        ordered = []

        for phase in ['prep', 'mix', 'rest', 'cook', 'finish']:
            items = sorted(buckets.get(phase, []), key=lambda x: x[0])
            print(f"\nDEBUG: ordering phase '{phase}' (items={items})")

            ordered.extend([s for _, s in items])

        result = ordered if ordered else steps

        print("\nDEBUG: reorder_steps() output =", result)
        print("DEBUG: END reorder_steps()\n")

        return result


    @staticmethod
    def has_time_or_temp(text: str) -> bool:
        res = bool(re.search(r'\b(\d+\s?(mins?|minutes?|hrs?|hours?|°\s?[CF]|°C|°F|degrees))\b', text, flags=re.I))
        print(f"DEBUG: has_time_or_temp('{text}') -> {res}")
        return res

    def compute_ai_confidence(self, num_sources: int, steps: List[str], generated_text: str) -> float:
        print("DEBUG: compute_ai_confidence() start", "num_sources=", num_sources, "len(steps)=", len(steps))
        base = 0.45
        src_bonus = min(0.25, 0.08 * num_sources)
        step_bonus = min(0.2, 0.02 * len(steps))
        time_bonus = 0.15 if any(self.has_time_or_temp(s) for s in steps) else 0.0
        length_penalty = 0.0
        if len(generated_text.split()) < 30:
            length_penalty = 0.1
        conf = base + src_bonus + step_bonus + time_bonus - length_penalty
        conf = round(max(0.0, min(0.99, conf)), 3)
        print(f"DEBUG: compute_ai_confidence() computed -> base={base}, src_bonus={src_bonus}, step_bonus={step_bonus}, time_bonus={time_bonus}, length_penalty={length_penalty}, conf={conf}")
        return conf

    def _strip_leading_number_prefixes(self, lines: List[str]) -> List[str]:

        return [re.sub(r'^\s*(?:step\s*)?\d+[\:\.\)]\s*', ' ', s, flags=re.I) for s in lines]


    #
    def synthesize(self, top_recipes: List[Recipe], requested_servings: int,
               llm_model: str = 'google/flan-t5-base', reorder: bool = True) -> Recipe:

        print("\nDEBUG: ===================== synthesize() START =====================")
        print(f"DEBUG: requested_servings = {requested_servings}")
        print(f"DEBUG: # of top_recipes    = {len(top_recipes)}")
        print(f"DEBUG: llm_model           = {llm_model}")
        print(f"DEBUG: reorder steps?      = {reorder}")

        if not top_recipes:
            raise ValueError("No recipes provided for synthesis")

        # treat only standalone soak/soaked as a soak step — ignore if other cooking verbs are present
        def is_pure_soak(s: str) -> bool:
            low = s.lower()
            # must contain soak / soaked
            if not re.search(r'\bsoak(?:ed)?\b', low):
                return False
            # if any other cooking verb exists in the same sentence, don't treat as a pure soak
            if re.search(r'\b(grind|mix|combine|spread|cook|fry|whisk|blend|pulse|beat|stir|bake|roast|saute)\b', low):
                return False
            return True


        # ---- Merge ingredients ----
        print("\nDEBUG: calling merge_ingredients()")
        merged_ings = self.merge_ingredients(top_recipes, requested_servings)
        print("DEBUG: merged_ings =", [asdict(ing) for ing in merged_ings])

        # ---- Generate prep from ingredients ----
        print("\nDEBUG: calling generate_prep_from_ingredients()")
        prep_from_ings = self.generate_prep_from_ingredients(merged_ings)
        print("DEBUG: prep_from_ings =", prep_from_ings)

        # ---- Normalize and canonicalize steps ----
        print("\nDEBUG: normalizing and canonicalizing recipe steps")
        raw_steps = []

        for r_index, r in enumerate(top_recipes):
            print(f"DEBUG:  processing recipe[{r_index}] with {len(r.steps)} steps")

            for s_index, s in enumerate(r.steps):
                print(f"DEBUG:    raw step[{s_index}] = {repr(s)}")

                # Normalize
                s_norm = self._normalize_step_text(s)
                print(f"DEBUG:      after normalize = {repr(s_norm)}")

                # Canonicalize
                s_norm = self.canonicalize_step_text(s_norm)
                print(f"DEBUG:      after canonicalize = {repr(s_norm)}")

                raw_steps.append(s_norm)

        # ---- Combine prep + steps ----
        raw_steps = prep_from_ings + raw_steps
        print("\nDEBUG: raw_steps (combined) =", raw_steps)

        # ---- Dump steps as bullet list for debugging ----
        src = "\n".join(f"- {s}" for s in raw_steps)
        print("\nDEBUG: raw_steps as bullet list:")
        print(src)


        prompt = (
              f"Combine the following cooking actions into one clear, merged recipe for {requested_servings} servings.\n\n"
              "Write 4-8 numbered steps. Keep steps short (one sentence each). Do NOT add new ingredients or quantities.\n"
              "Include times or temperatures when they appear in the source actions.\n\n"
              f"Source actions:\n{src}\n\n"
              "Output only numbered steps, starting strictly with:\n"
              "1. <step>\n"
              "2. <step>\n"
              "3. <step>\n"
              "...\n\n"
              "Do NOT output anything before step 1.\n"
              "Begin your answer with: 1. "
          )

        print("DEBUG: prompt constructed (truncated):", prompt[:400].replace("\n", "\\n"))

        llm = self.FreeOpenLLM(model_name=llm_model)
        print("DEBUG: llm available?", llm.available(), "llm init error:", getattr(llm, "_init_error", None))
        if not llm.available():
            print("DEBUG: entering fallback (no llm) path")
            #
            fallback_steps = []
            seen = set()

            print("DEBUG: raw_steps entering fallback block =", raw_steps)

            for s in raw_steps:
                print(f"\nDEBUG: processing raw step: {repr(s)}")

                s_clean = re.sub(r'\s+', ' ', s).strip()
                print(f"DEBUG:   cleaned step = {repr(s_clean)}")

                key = s_clean.lower()

                if key not in seen:
                    print("DEBUG:   > new step, keeping")
                    seen.add(key)
                    fallback_steps.append(s_clean)
                else:
                    print("DEBUG:   > duplicate step, skipping")

            print("\nDEBUG: fallback_steps deduped =", fallback_steps)

            # Take top 6 or fall back to default line
            out_lines = fallback_steps[:6] if fallback_steps else ["Combine ingredients and cook as directed."]

            print("DEBUG: fallback initial out_lines =", out_lines)

            # Reorder if enabled
            if reorder:
                print("\nDEBUG: calling reorder_steps() on fallback out_lines")
                out_lines = self.reorder_steps(out_lines)
                print("DEBUG: out_lines after reorder =", out_lines)
            else:
                print("DEBUG: reorder disabled, keeping order as-is")

            # after reorder_steps -> conservative fix:
            # If any 'soak' step appears AFTER a 'grind' step that mentions same ingredient, move soak earlier.
            for ing_tok in ['rice','urad','dal','semolina','besan','flour']:
                print(f"\nDEBUG: Checking ingredient token '{ing_tok}'")

                soak_idx = next(
                    (i for i, s in enumerate(out_lines)
                    if is_pure_soak(s) and ing_tok in s.lower()),
                    None
                )
                grind_idx = next(
                    (i for i, s in enumerate(out_lines)
                    if 'grind' in s.lower() and ing_tok in s.lower()),
                    None
                )

                print(f"DEBUG:   soak_idx={soak_idx}, grind_idx={grind_idx}")

                # Check if reorder is needed
                if soak_idx is not None and grind_idx is not None:
                    print(f"DEBUG:   order check soak_idx({soak_idx}) > grind_idx({grind_idx}) ? {soak_idx > grind_idx}")

                if soak_idx is not None and grind_idx is not None and soak_idx > grind_idx:
                    print("DEBUG:   > moving soak line above grind line")

                    line = out_lines.pop(soak_idx)
                    print(f"DEBUG:   popped soak line: {line}")

                    new_pos = max(0, grind_idx)
                    out_lines.insert(new_pos, line)
                    print(f"DEBUG:   inserted at index {new_pos}")

                else:
                    print("DEBUG:   no reorder needed for this token")

            print("\nDEBUG: out_lines before merge_semantic_steps =", out_lines)
            print("DEBUG: calling merge_semantic_steps()")

            out_lines = self.merge_semantic_steps(out_lines)
            print("DEBUG: out_lines after merge_semantic_steps =", out_lines)

            print("\nDEBUG: calling remove_invalid_leavening_from_steps()")
            out_lines = self.remove_invalid_leavening_from_steps(out_lines, merged_ings)
            print("DEBUG: out_lines after remove_invalid_leavening_from_steps =", out_lines)

            # ensure prep lines survive
            def weighted_jaccard_similarity(a_tokens: set, b_tokens: set, token_weights: dict | None = None) -> float:
                if not a_tokens and not b_tokens:
                    return 1.0
                if not a_tokens or not b_tokens:
                    return 0.0

                default_weights = {
                    # ingredients (examples)
                    "rice": 2.0, "urad": 2.0, "dal": 2.0, "flour": 1.8, "semolina": 1.8, "besan": 1.8,
                    # key cooking verbs / phases
                    "soak": 1.7, "grind": 1.7, "mix": 1.5, "combine": 1.5, "cook": 1.7, "spread": 1.3,
                    "ferment": 1.8, "batter": 1.6, "drain": 1.3, "beat": 1.5, "whisk": 1.5,
                    # wet/dry indicators & common tokens
                    "salt": 1.4, "oil": 1.4, "water": 1.4, "yogurt": 1.6, "eno": 2.0,
                    # time/temperature tokens
                    "hours": 1.5, "overnight": 1.6, "minutes": 1.4, "preheat": 1.5
                }
                weights = token_weights or default_weights

                def w(tok: str) -> float:
                    return float(weights.get(tok, 1.0))

                inter = a_tokens & b_tokens
                union = a_tokens | b_tokens

                w_inter = sum(w(t) for t in inter)
                w_union = sum(w(t) for t in union)

                if w_union <= 0:
                    return 0.0
                return w_inter / w_union

            if prep_from_ings:
                print("\nDEBUG: Normalizing prep_from_ings")
                prep_normed = [self._normalize_step_text(p) for p in prep_from_ings]
                print("DEBUG: prep_normed =", prep_normed)

                to_prepend = []
                #
                # --- REPLACE existing similarity-check loop with this enhanced logic ---
                for p in prep_normed:
                    print(f"\nDEBUG: evaluating prep step: {p}")

                    p_key = self._token_set(p)
                    print("DEBUG:   p_key =", p_key)

                    if not p_key:
                        print("DEBUG:   empty token set > skipping")
                        continue

                    exists_similar = False

                    # check similarity with both out_lines & pending prep
                    for s_exist in (to_prepend + out_lines):
                        s_key = self._token_set(s_exist)
                        if not s_key:
                            continue

                        # use weighted similarity instead of plain Jaccard
                        sim = weighted_jaccard_similarity(p_key, s_key)
                        print(f"DEBUG:     compare with: {s_exist}  sim={sim:.3f}")

                        if sim >= 0.40:
                            # SPECIAL CASE: prefer keeping an explicit 'grind soaked' prep line
                            # even when it has high overlap with a 'soak'-only line.
                            p_low = p.lower()
                            s_low = s_exist.lower()

                            p_has_grind = bool(re.search(r'\bgrind\b', p_low))
                            p_has_soak  = bool(re.search(r'\bsoak(?:ed)?\b', p_low))
                            s_has_grind = bool(re.search(r'\bgrind\b', s_low))
                            s_has_soak  = bool(re.search(r'\bsoak(?:ed)?\b', s_low))

                            # If p is a grind-of-soaked-ingredients and s_exist is only a soak (no grind),
                            # keep the prep line (do not treat as duplicate).
                            if p_has_grind and p_has_soak and (s_has_soak and not s_has_grind):
                                print("DEBUG:     special-case keep: prep contains 'grind'+'soak' while existing is soak-only -> keep prep")
                                # treat as NOT similar for the purposes of skipping
                                continue

                            # If p explicitly mentions 'soaked' or 'soak' and s_exist is more generic,
                            # prefer the more explicit one (keep p).
                            if (p_has_soak and not s_has_soak) and p_has_grind:
                                print("DEBUG:     special-case keep: prep mentions soaked+grind but existing doesn't mention soak -> keep prep")
                                continue

                            # Otherwise, treat as similar and skip
                            print("DEBUG:     > similar (>=0.40), skipping")
                            exists_similar = True
                            break

                    if not exists_similar:
                        print("DEBUG:   > adding to prepend list")
                        to_prepend.append(p)
                    else:
                        print("DEBUG:   > NOT adding (duplicate-ish)")



                print("\nDEBUG: to_prepend BEFORE soak-sort =", to_prepend)

                # ensure soak lines come before grind lines
                to_prepend.sort(key=lambda s: (0 if 'soak' in s.lower() else 1))

                print("DEBUG: to_prepend AFTER soak-sort =", to_prepend)

                # prepend
                out_lines = to_prepend + out_lines
                print("DEBUG: out_lines after prep prepend =", out_lines)

            # Collapse repeated words
            print("\nDEBUG: collapsing repeated words...")
            out_lines = [self._collapse_repeated_words(s).strip() for s in out_lines]
            print("DEBUG: out_lines after collapse =", out_lines)

            # Ensure ingredient coverage
            print("\nDEBUG: calling ensure_ingredient_coverage()")
            out_lines = self.ensure_ingredient_coverage(out_lines, merged_ings)
            print("DEBUG: out_lines after ensure_ingredient_coverage =", out_lines)

            # --- Remove redundant/weak 'ferment' mentions if an explicit fermentation exists ---
            # --- Remove duplicate/weak 'ferment' mentions when an explicit-duration ferment exists ---
            def _has_explicit_duration(text: str) -> bool:
                return bool(re.search(r'\b(overnight|overnight\.?|for\s+\d+\s*(hours?|hrs?|h|minutes?|mins?)|at\s*\d+°[CF])\b', text, flags=re.I))

            # identify explicit ferment steps
            explicit_ferment_indices = [i for i, s in enumerate(out_lines) if re.search(r'\bferment(?:ed|ing)?\b', s, flags=re.I) and _has_explicit_duration(s)]

            if explicit_ferment_indices:
                # keep the first explicit-duration ferment; remove 'weak' ferment mentions from other steps
                explicit_idx = explicit_ferment_indices[0]
                cleaned = []
                for i, s in enumerate(out_lines):
                    if i == explicit_idx:
                        cleaned.append(s)  # keep explicit one
                        continue
                    # detect weak/conditional ferment mentions
                    if re.search(r'\bferment(?:ed|ing)?\b', s, flags=re.I):
                        # common weak phrases we want to remove or strip
                        # If the line also contains other important verbs (like 'grind', 'knead', etc.)
                        # we attempt to strip just the ferment phrase; otherwise drop the ferment-only phrase.
                        if re.search(r'\b(grind|mix|combine|knead|stir|beat|whisk|fold)\b', s, flags=re.I):
                            # remove the fragment containing 'ferment' and surrounding qualifiers
                            s2 = re.sub(r'[,;]?\s*(and\s+)?(?:may\s+)?(?:then\s+)?(?:allow\s+to\s+)?(?:to\s+)?ferment(?:\s+if\s+required|\s+if\s+needed|(?:\s+for\s+[^\.,;]+)?)?', '', s, flags=re.I).strip()
                            # clean leftover punctuation/spacing
                            s2 = re.sub(r'\s{2,}', ' ', s2).strip(' ,;.')
                            if s2:
                                cleaned.append(s2 + ('.' if not s2.endswith('.') else ''))
                            # else drop line
                        else:
                            # line is largely about ferment -> drop it (explicit duration exists elsewhere)
                            # (Optional: you could keep conditional if it adds unique tokens — but we drop here)
                            continue
                    else:
                        cleaned.append(s)
                out_lines = cleaned



            # Final aggressive dedupe
            print("\nDEBUG: calling _dedupe_steps()")
            out_lines = self._dedupe_steps(out_lines)
            print("DEBUG: out_lines after _dedupe_steps =", out_lines)

            # --- FINAL SAFETY NORMALIZATION: ensure soak comes before grind ---
            final_lines = out_lines.copy()
            print("DEBUG: working copy final_lines =", final_lines)

            for tok in ['rice', 'urad', 'dal', 'semolina', 'besan', 'flour']:
                print(f"\nDEBUG: Checking token '{tok}' for soak/grind order fix")

                soak_idx = next(
                    (i for i, s in enumerate(final_lines)
                    if is_pure_soak(s) and tok in s.lower()),
                    None
                )
                grind_idx = next(
                    (i for i, s in enumerate(final_lines)
                    if re.search(r'\bgrind\b', s.lower()) and tok in s.lower()),
                    None
                )

                print(f"DEBUG:   soak_idx={soak_idx}, grind_idx={grind_idx}")

                if soak_idx is not None and grind_idx is not None:
                    print(f"DEBUG:   order check soak_idx({soak_idx}) > grind_idx({grind_idx}) ? {soak_idx > grind_idx}")

                if soak_idx is not None and grind_idx is not None and soak_idx > grind_idx:
                    print("DEBUG:   > moving soak line before grind line")

                    line = final_lines.pop(soak_idx)
                    print(f"DEBUG:   popped line: {line}")

                    final_lines.insert(grind_idx, line)
                    print(f"DEBUG:   inserted at index {grind_idx}")

                else:
                    print("DEBUG:   no fix needed for this token")

            out_lines = final_lines
            print("\nDEBUG: out_lines after final_lines reorder =", out_lines)

            # ensure fermentation happens before any cooking/steaming (robust: handle multiple ferment lines)
            cook_idx = next(
                (i for i, s in enumerate(out_lines)
                if any(k in s.lower() for k in ('steam', 'cook', 'bake', 'fry', 'grill', 'roast', 'simmer'))),
                None
            )
            ferment_idxs = [i for i, s in enumerate(out_lines) if re.search(r'\bferment(?:ed|ing)?\b', s.lower())]

            if cook_idx is not None and ferment_idxs:
                # move ferment lines that are *after* the first cook index to just before cook_idx,
                # preserving the original relative order of ferment lines.
                # Iterate over a copy of indices so popping doesn't break iteration; adjust cook_idx as we insert.
                for fi in sorted(ferment_idxs):
                    if fi > cook_idx:
                        line = out_lines.pop(fi)
                        out_lines.insert(cook_idx, line)
                        cook_idx += 1  # keep cook_idx after inserted ferment so subsequent inserts stay before cook



            generated_text = "\n".join(out_lines)

            print("\nDEBUG: computing AI confidence scores")
            ai_conf = self.compute_ai_confidence(len(top_recipes), out_lines, generated_text)
            validator_conf = round(min(1.0, ai_conf * 0.8), 3)

            print(f"DEBUG: fallback ai_conf={ai_conf}, validator_conf={validator_conf}")

            title_base = top_recipes[0].title.split(':')[0].strip()
            title = f"Synthesized \u2014 {title_base} (for {requested_servings} servings)"

            meta = {
                "sources": [r.id for r in top_recipes],
                "ai_confidence": ai_conf,
                "synthesis_method": "fallback:no-llm"
            }

            print("DEBUG: returning fallback Recipe with meta =", meta)

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

        # LLM generation section
        gen_kwargs = {
            "max_new_tokens": 180,
            "do_sample": True,
            "temperature": 0.35,
            "top_p": 0.9,
            "repetition_penalty": 1.2,       # discourages repeating phrases
            "no_repeat_ngram_size": 3,
        }

        def _strip_leading_number_prefixes(self, lines: List[str]) -> List[str]:

            return [re.sub(r'^\s*(?:step\s*)?\d+[\:\.\)]\s*', ' ', s, flags=re.I) for s in lines]

        #
        def sanitize_llm_output(raw_text: str, source_actions: List[str]) -> str:
            """
            Clean LLM generated_text so parser doesn't pick up template placeholders.
            Returns either cleaned lines joined by '\n' OR a numbered fallback built from source_actions
            when nothing useful remains.
            """
            if not raw_text:
                raw_text = ""

            text = raw_text if isinstance(raw_text, str) else str(raw_text)

            # Normalise newlines and collapse long blank regions
            text = text.replace('\r', '\n')
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = text.strip()

            # Remove obvious template / placeholder tokens and simple tags
            text = re.sub(r'\{\{.*?\}\}', ' ', text)                 # handlebars
            text = re.sub(r'<\s*<.*?>', ' ', text)                   # leading weird << ...>
            text = re.sub(r'[<>]{2,}', ' ', text)                    # long runs of < or >
            text = re.sub(r'<[^>\n]{0,60}>', ' ', text)              # simple angle-tag content
            text = re.sub(r'\b(step[>\:\s]*|<step>|<\s*step\s*>|template|placeholder)\b', ' ', text, flags=re.I)

            # collapse repeated angle bracket remnants / stray punctuation
            text = re.sub(r'[\u00A0\s]+', ' ', text).strip()

            # Split to lines and filter
            raw_lines = [ln.strip() for ln in re.split(r'[\n\r]+', text) if ln.strip()]
            lines = []
            for raw_line in raw_lines:
                line = raw_line.strip()

                if not line:
                    continue

                # Reject any line containing raw angle-brackets (these are usually template artifacts)
                if '<' in line or '>' in line or '&lt;' in line or '&gt;' in line:
                    continue

                # Reject lines with too many non-alphanumeric characters
                if re.fullmatch(r'[\W_]+', line):
                    continue

                # Require at least 3 words to be considered an actionable step
                if len(re.findall(r'\b\w+\b', line)) < 3:
                    continue

                # Reject lines that look like repeated short tokens (e.g., "< < step", ">> >>")
                toks = re.findall(r'\S+', line)
                token_counts = {}
                for t in toks:
                    token_counts[t] = token_counts.get(t, 0) + 1
                if toks and any(cnt > (len(toks) // 2) and len(tok) <= 3 for tok, cnt in token_counts.items()):
                    continue

                # Good line — normalize internal whitespace and keep
                normalized = re.sub(r'\s+', ' ', line).strip()
                lines.append(normalized)

            # Remove adjacent duplicates
            deduped = []
            last = None
            for l in lines:
                if l == last:
                    continue
                deduped.append(l)
                last = l

            # If nothing meaningful remains, build a numbered fallback from source_actions
            if not deduped:
                fallback = []
                for i, s in enumerate(source_actions, start=1):
                    s_clean = (s or "").strip()
                    if not s_clean:
                        continue
                    if not s_clean.endswith(('.', '!', '?')):
                        s_clean = s_clean.rstrip('.;') + '.'
                    fallback.append(f"{i}. {s_clean}")
                # If even source_actions is empty, give a very conservative single-line fallback
                if not fallback:
                    return "1. Combine the ingredients and cook as directed."
                return "\n".join(fallback)

            return "\n".join(deduped)


        #
        print("\nDEBUG: calling llm.generate with gen_kwargs =", gen_kwargs)
        _generated_raw = llm.generate(prompt, **gen_kwargs)

        # Ensure we have a string
        _generated_raw = _generated_raw if isinstance(_generated_raw, str) else str(_generated_raw)
        print("DEBUG: raw pipeline output (truncated) =", (_generated_raw[:1000] + '...') if len(_generated_raw) > 1000 else _generated_raw)

        # Sanitize model output to remove template placeholders / angle-bracket junk.
        # If sanitizer returns nothing meaningful, it will return a numbered fallback built from raw_steps.
        _sanitized = sanitize_llm_output(_generated_raw, raw_steps)

        # Minor extra cleanup (preserve internal newlines for numeric parsing)
                # Minor extra cleanup (preserve internal newlines for numeric parsing)
        generated = re.sub(
            r'^[`\-"\']*\s*(?:step[>\:\s]*|sure[>\:\s]*|ok[>\:\s]*|got it[>\:\s]*|\banswer\b[>\:\s]*)',
            '',
            _sanitized,
            flags=re.I
        ).strip()

        # --- NEW: normalize 'Step N:' / 'Step N.' / 'Step N -' -> 'N:' (case-insensitive)
        # This fixes outputs like "Step 1: Do X." which previously broke strict parser expectations.
        generated = re.sub(r'(?i)\bstep\s*(\d+)\s*(?:[:\-\.\)])', r'\1:', generated)
        # Also normalize "1) " -> "1:" and "1 : " -> "1:"
        generated = re.sub(r'(?m)^\s*(\d+)\s*[\)\:]\s*', r'\1: ', generated)
        generated = re.sub(r'(?m)^\s*(\d+)\s*\.\s*', r'\1: ', generated)
        # ensure inline numbered items like "1: ... 2: ... 3: ..." become one-number-per-line
        generated = re.sub(r'(?<!\n)(?<=\S)\s+(\d+)\s*[:\.]\s*', r'\n\1: ', generated)


        print("DEBUG: raw generated output (after sanitize + numbering normalize, truncated) =", (generated[:1000] + '...') if len(generated) > 1000 else generated)

        # --- robust parsing of LLM output (handles numbered items like "1: ...", "1." and optional free-form lead)
        # Accept "1:" and "1." as numbering tokens
        pattern = r'^\s*(\d+)[\:\.]\s*(.+?)(?=\n\s*\d+[\:\.]|\Z)'
        matches = re.findall(pattern, generated, flags=re.S | re.M)
        print("DEBUG: regex matches found =", matches)

        out_lines = []

        # 1) If there's text BEFORE the first numbered token (e.g. "Beat eggs...\n2: Heat..."),
        #    allow optional leading "Step" in the match so we catch "Step 1:" variants longer than a line
        m_lead = re.match(r'^(.*?)\n\s*(?:Step\s*)?\d+[\:\.]', generated, flags=re.S | re.I)
        print("DEBUG: regex match m_lead =", m_lead)


        if m_lead:
            lead_text = m_lead.group(1)
            print("DEBUG: extracted raw lead_text =", repr(lead_text))

            lead_text = re.sub(r'\s+', ' ', lead_text).strip()
            print("DEBUG: normalized lead_text =", repr(lead_text))

            if lead_text:
                leading_sents = re.split(r'(?<=[\.\?\!])\s+', lead_text)
                print("DEBUG: split leading_sents =", leading_sents)

                for s in leading_sents:
                    print(f"\nDEBUG: evaluating leading sentence candidate: {repr(s)}")

                    s = s.strip()
                    print("DEBUG:   stripped =", repr(s))

                    if not s:
                        print("DEBUG:   empty > skip")
                        continue

                    if not s.endswith(('.', '!', '?')):
                        print("DEBUG:   missing punctuation > adding '.'")
                        s = s + '.'

                    word_count = len(s.split())
                    print(f"DEBUG:   word_count={word_count}")

                    if word_count >= 3:
                        print("DEBUG:   > accepted, appending to out_lines")
                        out_lines.append(s)
                    else:
                        print("DEBUG:   > rejected (too short)")
        else:
            print("DEBUG: no leading block found before numbered list")

        # 2) Now append numbered matches (sorted by their numeric label so order is stable)
        if matches:
            print("DEBUG: matches FOUND > sorting by numeric index")

            # sort in numeric order
            matches_sorted = sorted(matches, key=lambda x: int(x[0]))
            print("DEBUG: matches_sorted =", matches_sorted)

            for idx, text in matches_sorted:
                print(f"\nDEBUG: processing numbered step {idx}: {repr(text)}")

                # normalize whitespace
                cleaned = re.sub(r'\s+', ' ', text).strip()
                print(f"DEBUG:   cleaned = {repr(cleaned)}")

                # ensure punctuation ending
                if not cleaned.endswith(('.', '!', '?')):
                    print("DEBUG:   missing punctuation > adding '.'")
                    cleaned = cleaned + '.'

                # basic length filter
                wc = len(cleaned.split())
                print(f"DEBUG:   word_count = {wc}")

                if wc >= 3:
                    print("DEBUG:   > accepted, adding to out_lines")
                    out_lines.append(cleaned)
                else:
                    print("DEBUG:   > rejected (too short)")

        else:
            print("DEBUG: no matches > skipping numbered step parsing")

        print("\nDEBUG: out_lines after parsing numbered/un-numbered steps =", out_lines)

        # --- handle case where model started numbering at 2 (or omitted leading "1.") using the already-captured matches ---

        if matches:
            print("DEBUG: matches found > sorting numerically")

            # convert matches into text list in numeric order
            matches_sorted = sorted(matches, key=lambda x: int(x[0]))
            print("DEBUG: matches_sorted =", matches_sorted)

            numbered_texts = []

            for idx, text in matches_sorted:
                print(f"\nDEBUG: processing numbered item {idx}: raw={repr(text)}")

                cleaned = re.sub(r'\s+', ' ', text).strip()
                print("DEBUG:   cleaned text =", repr(cleaned))

                if not cleaned.endswith(('.', '!', '?')):
                    print("DEBUG:   missing punctuation > adding '.'")
                    cleaned += '.'

                wc = len(cleaned.split())
                print("DEBUG:   word count =", wc)

                if wc >= 3:
                    print("DEBUG:   > accepted numbered item")
                    numbered_texts.append(cleaned)
                else:
                    print("DEBUG:   > rejected numbered item (too short)")

            print("\nDEBUG: numbered_texts =", numbered_texts)

            # if leading free text exists
            if m_lead:
                print("DEBUG: m_lead exists > append numbered items after leading text (avoid duplicates)")

                existing_lower = {s.lower() for s in out_lines}
                print("DEBUG: existing lowercased =", existing_lower)

                for t in numbered_texts:
                    if t.lower() not in existing_lower:
                        print("DEBUG:   appending numbered text:", t)
                        out_lines.append(t)
                    else:
                        print("DEBUG:   skipping duplicate numbered text:", t)

            else:
                print("DEBUG: no m_lead > using numbered_texts as core content")
                out_lines = numbered_texts.copy()

        else:
            print("DEBUG: no matches > skipping numbered text logic")

        print("DEBUG: final out_lines after numbered+lead merge =", out_lines)

        # fallback sentence-splitting if still empty
        if not out_lines:
            gen_clean = re.sub(r'\s+', ' ', generated).strip()
            sentences = re.split(r'(?<=[\.\?\!])\s+', gen_clean)
            short_sentences = [s.strip().rstrip('.') + '.' for s in sentences if len(s.split()) >= 3]
            out_lines = short_sentences[:8]
            print("DEBUG: out_lines from sentence-split fallback =", out_lines)


        if not out_lines:
            raise RuntimeError("Model failed to produce any usable steps.")

        out_lines = [' '.join(s.split()) for s in out_lines]
        out_lines = [self.canonicalize_step_text(s) for s in out_lines]
        print("DEBUG: out_lines after canonicalize_step_text =", out_lines)

        out_lines = self._strip_leading_number_prefixes(out_lines)
        print("DEBUG: out_lines after stripping leading numbers =", out_lines)

        if reorder:
            out_lines = self.reorder_steps(out_lines)
            print("DEBUG: out_lines after reorder =", out_lines)
        out_lines = self.merge_semantic_steps(out_lines)
        print("DEBUG: out_lines after merge_semantic_steps =", out_lines)
        out_lines = self.remove_invalid_leavening_from_steps(out_lines, merged_ings)
        print("DEBUG: out_lines after remove_invalid_leavening_from_steps =", out_lines)

        # ensure prep lines survive
        def weighted_jaccard_similarity(a_tokens: set, b_tokens: set, token_weights: dict | None = None) -> float:
            if not a_tokens and not b_tokens:
                return 1.0
            if not a_tokens or not b_tokens:
                return 0.0

            default_weights = {
                # ingredients (examples)
                "rice": 2.0, "urad": 2.0, "dal": 2.0, "flour": 1.8, "semolina": 1.8, "besan": 1.8,
                # key cooking verbs / phases
                "soak": 1.7, "grind": 1.7, "mix": 1.5, "combine": 1.5, "cook": 1.7, "spread": 1.3,
                "ferment": 1.8, "batter": 1.6, "drain": 1.3, "beat": 1.5, "whisk": 1.5,
                # wet/dry indicators & common tokens
                "salt": 1.4, "oil": 1.4, "water": 1.4, "yogurt": 1.6, "eno": 2.0,
                # time/temperature tokens
                "hours": 1.5, "overnight": 1.6, "minutes": 1.4, "preheat": 1.5
            }
            weights = token_weights or default_weights

            def w(tok: str) -> float:
                return float(weights.get(tok, 1.0))

            inter = a_tokens & b_tokens
            union = a_tokens | b_tokens

            w_inter = sum(w(t) for t in inter)
            w_union = sum(w(t) for t in union)

            if w_union <= 0:
                return 0.0
            return w_inter / w_union

        if prep_from_ings:
            print("\nDEBUG: Normalizing prep_from_ings")
            prep_normed = [self._normalize_step_text(p) for p in prep_from_ings]
            print("DEBUG: prep_normed =", prep_normed)

            to_prepend = []
            #
            # --- REPLACE existing similarity-check loop with this enhanced logic ---
            for p in prep_normed:
                print(f"\nDEBUG: evaluating prep step: {p}")

                p_key = self._token_set(p)
                print("DEBUG:   p_key =", p_key)

                if not p_key:
                    print("DEBUG:   empty token set > skipping")
                    continue

                exists_similar = False

                # check similarity with both out_lines & pending prep
                for s_exist in (to_prepend + out_lines):
                    s_key = self._token_set(s_exist)
                    if not s_key:
                        continue

                    # use weighted similarity instead of plain Jaccard
                    sim = weighted_jaccard_similarity(p_key, s_key)
                    print(f"DEBUG:     compare with: {s_exist}  sim={sim:.3f}")

                    if sim >= 0.40:
                        # SPECIAL CASE: prefer keeping an explicit 'grind soaked' prep line
                        # even when it has high overlap with a 'soak'-only line.
                        p_low = p.lower()
                        s_low = s_exist.lower()

                        p_has_grind = bool(re.search(r'\bgrind\b', p_low))
                        p_has_soak  = bool(re.search(r'\bsoak(?:ed)?\b', p_low))
                        s_has_grind = bool(re.search(r'\bgrind\b', s_low))
                        s_has_soak  = bool(re.search(r'\bsoak(?:ed)?\b', s_low))

                        # If p is a grind-of-soaked-ingredients and s_exist is only a soak (no grind),
                        # keep the prep line (do not treat as duplicate).
                        if p_has_grind and p_has_soak and (s_has_soak and not s_has_grind):
                            print("DEBUG:     special-case keep: prep contains 'grind'+'soak' while existing is soak-only -> keep prep")
                            # treat as NOT similar for the purposes of skipping
                            continue

                        # If p explicitly mentions 'soaked' or 'soak' and s_exist is more generic,
                        # prefer the more explicit one (keep p).
                        if (p_has_soak and not s_has_soak) and p_has_grind:
                            print("DEBUG:     special-case keep: prep mentions soaked+grind but existing doesn't mention soak -> keep prep")
                            continue

                        # Otherwise, treat as similar and skip
                        print("DEBUG:     > similar (>=0.40), skipping")
                        exists_similar = True
                        break

                if not exists_similar:
                    print("DEBUG:   > adding to prepend list")
                    to_prepend.append(p)
                else:
                    print("DEBUG:   > NOT adding (duplicate-ish)")

            print("\nDEBUG: to_prepend BEFORE soak/grind order =", to_prepend)

            to_prepend.sort(key=lambda s: (0 if 'soak' in s.lower() else 1))

            print("DEBUG: to_prepend AFTER soak/grind order =", to_prepend)

            out_lines = to_prepend + out_lines
            print("DEBUG: out_lines after prep prepend (final) =", out_lines)

        # Collapse repeated words
        print("\nDEBUG: collapsing repeated words in out_lines")
        out_lines = [self._collapse_repeated_words(s).strip() for s in out_lines]
        print("DEBUG: out_lines after collapse =", out_lines)

        out_lines = self._strip_leading_number_prefixes(out_lines)
        print("DEBUG: out_lines after stripping leading numbers =", out_lines)

        # Ensure ingredient coverage
        print("\nDEBUG: calling ensure_ingredient_coverage()")
        out_lines = self.ensure_ingredient_coverage(out_lines, merged_ings)
        print("DEBUG: out_lines after ensure_ingredient_coverage (final) =", out_lines)

        # Re-order steps after ensure_ingredient_coverage adds new steps
        if reorder:
            out_lines = self.reorder_steps(out_lines)
            print("DEBUG: out_lines after final reorder =", out_lines)

        # --- Remove redundant/weak 'ferment' mentions if an explicit fermentation exists ---
        # --- Remove duplicate/weak 'ferment' mentions when an explicit-duration ferment exists ---
        def _has_explicit_duration(text: str) -> bool:
            return bool(re.search(r'\b(overnight|overnight\.?|for\s+\d+\s*(hours?|hrs?|h|minutes?|mins?)|at\s*\d+°[CF])\b', text, flags=re.I))

        # identify explicit ferment steps
        explicit_ferment_indices = [i for i, s in enumerate(out_lines) if re.search(r'\bferment(?:ed|ing)?\b', s, flags=re.I) and _has_explicit_duration(s)]

        if explicit_ferment_indices:
            # keep the first explicit-duration ferment; remove 'weak' ferment mentions from other steps
            explicit_idx = explicit_ferment_indices[0]
            cleaned = []
            for i, s in enumerate(out_lines):
                if i == explicit_idx:
                    cleaned.append(s)  # keep explicit one
                    continue
                # detect weak/conditional ferment mentions
                if re.search(r'\bferment(?:ed|ing)?\b', s, flags=re.I):
                    # common weak phrases we want to remove or strip
                    # If the line also contains other important verbs (like 'grind', 'knead', etc.)
                    # we attempt to strip just the ferment phrase; otherwise drop the ferment-only phrase.
                    if re.search(r'\b(grind|mix|combine|knead|stir|beat|whisk|fold)\b', s, flags=re.I):
                        # remove the fragment containing 'ferment' and surrounding qualifiers
                        s2 = re.sub(r'[,;]?\s*(and\s+)?(?:may\s+)?(?:then\s+)?(?:allow\s+to\s+)?(?:to\s+)?ferment(?:\s+if\s+required|\s+if\s+needed|(?:\s+for\s+[^\.,;]+)?)?', '', s, flags=re.I).strip()
                        # clean leftover punctuation/spacing
                        s2 = re.sub(r'\s{2,}', ' ', s2).strip(' ,;.')
                        if s2:
                            cleaned.append(s2 + ('.' if not s2.endswith('.') else ''))
                        # else drop line
                    else:
                        # line is largely about ferment -> drop it (explicit duration exists elsewhere)
                        # (Optional: you could keep conditional if it adds unique tokens — but we drop here)
                        continue
                else:
                    cleaned.append(s)
            out_lines = cleaned



        # finally dedupe aggressively but preserve readable originals
        out_lines = self._dedupe_steps(out_lines)
        print("DEBUG: out_lines after _dedupe_steps (final) =", out_lines)

        # --- FINAL SAFETY NORMALIZATION: ensure soak comes before grind ---
        final_lines = out_lines.copy()
        print("DEBUG: working copy final_lines =", final_lines)

        for tok in ['rice', 'urad', 'dal', 'semolina', 'besan', 'flour']:
            print(f"\nDEBUG: Checking ingredient token '{tok}'")
            soak_idx = next(
                (i for i, s in enumerate(final_lines)
                if re.search(r'\bsoak\b', s.lower()) and tok in s.lower()),
                None
            )
            grind_idx = next(
                (i for i, s in enumerate(final_lines)
                if re.search(r'\bgrind\b', s.lower()) and tok in s.lower()),
                None
            )

            print(f"DEBUG:   soak_idx={soak_idx}, grind_idx={grind_idx}")

            if soak_idx is not None and grind_idx is not None:
                print(f"DEBUG:   order check soak_idx({soak_idx}) > grind_idx({grind_idx}) > {soak_idx > grind_idx}")

            if soak_idx is not None and grind_idx is not None and soak_idx > grind_idx:
                print("DEBUG:   > reordering needed (soak must come before grind)")

                line = final_lines.pop(soak_idx)
                print("DEBUG:   popped soak line:", line)

                final_lines.insert(grind_idx, line)
                print(f"DEBUG:   inserted soak line at index {grind_idx}")
            else:
                print("DEBUG:   no reorder needed for this token")

        out_lines = final_lines
        print("\nDEBUG: out_lines after final_lines reorder =", out_lines)

        # ensure fermentation happens before any cooking/steaming (robust: handle multiple ferment lines)
        cook_idx = next(
            (i for i, s in enumerate(out_lines)
            if any(k in s.lower() for k in ('steam', 'cook', 'bake', 'fry', 'grill', 'roast', 'simmer'))),
            None
        )
        ferment_idxs = [i for i, s in enumerate(out_lines) if re.search(r'\bferment(?:ed|ing)?\b', s.lower())]

        if cook_idx is not None and ferment_idxs:
            # move ferment lines that are *after* the first cook index to just before cook_idx,
            # preserving the original relative order of ferment lines.
            # Iterate over a copy of indices so popping doesn't break iteration; adjust cook_idx as we insert.
            for fi in sorted(ferment_idxs):
                if fi > cook_idx:
                    line = out_lines.pop(fi)
                    out_lines.insert(cook_idx, line)
                    cook_idx += 1  # keep cook_idx after inserted ferment so subsequent inserts stay before cook

        # Collapse duplicate grind steps (keep the most informative: soaked/ferment/time/batter)
        grind_idxs = [i for i, s in enumerate(out_lines) if re.search(r'\bgrind\b', s, flags=re.I)]
        if len(grind_idxs) > 1:
            def _grind_score(txt: str) -> int:
                low = txt.lower()
                score = 0
                score += 3 if 'soak' in low or 'soaked' in low else 0
                score += 2 if 'ferment' in low else 0
                score += 1 if re.search(r'\b\d+\s*(hours?|hrs?)\b', low) or 'overnight' in low else 0
                score += 1 if 'batter' in low else 0
                return score

            scored = [(idx, _grind_score(out_lines[idx])) for idx in grind_idxs]
            keep_idx, _ = max(scored, key=lambda t: (t[1], -t[0]))
            out_lines = [ln for i, ln in enumerate(out_lines) if i == keep_idx or i not in grind_idxs]

        # Ensure a clear fermentation duration for rice+urad batters
        ferment_lines = [i for i, s in enumerate(out_lines) if 'ferment' in s.lower()]
        has_ferment = bool(ferment_lines)
        has_ferment_duration = any(
            re.search(r'\b\d+\s*(hours?|hrs?)\b', out_lines[i], flags=re.I) or 'overnight' in out_lines[i].lower()
            for i in ferment_lines
        )
        has_idli_base = any('rice' in ing.name.lower() for ing in merged_ings) and any('urad' in ing.name.lower() for ing in merged_ings)
        if has_idli_base:
            if not has_ferment:
                # insert a ferment step after the best grind/mix slot
                insert_after = next((i for i, s in enumerate(out_lines) if re.search(r'\bgrind\b', s, flags=re.I)), None)
                ferment_line = "Ferment the batter for 6-8 hours or overnight."
                if insert_after is None:
                    out_lines.insert(0, ferment_line)
                else:
                    out_lines.insert(insert_after + 1, ferment_line)
            elif not has_ferment_duration:
                # upgrade existing ferment line with a clean duration sentence
                for idx in ferment_lines:
                    base = re.sub(r';?\s*ferment.*', '', out_lines[idx], flags=re.I).rstrip(' ;,')
                    base = base.rstrip(' .;') + '.' if base else ''
                    out_lines[idx] = base if base else out_lines[idx]
                    out_lines.insert(idx + 1, "Ferment the batter for 6-8 hours or overnight.")
                    break

        # Normalize soak duration to a single range and ensure drain
        for idx, s in enumerate(out_lines):
            if re.search(r'\bsoak\b', s, flags=re.I):
                if not re.search(r'\d+\s*[-–]\s*\d+\s*hours?', s, flags=re.I):
                    if re.search(r'\b\d+\s*hours?\b', s, flags=re.I):
                        out_lines[idx] = re.sub(r'\b\d+\s*hours?\b', '4-6 hours', s, flags=re.I)
                    else:
                        out_lines[idx] = s.rstrip(' .;') + ' for 4-6 hours.'
                if 'drain' not in out_lines[idx].lower():
                    out_lines[idx] = out_lines[idx].rstrip(' .;') + ', then drain.'

        # If soy sauce is already added later, remove earlier soy-only steps (don't strip from multi-ingredient steps)
        soy_indices = [i for i, s in enumerate(out_lines) if re.search(r'\bsoy\s+sauce\b', s, flags=re.I)]
        if len(soy_indices) > 1:
            keep = soy_indices[-1]
            cleaned_lines = []
            for i, line in enumerate(out_lines):
                # Skip processing of autogenerated steps - they need all ingredients
                if line.startswith('[AUTO-GEN]'):
                    cleaned_lines.append(line.replace('[AUTO-GEN] ', ''))
                    continue
                    
                if i == keep:
                    cleaned_lines.append(line)
                    continue

                if i in soy_indices:
                    # Check if this step is PRIMARILY about soy (soy-only step) vs multi-ingredient
                    # If line starts with "Add soy" or "Combine...soy" it's likely soy-focused
                    if re.search(r'^(add\s+soy|then\s+add\s+soy|combine.*soy)', line, flags=re.I):
                        # This is a soy-focused step - try to salvage other ingredients
                        s2 = re.sub(r'\b(?:then\s+)?add\s+soy\s+sauce\b', '', line, flags=re.I)
                        s2 = re.sub(r'\band\s+soy\s+sauce\b', '', s2, flags=re.I)
                        s2 = re.sub(r'\bsoy\s+sauce\b', '', s2, flags=re.I)
                        s2 = re.sub(r'^\s*and\s+', '', s2, flags=re.I).strip()
                        s2 = re.sub(r'\s{2,}', ' ', s2).strip(' ,.')
                        
                        # Keep only if other substantial ingredients remain (not just action words)
                        # Look for ingredient-like words (not just cooking verbs)
                        has_ingredients = re.search(r'\b(oil|water|salt|garlic|ginger|onion|pepper|carrot|broccoli|vegetable)\b', s2, flags=re.I)
                        if s2 and len(s2.split()) >= 3 and has_ingredients:
                            cleaned_lines.append(s2 if s2.endswith('.') else s2 + '.')
                        # else: discard this soy-only step
                    else:
                        # This step mentions soy but is not soy-focused (e.g., "Add vegetables and soy sauce")
                        # Strip soy mention but keep the step
                        s2 = re.sub(r'\b(?:then\s+)?add\s+soy\s+sauce\b', '', line, flags=re.I)
                        s2 = re.sub(r'\band\s+soy\s+sauce\b', '', s2, flags=re.I)
                        s2 = re.sub(r'\bsoy\s+sauce\b', '', s2, flags=re.I)
                        s2 = re.sub(r'\.\s*(and|then)\b', '. ', s2, flags=re.I)
                        s2 = re.sub(r'^\s*and\s+', '', s2, flags=re.I).strip()
                        s2 = re.sub(r'\s{2,}', ' ', s2).strip(' ,.')
                        if s2 and not re.search(r'^(and|then)\s+', s2, flags=re.I):
                            cleaned_lines.append(s2 if s2.endswith('.') else s2 + '.')
                    continue

                cleaned_lines.append(line)

            out_lines = cleaned_lines
            print(f"DEBUG: removed duplicate soy sauce mentions; kept index {keep} and removed soy-only steps")


        # --- Domain tweak for idli-style batters: fenugreek belongs in the grind, and water hint is explicit ---
        has_idli_base = any('rice' in ing.name.lower() for ing in merged_ings) and any('urad' in ing.name.lower() for ing in merged_ings)
        if has_idli_base:
            fenugreek_idx = next((i for i, s in enumerate(out_lines) if 'fenugreek' in s.lower()), None)
            grind_idx = next((i for i, s in enumerate(out_lines) if 'grind' in s.lower()), None)

            # Move fenugreek into the grind step
            if fenugreek_idx is not None and grind_idx is not None and fenugreek_idx != grind_idx:
                if 'fenugreek' not in out_lines[grind_idx].lower():
                    line = out_lines[grind_idx].rstrip(' .;')
                    line += '; include fenugreek while grinding.'
                    out_lines[grind_idx] = line
                # Remove fenugreek mention from the later line if it becomes redundant
                if 'fenugreek' in out_lines[fenugreek_idx].lower():
                    cleaned = re.sub(r'\s*with\s+salt\s+and\s+fenugreek', ' with salt', out_lines[fenugreek_idx], flags=re.I)
                    cleaned = re.sub(r'\s*and\s+fenugreek', '', cleaned, flags=re.I).strip()
                    cleaned = cleaned.rstrip(' .;')
                    if len(cleaned.split()) < 3:
                        out_lines.pop(fenugreek_idx)
                        # adjust indices if we removed a line before grind_idx
                        if fenugreek_idx < grind_idx:
                            grind_idx -= 1
                    else:
                        out_lines[fenugreek_idx] = cleaned + '.'

            # Ensure grind step mentions adding water to smooth
            if grind_idx is not None and 'water' not in out_lines[grind_idx].lower():
                out_lines[grind_idx] = out_lines[grind_idx].rstrip(' .;') + '; add water as needed to reach a smooth batter.'



        # Compute AI confidence
        generated_text = generated if isinstance(generated, str) else str(generated)
        print("DEBUG: generated_text type:", type(generated))

        print("\nDEBUG: computing AI confidence")
        ai_conf = self.compute_ai_confidence(len(top_recipes), out_lines, generated_text)
        validator_conf = round(min(1.0, ai_conf * 0.8), 3)
        print(f"DEBUG: final ai_conf={ai_conf}, validator_conf={validator_conf}")

        # Prepare title
        base_title = top_recipes[0].title.split(':')[0].strip()
        title = f"Synthesized -- {base_title} (for {requested_servings} servings)"
        print("DEBUG: final recipe title =", title)

        # Metadata
        meta = {
            "sources": [r.id for r in top_recipes],
            "ai_confidence": ai_conf,
            "synthesis_method": f"llm:{llm_model}"
        }
        print("DEBUG: returning LLM Recipe with meta =", meta)

        # Normalize leavening ingredients
        print("\nDEBUG: calling normalize_leavening() before creating final recipe")
        merged_ings = self.normalize_leavening(merged_ings)

        print("DEBUG: FINAL steps =", out_lines)
        print("DEBUG: FINAL merged_ings =", merged_ings)
        print("DEBUG: END LLM path\n")

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
    km = KitchenMind()
    # create users
    t = km.create_user('alice_trainer', role='trainer')
    v = km.create_user('bob_validator', role='validator')
    u = km.create_user('charlie_user', role='user')

    # trainer submits two versions of a dish
    r1 = km.submit_recipe(
        t,
        title='Idli \u2013 Traditional South Indian Steamed Rice Cakes',
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
        title='Rava Idli \u2013 Quick Version',
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
        title='Dosa \u2013 Crispy South Indian Crepe',
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

    # --- Additional recipes with same titles for synthesis comparison ---

    # Alt 1) Idli - variant with faster fermentation
    r1_alt = km.submit_recipe(
        t,
        title='Idli \u2013 Traditional South Indian Steamed Rice Cakes',
        ingredients=[
            {'name':'Rice', 'quantity':250, 'unit':'g'},
            {'name':'Urad Dal', 'quantity':120, 'unit':'g'},
            {'name':'Water', 'quantity':300, 'unit':'ml'},
            {'name':'Salt', 'quantity':6, 'unit':'g'},
            {'name':'Fenugreek', 'quantity':2, 'unit':'g'},
        ],
        steps=[
            'Rinse rice and urad dal, then soak for 3 hours.',
            'Grind rice and dal separately into fine batter.',
            'Mix both batters with salt and fenugreek.',
            'Ferment for 6-8 hours.',
            'Pour into idli molds and steam for 10 minutes.'
        ],
        servings=5
    )

    # Alt 2) Idli - quick no-ferment version
    r1_alt2 = km.submit_recipe(
        t,
        title='Idli \u2013 Traditional South Indian Steamed Rice Cakes',
        ingredients=[
            {'name':'Rice', 'quantity':300, 'unit':'g'},
            {'name':'Urad Dal', 'quantity':100, 'unit':'g'},
            {'name':'Water', 'quantity':350, 'unit':'ml'},
            {'name':'Salt', 'quantity':5, 'unit':'g'},
            {'name':'Baking soda', 'quantity':1, 'unit':'g'},
        ],
        steps=[
            'Soak rice and urad dal for 2 hours.',
            'Grind into smooth batter.',
            'Add salt and baking soda.',
            'Steam immediately for 12 minutes without fermentation.'
        ],
        servings=4
    )

    # Alt 3) Vegetable Stir-Fry - with garlic and ginger
    r5_alt = km.submit_recipe(
        t,
        title='Quick Vegetable Stir-Fry',
        ingredients=[
            {'name':'Carrot', 'quantity':150, 'unit':'g'},
            {'name':'Bell pepper', 'quantity':100, 'unit':'g'},
            {'name':'Broccoli', 'quantity':100, 'unit':'g'},
            {'name':'Garlic', 'quantity':10, 'unit':'g'},
            {'name':'Ginger', 'quantity':10, 'unit':'g'},
            {'name':'Soy sauce', 'quantity':20, 'unit':'ml'},
            {'name':'Oil', 'quantity':20, 'unit':'ml'},
        ],
        steps=[
            'Chop all vegetables, garlic, and ginger.',
            'Heat oil and fry garlic and ginger for 1 minute.',
            'Add vegetables and stir-fry for 7 minutes on high heat.',
            'Add soy sauce and toss well for 2 minutes.',
            'Serve immediately.'
        ],
        servings=2
    )

    # Alt 4) Yeast Bread - with honey for better rise
    r6_alt = km.submit_recipe(
        t,
        title='Quick Yeast Bread',
        ingredients=[
            {'name':'All-purpose flour', 'quantity':500, 'unit':'g'},
            {'name':'Warm water', 'quantity':300, 'unit':'ml'},
            {'name':'Instant yeast', 'quantity':7, 'unit':'g'},
            {'name':'Salt', 'quantity':8, 'unit':'g'},
            {'name':'Honey', 'quantity':15, 'unit':'g'},
            {'name':'Oil', 'quantity':10, 'unit':'ml'},
        ],
        steps=[
            'Mix flour, yeast, salt, and honey.',
            'Add warm water and knead for 12 minutes until smooth.',
            'Let rise for 1.5 hours in a warm place.',
            'Shape and proof for 30 minutes.',
            'Bake at 200°C for 30 minutes until golden.'
        ],
        servings=8
    )

    # Alt 5) Cucumber Tomato Salad - with vinaigrette
    r12_alt = km.submit_recipe(
        t,
        title='Cucumber Tomato Salad',
        ingredients=[
            {'name':'Cucumber', 'quantity':200, 'unit':'g'},
            {'name':'Tomato', 'quantity':200, 'unit':'g'},
            {'name':'Red onion', 'quantity':50, 'unit':'g'},
            {'name':'Coriander', 'quantity':10, 'unit':'g'},
            {'name':'Olive oil', 'quantity':15, 'unit':'ml'},
            {'name':'Lemon juice', 'quantity':15, 'unit':'ml'},
            {'name':'Salt', 'quantity':3, 'unit':'g'},
        ],
        steps=[
            'Dice cucumber and tomato into bite-sized pieces.',
            'Thinly slice red onion.',
            'Mix vegetables with salt.',
            'Make vinaigrette with olive oil and lemon juice.',
            'Toss salad with vinaigrette and garnish with fresh coriander.',
            'Serve chilled.'
        ],
        servings=3
    )

    # Alt 6) Cucumber Tomato Salad - Mediterranean style
    r12_alt2 = km.submit_recipe(
        t,
        title='Cucumber Tomato Salad',
        ingredients=[
            {'name':'Cucumber', 'quantity':150, 'unit':'g'},
            {'name':'Tomato', 'quantity':150, 'unit':'g'},
            {'name':'Feta cheese', 'quantity':50, 'unit':'g'},
            {'name':'Black olives', 'quantity':30, 'unit':'g'},
            {'name':'Olive oil', 'quantity':10, 'unit':'ml'},
            {'name':'Oregano', 'quantity':2, 'unit':'g'},
            {'name':'Salt', 'quantity':2, 'unit':'g'},
        ],
        steps=[
            'Chop cucumber and tomato.',
            'Add olives and crumbled feta cheese.',
            'Drizzle with olive oil and season with oregano and salt.',
            'Mix gently and let sit for 10 minutes.',
            'Serve at room temperature.'
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
        try:
            synthesized = km.request_recipe(u, 'Idli \u2013 Traditional South Indian Steamed Rice Cakes', servings=5)
        except UnicodeEncodeError as ue:
            print("Unicode encoding error during synthesis (non-fatal), continuing...")
            print(f"  Error: {str(ue)[:100]}")
            import traceback
            traceback.print_exc()
            synthesized = None
        
        if synthesized:
            print('\n--- Synthesized Recipe (for 5) ---')
            try:
                pprint.pprint(asdict(synthesized))
            except UnicodeEncodeError:
                # Fallback for encoding issues
                import json
                print(json.dumps(asdict(synthesized), indent=2, ensure_ascii=True, default=str))
    except Exception as e:
        print("Synthesis failed:", str(e))
        import traceback
        traceback.print_exc()
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
