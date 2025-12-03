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
            if 'beaten' in line.lower():
                beaten_lines.append((i, line))
                if beaten_idx is None:
                    beaten_idx = i
                    print(f"DEBUG: found first BEATEN mention at index {i}: {repr(line)}")
                else:
                    print(f"DEBUG: found additional BEATEN mention at index {i}: {repr(line)}")
        if not beaten_lines:
            print("DEBUG: no BEATEN mentions found")

        # choose the first cook-like index with preference for heat index
        if heat_idx is not None:
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
            insert_idx = first_cook_idx
            print("DEBUG: will insert before first cook/rest/finish/time index:", insert_idx)
        else:
            insert_idx = None
            print("DEBUG: no removals and no protected index -> will append combined step later")

        print("DEBUG: chosen insert_idx =", insert_idx)
        print("DEBUG: END scanning for wet-add candidate\n")


        # Remove the selected indices (descending order to keep indices valid)
        removed_count_before = 0
        removed_indices_sorted = sorted(indices_to_remove, reverse=True)
        removed_indices_set = set(removed_indices_sorted)

        print("\nDEBUG: BEGIN removal pass")
        print("DEBUG: indices_to_remove (sorted desc) =", removed_indices_sorted)
        print("DEBUG: removed_indices_set =", removed_indices_set)
        print("DEBUG: current out_lines before removal:")
        for i, line in enumerate(out_lines):
            print(f"  [{i}] {repr(line)}")

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

        # Build combined instruction (this is your add_step)
        if disp_dry and disp_wet:
            if len(disp_dry) > 1:
                dry_txt = ", ".join(disp_dry[:-1]) + " and " + disp_dry[-1]
            else:
                dry_txt = disp_dry[0]
            if len(disp_wet) > 1:
                wet_txt = ", ".join(disp_wet[:-1]) + " and " + disp_wet[-1]
            else:
                wet_txt = disp_wet[0]
            add_step = f"Combine {dry_txt}. Then add {wet_txt} and mix until just combined."
            print("DEBUG: built add_step (dry+wet) ->", add_step)
        elif disp_dry:
            if len(disp_dry) > 1:
                dry_txt = ", ".join(disp_dry[:-1]) + " and " + disp_dry[-1]
            else:
                dry_txt = disp_dry[0]
            add_step = f"Combine {dry_txt} and mix as required."
            print("DEBUG: built add_step (dry only) ->", add_step)
        else:
            if len(disp_wet) > 1:
                wet_txt = ", ".join(disp_wet[:-1]) + " and " + disp_wet[-1]
            else:
                wet_txt = disp_wet[0] if disp_wet else ""
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

                if overlap >= 0.55:
                    found_similar = k_existing
                    print(f"DEBUG:   -> found similar existing key={repr(k_existing)} with overlap={overlap:.4f}")
                    break
                else:
                    print("DEBUG:      -> not similar enough (needs >= 0.55)")

            if found_similar:
                # If new key is more informative (more distinct tokens), replace the existing kept step
                existing_tokens = set(found_similar.split())
                if len(new_tokens) > len(existing_tokens):
                    replaced_idx = seen_keys[found_similar]
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
            prep_lines.append("Mix semolina with yogurt and water to make a batter; let it rest for 10\u201315 minutes if using semolina.")
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
        # helper to split composite step into prep + cook if it contains both
        _prep_verbs = r'\b(beat|whisk|mix|combine|stir|fold|knead|blend|whisked|beaten)\b'
        _cook_verbs = r'\b(heat|cook|fry|sauté|saute|bake|roast|grill|steam|simmer)\b'
        #
        def _split_prep_and_cook(raw_steps):
            print("DEBUG: _split_prep_and_cook START, raw_steps =", raw_steps)
            out = []
            for idx, s in enumerate(raw_steps):
                print(f"DEBUG:   Step[{idx}] = {repr(s)}")
                low = s.lower()
                if re.search(_prep_verbs, low) and re.search(_cook_verbs, low):
                    print("DEBUG:     -> contains both prep and cook verbs")
                    # attempt to split on common conjunctions/commas/then
                    parts = re.split(r'\b(?:then|and then|, then|;| and | then )\b', s, flags=re.IGNORECASE)
                    print("DEBUG:     split parts =", parts)
                    # find prep part (first containing prep verb), and cook part (first containing cook verb after)
                    prep = None
                    cook = None
                    for pi, p in enumerate(parts):
                        p_stripped = p.strip()
                        print(f"DEBUG:       examining part[{pi}] = {repr(p_stripped)}")
                        if prep is None and re.search(_prep_verbs, p_stripped, re.IGNORECASE):
                            prep = p_stripped
                            print(f"DEBUG:         -> selected as PREP part: {repr(prep)}")
                            continue
                        # look for cook part in remaining pieces (allow cook to appear in same part if prep already set)
                        if cook is None and re.search(_cook_verbs, p_stripped, re.IGNORECASE):
                            cook = p_stripped
                            print(f"DEBUG:         -> selected as COOK part: {repr(cook)}")
                            # do not break — we want to log remaining parts too
                    if prep and cook:
                        prep_line = prep if prep.endswith('.') else prep + '.'
                        cook_line = cook if cook.endswith('.') else cook + '.'
                        print("DEBUG:     -> splitting into prep_line =", repr(prep_line), "and cook_line =", repr(cook_line))
                        out.append(prep_line)
                        out.append(cook_line)
                        continue
                    else:
                        print("DEBUG:     -> could not find both prep and cook parts after split; will keep original step")
                else:
                    print("DEBUG:     -> does NOT contain both prep and cook verbs (skip split)")

                out.append(s)
            print("DEBUG: _split_prep_and_cook END, out =", out)
            return out


        # Usage (before normalization):
        steps = _split_prep_and_cook(steps)
        print("DEBUG:steps =", steps)

        norm_steps = []
        seen = set()

        # --- Normalize and dedupe initial input ---
        for s in steps:
            print(f"DEBUG: processing step: {repr(s)}")
            if not s:
                print("DEBUG:  -> SKIP empty step")
                continue
            s_norm = self._normalize_step_text(s)
            key = s_norm.lower()
            if key and key not in seen:
                seen.add(key)
                norm_steps.append(s_norm)
                print(f"DEBUG:  -> ADD normalized step: {repr(s_norm)}")
            else:
                print(f"DEBUG:  -> SKIP (duplicate or empty): {repr(s_norm)}")

        print("DEBUG: norm_steps after initial normalization =", norm_steps)

        if not norm_steps:
            print("DEBUG: no normalized steps -> return []")
            return []

        # ------------------- Detect preserve_combine -------------------
        preserve_combine = None
        dry_keep_keywords = ["flour", "gram flour", "besan", "all-purpose", "yeast", "salt", "egg", "eggs"]

        print("DEBUG: checking for preserve_combine candidate...")
        for s in norm_steps:
            low = s.lower()
            if any(k in low for k in ("combine", "whisk", "mix")):
                if any(dk in low for dk in dry_keep_keywords):
                    preserve_combine = s
                    print(f"DEBUG: preserve_combine FOUND = {repr(preserve_combine)}")
                    break

        # ------------------- Detect batter_step -------------------
        flour_pattern = r"(gram flour|besan|semolina|suji|maida|atta|rice|[a-z ]+flour)"
        yogurt_pattern = r"(yogurt|curd|dahi|yoghurt)"

        batter_step = None
        print("DEBUG: checking for batter_step...")

        for idx, s in enumerate(norm_steps):
            low = s.lower()
            print(f"DEBUG:   Step[{idx}] = {repr(s)}")
            print("DEBUG:     lower =", repr(low))

            # Check for mixing verbs
            has_mix_verb = any(v in low for v in ["mix", "whisk", "combine", "stir"])
            print("DEBUG:     has_mix_verb =", has_mix_verb)

            if not has_mix_verb:
                print("DEBUG:     -> no mix/whisk/combine/stir verb, skip")
                continue

            # Check flour/yogurt tokens
            flour_match = re.search(flour_pattern, low)
            yogurt_match = re.search(yogurt_pattern, low)
            print("DEBUG:     flour_match =", flour_match)
            print("DEBUG:     yogurt_match =", yogurt_match)

            if flour_match and yogurt_match:
                print(f"DEBUG:     batter pattern matched in {repr(s)}")

                # Extract and debug group values
                m_flour = flour_match
                m_yog = yogurt_match

                flour_txt_raw = m_flour.group(1) if m_flour else "flour"
                yogurt_txt_raw = m_yog.group(1) if m_yog else "yogurt"

                print("DEBUG:       flour_txt_raw =", repr(flour_txt_raw))
                print("DEBUG:       yogurt_txt_raw =", repr(yogurt_txt_raw))

                flour_txt = flour_txt_raw.strip().title()
                yog_txt = yogurt_txt_raw.strip().title()

                print("DEBUG:       flour_txt =", flour_txt)
                print("DEBUG:       yog_txt =", yog_txt)

                batter_step = (
                    f"Whisk the {flour_txt} and {yog_txt} together, adding water gradually to form a smooth batter."
                )
                print("DEBUG: batter_step CREATED =", repr(batter_step))
                break
            else:
                print("DEBUG:     -> flour/yogurt patterns NOT both present; continue")

        print("DEBUG: final batter_step =", repr(batter_step))


        # ------------------- Detect add_step (summary fallback) -------------------
        key_add_names = ["water", "eno", "baking soda", "sugar", "salt"]
        seen_add = []
        #
        print("DEBUG: scanning for add_step...")
        for idx, s in enumerate(norm_steps):
            low = s.lower()
            print(f"DEBUG:   Step[{idx}] = {repr(s)}")
            print("DEBUG:     lower =", repr(low))

            if "add" in low:
                print("DEBUG:     -> contains 'add'")
                for name in key_add_names:
                    print(f"DEBUG:       checking if name={repr(name)} in step...")
                    if name in low:
                        if name not in seen_add:
                            seen_add.append(name)
                            print(f"DEBUG:         -> detected add ingredient = {name}")
                        else:
                            print("DEBUG:         -> already recorded, skipping")
                    else:
                        print("DEBUG:         -> not present")
            else:
                print("DEBUG:     -> does NOT contain 'add', skip ingredient checks")

        print("DEBUG: after scan, seen_add =", seen_add)

        add_step = None
        if seen_add:
            print("DEBUG: building add_step from seen_add...")
            display_parts = []
            for n in seen_add:
                disp = "Eno" if n == "eno" else n
                print(f"DEBUG:   converting token {repr(n)} -> display {repr(disp)}")
                display_parts.append(disp)

            if len(display_parts) == 1:
                list_txt = display_parts[0]
            else:
                list_txt = ", ".join(display_parts[:-1]) + " and " + display_parts[-1]

            add_step = f"Add {list_txt}. Mix gently until just combined."
            print("DEBUG: add_step CREATED =", repr(add_step))
        else:
            print("DEBUG: no add ingredients detected -> add_step remains None")

        print("DEBUG: final add_step =", repr(add_step))


        # ------------------- Detect cook_step (steam fallback) -------------------
        cook_step = None
        print("DEBUG: scanning for cook_step...")
        for s in norm_steps:
            low = s.lower()
            if "steam" in low:
                print(f"DEBUG: steam detected in: {repr(s)}")
                m_time = re.search(r"(\d+)\s*(?:mins?|minutes?)", low)
                if m_time:
                    cook_step = f"Steam for {m_time.group(1)} minutes."
                else:
                    cook_step = "Steam until cooked through."
                print("DEBUG: cook_step CREATED =", cook_step)
                break

        if not cook_step:
            if any("steam" in s.lower() for s in norm_steps):
                cook_step = "Steam until cooked through."
                print("DEBUG: cook_step fallback CREATED =", cook_step)

        # Local fallbacks to use if self doesn't provide these helpers
        def _normalize_step_text_local(s: str) -> str:
            print(f"DEBUG: _normalize_step_text_local START input={repr(s)}")
            try:
                out = self._normalize_step_text(s)
                print(f"DEBUG: _normalize_step_text_local -> used primary _normalize_step_text output={repr(out)}")
                return out
            except Exception as e:
                print("DEBUG: _normalize_step_text_local -> primary _normalize_step_text raised exception:", repr(e))
                # fallback behavior (preserve original logic)
                if not s:
                    print("DEBUG: _normalize_step_text_local fallback -> input empty/None, returning empty string")
                    return ""
                s2 = s
                print(f"DEBUG: _normalize_step_text_local fallback before strip: {repr(s2)}")
                s2 = s2.strip()
                print(f"DEBUG: _normalize_step_text_local after strip: {repr(s2)}")
                s2 = re.sub(r'\s+', ' ', s2)
                print(f"DEBUG: _normalize_step_text_local after collapse spaces: {repr(s2)}")
                # ensure sentence-ending punctuation
                s2 = s2.rstrip('.;') + '.'
                print(f"DEBUG: _normalize_step_text_local after ensure punctuation: {repr(s2)}")
                return s2

        #
        def _normalize_for_dedupe_local(s: str) -> str:
            print(f"DEBUG: _normalize_for_dedupe_local START input={repr(s)}")
            try:
                out = self._normalize_for_dedupe(s)
                print(f"DEBUG: _normalize_for_dedupe_local -> primary _normalize_for_dedupe output={repr(out)}")
                return out
            except Exception as e:
                print("DEBUG: _normalize_for_dedupe_local -> primary _normalize_for_dedupe raised exception:", repr(e))

                # --- fallback normalization (unchanged logic) ---
                print("DEBUG: _normalize_for_dedupe_local fallback path entered")

                s2 = s
                print(f"DEBUG:   fallback initial value: {repr(s2)}")

                s2 = s2.lower()
                print(f"DEBUG:   after lower(): {repr(s2)}")

                # remove all punctuation except alphanumerics and spaces
                before_clean = s2
                s2 = re.sub(r'[^a-z0-9\s]', ' ', s2)
                print(f"DEBUG:   after removing punctuation: {repr(s2)} (before was {repr(before_clean)})")

                # collapse whitespace
                before_space = s2
                s2 = re.sub(r'\s+', ' ', s2).strip()
                print(f"DEBUG:   after collapsing whitespace: {repr(s2)} (before was {repr(before_space)})")

                print("DEBUG: _normalize_for_dedupe_local END output=", repr(s2))
                return s2

        #
        def classify_phase_local(text: str) -> str:
            print(f"DEBUG: classify_phase_local START input={repr(text)}")

            try:
                phase = self.classify_phase(text)
                print(f"DEBUG: classify_phase_local -> primary classify_phase returned {repr(phase)}")
                return phase
            except Exception as e:
                print("DEBUG: classify_phase_local -> primary classify_phase raised exception:", repr(e))
                print("DEBUG: classify_phase_local -> entering fallback classification")

                t = text.lower()
                print(f"DEBUG:   fallback lowered text={repr(t)}")

                # cook
                cook_keys = ['bake','roast','cook','fry','simmer','saute','steam']
                if any(k in t for k in cook_keys):
                    print("DEBUG:   matched COOK keys:", cook_keys)
                    print("DEBUG: classify_phase_local END -> 'cook'")
                    return 'cook'

                # rest
                rest_keys = ['rest','rise','proof','ferment','hang','set']
                if any(k in t for k in rest_keys):
                    print("DEBUG:   matched REST keys:", rest_keys)
                    print("DEBUG: classify_phase_local END -> 'rest'")
                    return 'rest'

                # finish
                finish_keys = ['finish','serve','garnish']
                if any(k in t for k in finish_keys):
                    print("DEBUG:   matched FINISH keys:", finish_keys)
                    print("DEBUG: classify_phase_local END -> 'finish'")
                    return 'finish'

                # add
                add_keys = ['add','mix','combine','stir','knead','whisk','grind','soak','soaked','beat']
                if any(k in t for k in add_keys):
                    print("DEBUG:   matched ADD keys:", add_keys)
                    print("DEBUG: classify_phase_local END -> 'add'")
                    return 'add'

                print("DEBUG:   no fallback category matched")
                print("DEBUG: classify_phase_local END -> 'other'")
                return 'other'

        #
        def has_time_or_temp_local(text: str) -> bool:
            print(f"\nDEBUG: has_time_or_temp_local START input={repr(text)}")
            try:
                out = self.has_time_or_temp(text)
                print(f"DEBUG: has_time_or_temp_local -> primary has_time_or_temp returned {repr(out)}")
                return out
            except Exception as e:
                print("DEBUG: has_time_or_temp_local -> primary raised exception:", repr(e))
                print("DEBUG: entering fallback regex check")

                pattern = r'\b(\d+\s*(?:-|\u2013)?\d*\s*(?:min|mins|minutes|h|hr|hour|hours)|\d+°C|\d+°F|for \d+|overnight)\b'
                print("DEBUG:   fallback pattern =", pattern)

                t = text.lower()
                print("DEBUG:   lowered text =", repr(t))

                matched = bool(re.search(pattern, t))
                print("DEBUG:   fallback matched =", matched)
                print("DEBUG: has_time_or_temp_local END ->", matched)
                return matched


        # hours extractor (kept local)
        def extract_hours(text: str):
            print(f"\nDEBUG: extract_hours START input={repr(text)}")
            t = text.lower()
            print("DEBUG: lowered =", repr(t))

            # overnight → special case
            if 'overnight' in t:
                print("DEBUG: matched 'overnight' -> returning 12.0 hours")
                return 12.0

            # range case: "3-5 hours", "3 to 5 hr"
            m_range = re.search(r'(\d+(?:\.\d*)?)\s*[-–to]+\s*(\d+(?:\.\d*)?)\s*(?:hours?|hrs?|h)\b', t)
            print("DEBUG: m_range =", m_range)
            if m_range:
                a = float(m_range.group(1))
                b = float(m_range.group(2))
                val = max(a, b)
                print(f"DEBUG:   parsed range ({a}, {b}) -> returning {val}")
                return val

            # single explicit hour value
            m_single = re.search(r'(\d+(?:\.\d*)?)\s*(?:hours?|hrs?|h)\b', t)
            print("DEBUG: m_single =", m_single)
            if m_single:
                val = float(m_single.group(1))
                print(f"DEBUG:   parsed single hours -> returning {val}")
                return val

            # fallback: any number
            m_num = re.search(r'\b(\d+(?:\.\d*)?)\b', t)
            print("DEBUG: m_num =", m_num)
            if m_num:
                val = float(m_num.group(1))
                print(f"DEBUG:   fallback numeric -> returning {val}")
                return val

            print("DEBUG: extract_hours -> no matches, returning None")
            return None


        merged = []
        add_buffer = []
        #
        def flush_add_buffer():
            """Flush buffered 'add' lines into merged (with normalization) and clear buffer."""
            if not add_buffer:
                print("DEBUG: flush_add_buffer called but add_buffer is empty -> nothing to do")
                return
            joined = " ".join(add_buffer).strip()
            joined = re.sub(r'\s+', ' ', joined)
            if not joined.endswith('.'):
                joined += '.'
            try:
                norm_joined = _normalize_step_text_local(joined)
            except Exception as exc:
                print("DEBUG: _normalize_step_text_local raised exception on joined buffer:", exc)
                norm_joined = joined
            merged.append(norm_joined)
            print("DEBUG: flushed add_buffer ->", repr(joined), "normalized ->", repr(norm_joined))
            add_buffer.clear()

        # streaming: buffer 'add' steps and append others
        for idx, s in enumerate(norm_steps):
            try:
                s_norm = _normalize_step_text_local(s)
            except Exception as exc:
                print(f"DEBUG: _normalize_step_text_local failed for step idx={idx} original={repr(s)} -> {exc}")
                # best-effort fallback: simple clean
                s_norm = (s or "").strip()
                s_norm = re.sub(r'\s+', ' ', s_norm)
                if s_norm and not s_norm.endswith('.'):
                    s_norm += '.'
                print(f"DEBUG: fallback normalized step -> {repr(s_norm)}")

            try:
                phase = classify_phase_local(s_norm)
            except Exception as exc:
                print(f"DEBUG: classify_phase_local raised for idx={idx} step={repr(s_norm)} -> {exc}")
                phase = 'other'

            try:
                protected = (phase in {'cook', 'rest', 'finish'}) or has_time_or_temp_local(s_norm)
            except Exception as exc:
                print(f"DEBUG: has_time_or_temp_local raised for idx={idx} step={repr(s_norm)} -> {exc}")
                protected = (phase in {'cook', 'rest', 'finish'})

            print(
                f"DEBUG: processing idx={idx} original={repr(s)} normalized={repr(s_norm)} "
                f"phase={phase} protected={protected}"
            )

            if phase == 'add' and not protected:
                add_buffer.append(s_norm)
                print(f"DEBUG: buffered add step (idx={idx}):", repr(s_norm), "-> add_buffer_size=", len(add_buffer))
                # continue buffering — do not append to merged yet
                continue
            else:
                # before appending a non-add/protected step, flush any buffered add steps
                if add_buffer:
                    print("DEBUG: non-add/protected encountered -> flushing add_buffer before appending this step")
                flush_add_buffer()

                merged.append(s_norm)
                print("DEBUG: appended protected/non-add step (idx={}): {}".format(idx, repr(s_norm)))

        # final flush after loop
        if add_buffer:
            print("DEBUG: end of stream -> flushing remaining add_buffer")
        flush_add_buffer()

        # append fallback add_step/cook_step if absent
        if not any('add' in x.lower() for x in merged) and add_step:
            merged.append(_normalize_step_text_local(add_step))
            print("DEBUG: appended fallback add_step:", add_step)

        if cook_step and not any('steam' in x.lower() for x in merged):
            merged.append(_normalize_step_text_local(cook_step))
            print("DEBUG: appended fallback cook_step:", cook_step)

        print("DEBUG: merged after streaming =", merged)

        # Ensure heating before cooking when eggs/beaten referenced
        try:
            low_merged = [s.lower() for s in merged]
            cook_idx = next((i for i,s in enumerate(low_merged) if re.search(r'\b(cook|fry|bake|simmer|steam)\b', s)), None)
            heat_idx = next((i for i,s in enumerate(low_merged) if re.search(r'\b(heat|preheat)\b', s)), None)

            if cook_idx is not None and heat_idx is not None and cook_idx < heat_idx:
                cook_text = low_merged[cook_idx]
                if any(k in cook_text for k in ('beaten', 'beaten eggs', 'beat', 'egg mixture', 'pour')):
                    heat_line = merged.pop(heat_idx)
                    merged.insert(cook_idx, heat_line)
                    print(f"DEBUG: moved heat line from index {heat_idx} to {cook_idx} in merged steps")
        except Exception as _e:
            print("DEBUG: heat/cook reorder check failed:", _e)

        # insert pour step if needed when beaten eggs referenced
        try:
            low_merged = [s.lower() for s in merged]
            cook_idx = next((i for i,s in enumerate(low_merged) if re.search(r'\b(cook|fry|bake|simmer|steam)\b', s)), None)
            if cook_idx is not None and 'beaten' in low_merged[cook_idx]:
                has_pour = any('pour' in s or 'pour the' in s for s in low_merged)
                if not has_pour:
                    heat_idx = next((i for i,s in enumerate(low_merged) if re.search(r'\b(heat|preheat)\b', s)), None)
                    insert_pos = (heat_idx + 1) if heat_idx is not None else cook_idx
                    pour_step = "Pour the egg mixture into the pan and cook until set."
                    merged.insert(insert_pos, pour_step)
                    print(f"DEBUG: inserted pour step at index {insert_pos}")
        except Exception as _e:
            print("DEBUG: pour-step insertion check failed:", _e)

        # Split combined soak+grind (safer split) to allow ordering
        try:
            print("DEBUG: ORIGINAL merged =", merged)

            new_merged = []
            for idx, s in enumerate(merged):
                print(f"\nDEBUG: Processing line {idx}: {s}")
                low = s.lower()

                # Check for combined soak + grind line
                if ('grind' in low) and (('soak' in low) or ('soaked' in low)):
                    print("DEBUG: detected possible combined soak+grind line")

                    m = re.search(
                        r'\bsoak(?:ed)?\b\s*(.+?)(?:\s+\bto\b|\s+\binto\b|$)',
                        s, flags=re.I
                    )
                    print("DEBUG: regex match =", m)

                    if m:
                        ing_chunk = m.group(1).strip().rstrip('.;')
                        print("DEBUG: extracted ing_chunk =", ing_chunk)

                        if ing_chunk:
                            soak_line = f"Soak {ing_chunk} as required."
                            print("DEBUG: soak_line =", soak_line)

                            grind_line = re.sub(
                                r'\bsoak(?:ed)?\b\s*' + re.escape(ing_chunk),
                                '',
                                s,
                                flags=re.I
                            ).strip()
                            print("DEBUG: grind_line after removal =", grind_line)

                            grind_line = re.sub(r'\s+', ' ', grind_line).strip().rstrip('.;') + '.'
                            print("DEBUG: final grind_line =", grind_line)

                            new_merged.append(soak_line)
                            new_merged.append(grind_line)
                            continue

                print("DEBUG: no split, keeping line as-is")
                new_merged.append(s)

            merged = new_merged
            print("DEBUG: split combined soak+grind lines, merged now =", merged)

        except Exception as _e:
            print("DEBUG: combined soak+grind split failed:", _e)


        # Ensure soak before grind for ingredient tokens
        try:
            print("DEBUG: START soak/grind reorder block")
            print("DEBUG: initial merged =", merged)

            for ing_tok in ['rice', 'urad', 'dal', 'semolina', 'besan', 'flour']:
                print(f"\nDEBUG: Checking ingredient token: {ing_tok}")

                soak_idx = next(
                    (i for i, s in enumerate(merged)
                    if is_pure_soak(s) and ing_tok in s.lower()),
                    None
                )
                grind_idx = next(
                    (i for i, s in enumerate(merged)
                    if re.search(r'\bgrind\b', s.lower()) and ing_tok in s.lower()),
                    None
                )

                print(f"DEBUG: found soak_idx={soak_idx}, grind_idx={grind_idx}")

                if soak_idx is not None and grind_idx is not None:
                    print(f"DEBUG: order check: soak_idx({soak_idx}) > grind_idx({grind_idx})? {soak_idx > grind_idx}")

                if soak_idx is not None and grind_idx is not None and soak_idx > grind_idx:
                    print("DEBUG: condition true → moving line")

                    moved_line = merged.pop(soak_idx)
                    print(f"DEBUG: popped soak line: {moved_line}")

                    merged.insert(grind_idx, moved_line)
                    print(
                        f"DEBUG: moved soak line for {ing_tok} "
                        f"from index {soak_idx} to {grind_idx}"
                    )
                else:
                    print(f"DEBUG: no movement needed for {ing_tok}")

            print("DEBUG: final merged after reorder =", merged)
            print("DEBUG: END soak/grind reorder block")

        except Exception as _e:
            print("DEBUG: soak/grind reorder check failed:", _e)


        # Post-processing: remove generic soaks if timed ones exist, collapse duplicates, prefer informative steps
        ing_tokens = ['rice', 'urad', 'dal', 'semolina', 'besan', 'flour']
        lower_merged = [s.lower() for s in merged]

        print("DEBUG: lower_merged =", lower_merged)

        to_remove_indices = set()

        for tok in ing_tokens:
            print(f"\nDEBUG: Checking ingredient token: '{tok}'")

            time_idx = None
            generic_idxs = []

            for i, s in enumerate(lower_merged):
                print(f"DEBUG:   scanning index {i}: {s}")

                if is_pure_soak(s) and tok in s:
                    print("DEBUG:     soak+token match")

                    # Check for time-based soak
                    has_time = (
                        has_time_or_temp_local(s) or
                        re.search(r'(\b\d+\b|\b\d+\-\d+\b|\b4–6\b|\bovernight\b)', s)
                    )

                    print(f"DEBUG:     has_time = {bool(has_time)}")

                    if has_time:
                        print(f"DEBUG:     → time-based soak found at index {i}")
                        time_idx = i
                    else:
                        print(f"DEBUG:     → generic soak found at index {i}")
                        generic_idxs.append(i)

            print(f"DEBUG: For '{tok}', time_idx={time_idx}, generic_idxs={generic_idxs}")

            if time_idx is not None and generic_idxs:
                print(f"DEBUG: Removing generic soaks for '{tok}' because timed soak exists")

                for gi in generic_idxs:
                    to_remove_indices.add(gi)
                    print(f"DEBUG:   marking index {gi} for removal")

        print("\nDEBUG: to_remove_indices =", to_remove_indices)

        merged2 = [s for idx, s in enumerate(merged) if idx not in to_remove_indices]
        print("DEBUG: merged2 after removing generic soaks =", merged2)

        # collapse multiple time-based soaks for same ingredient (prefer range/separately/longer/overnight)
        for tok in ing_tokens:
            print(f"\nDEBUG: Checking ingredient token: '{tok}'")

            indices = [i for i, s in enumerate(merged2) if is_pure_soak(s) and tok in s.lower()]
            print(f"DEBUG: found soak indices for '{tok}': {indices}")

            if len(indices) <= 1:
                print(f"DEBUG: only {len(indices)} soak(s) → skipping '{tok}'")
                continue

            best_idx = None
            best_score = float('-inf')

            for i in indices:
                s = merged2[i]
                low = s.lower()
                score = 0.0

                # strong indicators
                if re.search(r'\d+\s*[-–to]\s*\d+|separately', low):
                    score += 100.0
                    print(f"DEBUG:   idx {i}: +100 (range/separately)")

                if has_time_or_temp_local(low):
                    score += 10.0
                    print(f"DEBUG:   idx {i}: +10 (time/temp detected)")

                # numeric hours
                h = extract_hours(low)
                if h is not None:
                    score += float(h)
                    print(f"DEBUG:   idx {i}: +{h} (hours extracted)")

                # tiny score for length
                score += len(low) / 1000.0

                print(f"DEBUG: soak candidate for '{tok}' idx={i} score={score} text={s}")

                if score > best_score:
                    print(f"DEBUG:   idx {i} becomes new best (prev={best_score})")
                    best_score = score
                    best_idx = i

            print(f"DEBUG: Best soak for '{tok}' is index {best_idx} with score {best_score}")

            # remove all except best
            for i in sorted(indices, reverse=True):
                if i != best_idx:
                    removed = merged2.pop(i)
                    print(f"DEBUG: removing duplicate soak for '{tok}' at index {i}, removed: {removed}")

        print("\nDEBUG: final merged2 after duplicate soak cleanup =", merged2)

        # move soak before grind again if needed
        for tok in ing_tokens:
            print(f"\nDEBUG: Checking token '{tok}'")

            soak_idx = next(
                (i for i, s in enumerate(merged2)
                if is_pure_soak(s) and tok in s.lower()),
                None
            )
            grind_idx = next(
                (i for i, s in enumerate(merged2)
                if re.search(r'\bgrind\b', s.lower()) and tok in s.lower()),
                None
            )

            print(f"DEBUG: for '{tok}' → soak_idx={soak_idx}, grind_idx={grind_idx}")

            if soak_idx is not None and grind_idx is not None:
                print(f"DEBUG: order check soak_idx({soak_idx}) > grind_idx({grind_idx}) ? {soak_idx > grind_idx}")

            if soak_idx is not None and grind_idx is not None and soak_idx > grind_idx:
                print("DEBUG: condition true → moving soak line")

                line = merged2.pop(soak_idx)
                print(f"DEBUG:   popped soak line: {line}")

                merged2.insert(grind_idx, line)
                print(f"DEBUG: moved soak line for '{tok}' from {soak_idx} to {grind_idx}")
            else:
                print(f"DEBUG: no move needed for '{tok}'")
        print("\nDEBUG: final merged2 after reorder =", merged2)

        # final dedupe: prefer more informative steps (time/soak/longer)
        final = []
        seen_keys = {}

        for s in merged2:
            print(f"\nDEBUG: processing step: {s}")

            key = _normalize_for_dedupe_local(s)
            print(f"DEBUG:   normalized key = '{key}'")

            if key not in seen_keys:
                print("DEBUG:   → new key, adding to final")
                seen_keys[key] = s
                final.append(s)
            else:
                print("DEBUG:   → duplicate key detected")

                existing = seen_keys[key]

                # Compute scores
                score_existing = (
                    (1 if has_time_or_temp_local(existing) else 0) +
                    (1 if 'soak' in existing.lower() or 'soaked' in existing.lower() else 0) +
                    (len(existing) / 200.0)
                )

                score_new = (
                    (1 if has_time_or_temp_local(s) else 0) +
                    (1 if 'soak' in s.lower() or 'soaked' in s.lower() else 0) +
                    (len(s) / 200.0)
                )

                print(f"DEBUG:     existing='{existing}', score_existing={score_existing}")
                print(f"DEBUG:     new     ='{s}', score_new={score_new}")

                if score_new > score_existing + 0.01:
                    print("DEBUG:     → replacing existing with more informative new step")

                    idx = final.index(existing)
                    final[idx] = s
                    seen_keys[key] = s

                    print("DEBUG: replaced less-informative step with more informative:", s)
                else:
                    print("DEBUG:     → keeping existing, skipping new")
                    print("DEBUG: skipped duplicate step:", s)

        print("\nDEBUG: final deduped steps =", final)

        # compact duplicates preserving order
        compacted = []
        seen_keys2 = set()

        for s in final:
            print(f"\nDEBUG: processing step: {s}")

            k = _normalize_for_dedupe_local(s)
            print(f"DEBUG:   normalized key = '{k}'")

            if k in seen_keys2:
                print("DEBUG:   → duplicate detected, skipping:", s)
                print("DEBUG: final duplicate removal skip:", s)
                continue

            print("DEBUG:   → new key, adding to compacted")
            seen_keys2.add(k)
            compacted.append(s)

        print("\nDEBUG: compacted final steps =", compacted)

        # collapse redundant grind lines: group by ingredient key or use generic 'grind_smooth_batter' for short keys
        grind_seen = {}
        new_compacted = []

        for s in compacted:
            print(f"\nDEBUG: processing step: {s}")
            low = s.lower()

            if 'grind' in low or 'ground' in low:
                print("DEBUG:   → grind-related step detected")

                # Normalize dedupe key
                base = re.sub(
                    r'\b(grind|ground)\b.*?(?:\bto\b|\binto\b|\band\b|,|$)',
                    '',
                    low
                )
                k = _normalize_for_dedupe_local(base)
                print(f"DEBUG:   preliminary grind key = '{k}'")

                # fallback when key is too short or generic
                if not k or len(k.split()) < 3 or 'smooth batter' in low:
                    print("DEBUG:   key too short or 'smooth batter' detected — using fallback key")
                    k = 'grind_smooth_batter'

                print(f"DEBUG:   final grind key = '{k}'")

                if k in grind_seen:
                    old = grind_seen[k]
                    print(f"DEBUG:   duplicate grind group found: old='{old}'")

                    # preference logic
                    pref_new = any(w in low for w in ('soak', 'soaked', 'ferment', 'combine', 'overnight'))
                    pref_old = any(w in old.lower() for w in ('soak', 'soaked', 'ferment', 'combine', 'overnight'))

                    print(f"DEBUG:     pref_new={pref_new}, pref_old={pref_old}")

                    if pref_new and not pref_old:
                        print("DEBUG:     → new step is more informative; replacing old")

                        idx = new_compacted.index(old)
                        new_compacted[idx] = s
                        grind_seen[k] = s

                        print("DEBUG: replaced grind candidate with more informative:", s)
                    else:
                        print("DEBUG:     → skipping new grind step, old is better or equal")
                        print("DEBUG: skipped redundant grind:", s)
                    continue

                else:
                    print("DEBUG:   first time seeing this grind key → adding")
                    grind_seen[k] = s
                    new_compacted.append(s)

            else:
                print("DEBUG:   not a grind step → keeping unchanged")
                new_compacted.append(s)

        compacted = new_compacted

        print("\nDEBUG: final compacted after grind-dedupe =", compacted)

        # bucketize and reorder: soak -> grind -> add -> other -> cook
        soak_bucket = []
        grind_bucket = []
        add_bucket = []
        cook_bucket = []
        other_bucket = []

        for s in compacted:
            low = s.lower()
            print(f"\nDEBUG: processing step: {s}")

            if 'soak' in low:
                soak_bucket.append(s)
                print("DEBUG:   → added to soak_bucket")
            elif 'grind' in low:
                grind_bucket.append(s)
                print("DEBUG:   → added to grind_bucket")
            elif any(k in low for k in ['combine', 'add', 'mix', 'whisk', 'stir']):
                add_bucket.append(s)
                print("DEBUG:   → added to add_bucket")
            elif any(k in low for k in ['cook', 'fry', 'spread', 'bake', 'steam', 'roast']):
                cook_bucket.append(s)
                print("DEBUG:   → added to cook_bucket")
            else:
                other_bucket.append(s)
                print("DEBUG:   → added to other_bucket")

        print("\nDEBUG: Buckets filled:")
        print("DEBUG: soak_bucket =", soak_bucket)
        print("DEBUG: grind_bucket =", grind_bucket)
        print("DEBUG: add_bucket =", add_bucket)
        print("DEBUG: other_bucket =", other_bucket)
        print("DEBUG: cook_bucket =", cook_bucket)

        reordered = []

        # Soak always first
        reordered.extend(soak_bucket)

        # Grinder rules
        grind_soaked = [g for g in grind_bucket if 'soak' in g.lower() or 'soaked' in g.lower()]
        grind_other = [g for g in grind_bucket if g not in grind_soaked]

        print("\nDEBUG: grind_soaked =", grind_soaked)
        print("DEBUG: grind_other =", grind_other)

        reordered.extend(grind_soaked + grind_other)

        # Add / combine / mix steps
        reordered.extend(add_bucket)

        # Other steps
        reordered.extend(other_bucket)

        # Cooking steps last
        reordered.extend(cook_bucket)

        print("\nDEBUG: Reordered (before final dedupe) =", reordered)

        final_ordered = []
        seen_final = set()

        for s in reordered:
            k = _normalize_for_dedupe_local(s)
            print(f"DEBUG: checking final dedupe for: '{s}' key='{k}'")

            if k not in seen_final:
                print("DEBUG:   → new key, adding")
                seen_final.add(k)
                final_ordered.append(s)
            else:
                print("DEBUG:   → duplicate, skipping")

        print("\nDEBUG: merge_semantic_steps FINAL =", final_ordered)
        print("DEBUG: END semantic-bucket-reorder block")

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
                print(f"DEBUG:     replaced alias '{alias}' → '{canon.title()}'")
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
        for phase in ['prep', 'mix', 'rest', 'cook', 'finish']:
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
              "Write 4–8 numbered steps. Keep steps short (one sentence each). Do NOT add new ingredients or quantities.\n"
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
                    print("DEBUG:   → new step, keeping")
                    seen.add(key)
                    fallback_steps.append(s_clean)
                else:
                    print("DEBUG:   → duplicate step, skipping")

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
                    print("DEBUG:   → moving soak line above grind line")

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

                for p in prep_normed:
                    print(f"\nDEBUG: evaluating prep step: {p}")

                    p_key = self._token_set(p)
                    print("DEBUG:   p_key =", p_key)

                    if not p_key:
                        print("DEBUG:   empty token set → skipping")
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

                        if sim >= 0.55:
                            print("DEBUG:     → similar (>=0.55), skipping")
                            exists_similar = True
                            break

                    if not exists_similar:
                        print("DEBUG:   → adding to prepend list")
                        to_prepend.append(p)
                    else:
                        print("DEBUG:   → NOT adding (duplicate-ish)")


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
                    print("DEBUG:   → moving soak line before grind line")

                    line = final_lines.pop(soak_idx)
                    print(f"DEBUG:   popped line: {line}")

                    final_lines.insert(grind_idx, line)
                    print(f"DEBUG:   inserted at index {grind_idx}")

                else:
                    print("DEBUG:   no fix needed for this token")

            out_lines = final_lines
            print("\nDEBUG: out_lines after final_lines reorder =", out_lines)

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
                        print("DEBUG:   empty → skip")
                        continue

                    if not s.endswith(('.', '!', '?')):
                        print("DEBUG:   missing punctuation → adding '.'")
                        s = s + '.'

                    word_count = len(s.split())
                    print(f"DEBUG:   word_count={word_count}")

                    if word_count >= 3:
                        print("DEBUG:   → accepted, appending to out_lines")
                        out_lines.append(s)
                    else:
                        print("DEBUG:   → rejected (too short)")
        else:
            print("DEBUG: no leading block found before numbered list")

        # 2) Now append numbered matches (sorted by their numeric label so order is stable)
        if matches:
            print("DEBUG: matches FOUND → sorting by numeric index")

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
                    print("DEBUG:   missing punctuation → adding '.'")
                    cleaned = cleaned + '.'

                # basic length filter
                wc = len(cleaned.split())
                print(f"DEBUG:   word_count = {wc}")

                if wc >= 3:
                    print("DEBUG:   → accepted, adding to out_lines")
                    out_lines.append(cleaned)
                else:
                    print("DEBUG:   → rejected (too short)")

        else:
            print("DEBUG: no matches → skipping numbered step parsing")

        print("\nDEBUG: out_lines after parsing numbered/un-numbered steps =", out_lines)

        # --- handle case where model started numbering at 2 (or omitted leading "1.") using the already-captured matches ---

        if matches:
            print("DEBUG: matches found → sorting numerically")

            # convert matches into text list in numeric order
            matches_sorted = sorted(matches, key=lambda x: int(x[0]))
            print("DEBUG: matches_sorted =", matches_sorted)

            numbered_texts = []

            for idx, text in matches_sorted:
                print(f"\nDEBUG: processing numbered item {idx}: raw={repr(text)}")

                cleaned = re.sub(r'\s+', ' ', text).strip()
                print("DEBUG:   cleaned text =", repr(cleaned))

                if not cleaned.endswith(('.', '!', '?')):
                    print("DEBUG:   missing punctuation → adding '.'")
                    cleaned += '.'

                wc = len(cleaned.split())
                print("DEBUG:   word count =", wc)

                if wc >= 3:
                    print("DEBUG:   → accepted numbered item")
                    numbered_texts.append(cleaned)
                else:
                    print("DEBUG:   → rejected numbered item (too short)")

            print("\nDEBUG: numbered_texts =", numbered_texts)

            # if leading free text exists
            if m_lead:
                print("DEBUG: m_lead exists → append numbered items after leading text (avoid duplicates)")

                existing_lower = {s.lower() for s in out_lines}
                print("DEBUG: existing lowercased =", existing_lower)

                for t in numbered_texts:
                    if t.lower() not in existing_lower:
                        print("DEBUG:   appending numbered text:", t)
                        out_lines.append(t)
                    else:
                        print("DEBUG:   skipping duplicate numbered text:", t)

            else:
                print("DEBUG: no m_lead → using numbered_texts as core content")
                out_lines = numbered_texts.copy()

        else:
            print("DEBUG: no matches → skipping numbered text logic")

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

            for p in prep_normed:
                print(f"\nDEBUG: evaluating prep step: {p}")

                p_key = self._token_set(p)
                print("DEBUG:   p_key =", p_key)

                if not p_key:
                    print("DEBUG:   empty token set → skipping")
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

                    if sim >= 0.55:
                        print("DEBUG:     → similar (>=0.55), skipping")
                        exists_similar = True
                        break

                if not exists_similar:
                    print("DEBUG:   → adding to prepend list")
                    to_prepend.append(p)
                else:
                    print("DEBUG:   → NOT adding (duplicate-ish)")

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

        # finally dedupe aggressively but preserve readable originals
        out_lines = self._dedupe_steps(out_lines)
        print("DEBUG: out_lines after _dedupe_steps (final) =", out_lines)

        # --- FINAL SAFETY NORMALIZATION: ensure soak comes before grind ---
        final_lines = out_lines.copy()
        print("DEBUG: working copy final_lines =", final_lines)

        for tok in ['rice', 'urad', 'dal', 'semolina', 'besan', 'flour']:
            print(f"\nDEBUG: Checking ingredient token '{tok}'")
            #
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
                print(f"DEBUG:   order check soak_idx({soak_idx}) > grind_idx({grind_idx}) → {soak_idx > grind_idx}")

            if soak_idx is not None and grind_idx is not None and soak_idx > grind_idx:
                print("DEBUG:   → reordering needed (soak must come before grind)")

                line = final_lines.pop(soak_idx)
                print("DEBUG:   popped soak line:", line)

                final_lines.insert(grind_idx, line)
                print(f"DEBUG:   inserted soak line at index {grind_idx}")
            else:
                print("DEBUG:   no reorder needed for this token")

        out_lines = final_lines
        print("\nDEBUG: out_lines after final_lines reorder =", out_lines)

        # Compute AI confidence
        generated_text = generated if isinstance(generated, str) else str(generated)
        print("DEBUG: generated_text type:", type(generated))

        print("\nDEBUG: computing AI confidence")
        ai_conf = self.compute_ai_confidence(len(top_recipes), out_lines, generated_text)
        validator_conf = round(min(1.0, ai_conf * 0.8), 3)
        print(f"DEBUG: final ai_conf={ai_conf}, validator_conf={validator_conf}")

        # Prepare title
        base_title = top_recipes[0].title.split(':')[0].strip()
        title = f"Synthesized \u2014 {base_title} (for {requested_servings} servings)"
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
    # --------------------------------------------------------------------


        # ---------- VALIDATE ALL RECIPES ----------
    for r in km.recipes.recipes.values():
        if r.metadata.get("submitted_by") == "alice_trainer":
            km.validate_recipe(v, r.id, approved=True, feedback="Auto-approved", confidence=0.85)

    # ---------- Request Synthesis ----------
    try:
        synthesized = km.request_recipe(u, 'Khaman (variant with mixed leavening)', servings=5)
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
