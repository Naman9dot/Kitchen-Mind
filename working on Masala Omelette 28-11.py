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

        # After populating `missing`
        if "beaten" in all_text and "Eggs" not in {name for name, _ in missing}:
            missing.append(("Eggs", "pc"))

        if missing:
            print("DEBUG: missing ingredients detected:")
            for name, unit in missing:
                print("  -", repr(name), "unit=", repr(unit), "tokens=", toks_by_name.get(name))
        else:
            print("DEBUG: no missing ingredients -> returning unchanged out_lines")
            return out_lines

        # Identify candidate lines to remove
        indices_to_remove = set()
        add_line_pattern = re.compile(r'^\s*(add|mix|combine)\b.*$', flags=re.I)

        protected_indices = set()
        for i, s in enumerate(out_lines):
            if self.classify_phase(s) in ('cook', 'rest', 'finish') or self.has_time_or_temp(s):
                protected_indices.add(i)

        print("DEBUG: protected_indices (cook/rest/finish/time):", protected_indices)

        for i, s in enumerate(out_lines):
            if i in protected_indices:
                continue
            low = s.lower()
            # Prefer removing short add/mix/combine lines that mention any *missing* token
            if add_line_pattern.match(s):
                for toks in toks_by_name.values():  # NOTE: only missing ingredient tokens
                    if any(re.search(r'\b' + re.escape(tok) + r'\b', low) for tok in toks):
                        # extra guard
                        if self.classify_phase(s) in ('cook', 'rest', 'finish') or self.has_time_or_temp(s):
                            break
                        indices_to_remove.add(i)
                        break
            else:
                # Consider removing only if it's short-ish and mentions missing tokens
                for toks in toks_by_name.values():
                    if any(re.search(r'\b' + re.escape(tok) + r'\b', low) for tok in toks):
                        if len(low.split()) <= 6:
                            indices_to_remove.add(i)
                        break

        print("DEBUG: candidate indices_to_remove before protection check:", indices_to_remove)
        indices_to_remove = {i for i in indices_to_remove if i not in protected_indices}
        print("DEBUG: final indices_to_remove (after excluding protected):", indices_to_remove)

        # Debug: list current out_lines
        print("DEBUG: current out_lines:")
        for i, line in enumerate(out_lines):
            print(f"  [{i}] {repr(line)}")

        # --- Prefer heat/preheat over generic cook when picking insertion fallback ---
        # find index of first 'cook' or 'heat' step (or a step mentioning 'beaten')
        # but bias towards explicit heat/preheat if present
        heat_idx = None
        cook_idx = None
        for i, line in enumerate(out_lines):
            low = line.lower()
            if heat_idx is None and re.search(r'\b(heat|preheat)\b', low):
                heat_idx = i
            if cook_idx is None and (re.search(r'\bcook\b', low) or re.search(r'\bbake\b', low) or re.search(r'\bfry\b', low) or re.search(r'\bsimmer\b', low) or re.search(r'\bsteam\b', low)):
                cook_idx = i

        # if there is a 'beaten' mention earlier than both, prefer the first occurrence of beaten/cook/heat
        beaten_idx = None
        for i, line in enumerate(out_lines):
            if beaten_idx is None and 'beaten' in line.lower():
                beaten_idx = i

        # choose the first cook-like index with preference for heat index
        if heat_idx is not None:
            first_cook_idx = heat_idx
        elif cook_idx is not None:
            first_cook_idx = cook_idx
        elif beaten_idx is not None:
            first_cook_idx = beaten_idx
        else:
            first_cook_idx = None

        print("DEBUG: debug heat_idx =", heat_idx, "cook_idx =", cook_idx, "beaten_idx =", beaten_idx, "chosen first_cook_idx =", first_cook_idx)

        # Identify wet-add candidate (use merged_ings info)
        liquid_keys = {'water', 'milk', 'buttermilk', 'yogurt', 'curd', 'oil', 'olive oil', 'lemon juice', 'juice'}
        liquid_units = {'ml', 'l', 'litre', 'liter', 'cup', 'cups', 'tbsp', 'tsp'}
        wet_tokens_all = set()
        for ing in merged_ings:
            nlow = ing.name.strip().lower()
            unit = (ing.unit or "").strip().lower()
            toks = self._ingredient_tokens(ing.name)
            is_liquid = any(k in nlow for k in liquid_keys) or unit in liquid_units
            if is_liquid:
                for t in toks:
                    wet_tokens_all.add(t)

        wet_add_index = None
        if wet_tokens_all:
            for i, s in enumerate(out_lines):
                if i in protected_indices:
                    continue
                if add_line_pattern.match(s):
                    low = s.lower()
                    if any(w in low for w in liquid_keys) or any(re.search(r'\b' + re.escape(tok) + r'\b', low) for tok in wet_tokens_all):
                        wet_add_index = i
                        print("DEBUG: found wet-add candidate at index", i, "line:", repr(s))
                        break

        # If wet_add_index exists prefer removing it (and we'll insert before it)
        if wet_add_index is not None:
            indices_to_remove.add(wet_add_index)

        # Decide insertion index based on priority:
        # 1) wet_add_index (we will insert at that index)
        # 2) earliest removed index
        # 3) first cook/rest/finish/time index (with heat preference)
        # 4) append
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

        # Remove the selected indices (descending order to keep indices valid)
        removed_count_before = 0
        removed_indices_sorted = sorted(indices_to_remove, reverse=True)
        removed_indices_set = set(removed_indices_sorted)
        for idx in removed_indices_sorted:
            try:
                popped = out_lines.pop(idx)
                removed_count_before += 1
                print(f"DEBUG: popped out_lines[{idx}] = {repr(popped)}")
            except Exception as exc:
                print(f"DEBUG: failed to pop index {idx}: {exc}")

        print("DEBUG: out_lines AFTER removal:")
        for i, line in enumerate(out_lines):
            print(f"  [{i}] {repr(line)}")

        # --- Reorder heat/cook if cook-with-prep appears before heat (post-removal fix) ---
        # If a cook step that references "beaten" or 'beaten eggs' occurs before a heat step,
        # move that heat step so it occurs before the cook step.
        cook_idx_post = None
        heat_idx_post = None
        for i, line in enumerate(out_lines):
            low = line.lower()
            if cook_idx_post is None and re.search(r'\b(cook|bake|fry|simmer|steam)\b', low):
                cook_idx_post = i
            if heat_idx_post is None and re.search(r'\b(heat|preheat)\b', low):
                heat_idx_post = i

        if cook_idx_post is not None and heat_idx_post is not None and cook_idx_post < heat_idx_post:
            try:
                heat_line = out_lines.pop(heat_idx_post)
                out_lines.insert(cook_idx_post, heat_line)
                print(f"DEBUG: moved heat line from index {heat_idx_post} to {cook_idx_post} to ensure heating before cooking")
                # adjust insert_idx if we moved a line that affects it
                if insert_idx is not None:
                    if heat_idx_post < insert_idx and cook_idx_post >= insert_idx:
                        # heat was removed before insert point and inserted after — adjust accordingly
                        insert_idx -= 1
                    elif heat_idx_post > insert_idx and cook_idx_post <= insert_idx:
                        insert_idx += 1
            except Exception as exc:
                print("DEBUG: failed to reorder heat/cook lines:", exc)

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
            if any(k in nlow for k in liquid_keys) or (unit in liquid_units or unit in {'ml', 'l'}):
                wet.append(name)
            else:
                if unit in {'pc', 'pcs', 'piece'} and any(k in nlow for k in ('egg', 'eggs', 'egg')):
                    wet.append(name)
                else:
                    dry.append(name)

        # Also include wet display names from merged ingredients (even if not missing) if they exist and dry exists
        merged_wet_names = []
        for ing in merged_ings:
            nlow = ing.name.strip().lower()
            unit = (ing.unit or "").strip().lower()
            if any(k in nlow for k in liquid_keys) or unit in liquid_units:
                merged_wet_names.append(ing.name)
        if not wet and merged_wet_names and dry:
            preferred = None
            for w in merged_wet_names:
                if 'warm' in w.lower():
                    preferred = w
                    break
            if not preferred:
                preferred = merged_wet_names[0]
            if preferred not in wet:
                wet.append(preferred)

        def short_label(full_name: str, keep_warm_for_liquid: bool = True) -> str:
            s = full_name.strip()
            s = re.sub(r'[\(\)\[\]\,]', ' ', s)
            s = s.replace('-', ' ')
            s = re.sub(r'\s+', ' ', s).strip()
            if not s:
                return full_name.title()
            parts = s.split()
            if keep_warm_for_liquid and any(k in s.lower() for k in ('water', 'milk', 'buttermilk', 'yogurt', 'oil', 'juice', 'curd')):
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

        # Build combined instruction (this is your add_step)
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

        print("DEBUG: classified missing into dry and wet:")
        print("  dry:", [d for d in dry])
        print("  wet:", [w for w in wet])

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
        print(f"DEBUG: _collapse_repeated_words INPUT = {repr(s)}")
        out = re.sub(r'\b(\w+)(?: \1\b)+', r'\1', s, flags=re.I)
        print(f"DEBUG: _collapse_repeated_words OUTPUT = {repr(out)}")
        return out


    def _normalize_for_dedupe(self, s: str) -> str:
        print(f"DEBUG: _normalize_for_dedupe START input={repr(s)}")

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

        # remove punctuation and digits
        before_clean = s
        s = re.sub(r'[^a-z\s]', ' ', s)
        print(f"DEBUG: Removed punctuation/digits: {repr(before_clean)} -> {repr(s)}")

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


    def _dedupe_steps(self, steps: List[str]) -> List[str]:
        """Preserve first occurrence; remove later steps that normalize-identically."""
        print("DEBUG: _dedupe_steps START")
        seen = set()
        out = []
        for idx, s in enumerate(steps):
            print(f"DEBUG:   Step[{idx}] original={repr(s)}")
            key = self._normalize_for_dedupe(s)
            print(f"DEBUG:   Step[{idx}] normalized key={repr(key)}")
            if not key:
                print(f"DEBUG:   -> SKIP (empty key)")
                continue
            if key in seen:
                print(f"DEBUG:   -> SKIP (duplicate key)")
                continue
            seen.add(key)
            print(f"DEBUG:   -> KEEP")
            out.append(s)

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

    # helper to split composite step into prep + cook if it contains both
    _prep_verbs = r'\b(beat|whisk|mix|combine|stir|fold|knead|blend|whisked|beaten)\b'
    _cook_verbs = r'\b(heat|cook|fry|sauté|saute|bake|roast|grill|steam|simmer)\b'

    def _split_prep_and_cook(raw_steps):
        out = []
        for s in raw_steps:
            low = s.lower()
            if re.search(_prep_verbs, low) and re.search(_cook_verbs, low):
                # attempt to split on common conjunctions/commas/then
                parts = re.split(r'\b(?:then|and then|, then|;| and | then )\b', s, flags=re.IGNORECASE)
                # find prep part (first containing prep verb), and cook part (first containing cook verb after)
                prep = None
                cook = None
                for p in parts:
                    if prep is None and re.search(_prep_verbs, p, re.IGNORECASE):
                        prep = p.strip()
                    elif cook is None and re.search(_cook_verbs, p, re.IGNORECASE):
                        cook = p.strip()
                if prep and cook:
                    out.append(prep if prep.endswith('.') else prep + '.')
                    out.append(cook if cook.endswith('.') else cook + '.')
                    continue
            out.append(s)
        return out



    def merge_semantic_steps(self, steps: List[str]) -> List[str]:
        print("DEBUG: merge_semantic_steps START")
        print("DEBUG: input steps =", steps)


        # helper to split composite step into prep + cook if it contains both
        _prep_verbs = r'\b(beat|whisk|mix|combine|stir|fold|knead|blend|whisked|beaten)\b'
        _cook_verbs = r'\b(heat|cook|fry|sauté|saute|bake|roast|grill|steam|simmer)\b'

        def _split_prep_and_cook(raw_steps):
            out = []
            for s in raw_steps:
                low = s.lower()
                if re.search(_prep_verbs, low) and re.search(_cook_verbs, low):
                    # attempt to split on common conjunctions/commas/then
                    parts = re.split(r'\b(?:then|and then|, then|;| and | then )\b', s, flags=re.IGNORECASE)
                    # find prep part (first containing prep verb), and cook part (first containing cook verb after)
                    prep = None
                    cook = None
                    for p in parts:
                        if prep is None and re.search(_prep_verbs, p, re.IGNORECASE):
                            prep = p.strip()
                        elif cook is None and re.search(_cook_verbs, p, re.IGNORECASE):
                            cook = p.strip()
                    if prep and cook:
                        out.append(prep if prep.endswith('.') else prep + '.')
                        out.append(cook if cook.endswith('.') else cook + '.')
                        continue
                out.append(s)
            return out

        # Usage (before normalization):
        steps = _split_prep_and_cook(steps)


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
        for s in norm_steps:
            low = s.lower()
            if any(v in low for v in ["mix", "whisk", "combine", "stir"]):
                if re.search(flour_pattern, low) and re.search(yogurt_pattern, low):
                    print(f"DEBUG: batter pattern matched in {repr(s)}")
                    m_flour = re.search(flour_pattern, low)
                    m_yog = re.search(yogurt_pattern, low)
                    flour_txt = (m_flour.group(1) if m_flour else "flour").strip().title()
                    yog_txt = (m_yog.group(1) if m_yog else "yogurt").strip().title()

                    batter_step = (
                        f"Whisk the {flour_txt} and {yog_txt} together, adding water gradually to form a smooth batter."
                    )
                    print("DEBUG: batter_step CREATED =", batter_step)
                    break

        # ------------------- Detect add_step (summary fallback) -------------------
        key_add_names = ["water", "eno", "baking soda", "sugar", "salt"]
        seen_add = []

        print("DEBUG: scanning for add_step...")
        for s in norm_steps:
            low = s.lower()
            if "add" in low:
                for name in key_add_names:
                    if name in low and name not in seen_add:
                        seen_add.append(name)
                        print(f"DEBUG: detected add ingredient = {name}")

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
            print("DEBUG: add_step CREATED =", add_step)

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

        # ------------------- New Merge logic (streaming) -------------------
        print("DEBUG: Starting merge stage (streaming merge)...")
        merged = []

        # Optionally prepend batter_step if created and not duplicative
        if batter_step:
            low_b = batter_step.lower()
            duplicate_found = any(
                (low_b in s.lower()) or
                (s.lower() in low_b) or
                (self._normalize_for_dedupe(s) == self._normalize_for_dedupe(batter_step))
                for s in norm_steps
            )
            print(f"DEBUG: batter_step duplicate_found = {duplicate_found}")
            if not duplicate_found:
                merged.append(self._normalize_step_text(batter_step))
                print("DEBUG: merged ADD batter_step at start")

        # Helper: try to use existing classifier on self, else fallback
        def _classify_phase(step_text: str) -> str:
            try:
                return self.classify_phase(step_text)
            except Exception:
                t = step_text.lower()
                if any(k in t for k in ['bake', 'roast', 'cook', 'fry', 'simmer', 'saute']):
                    return 'cook'
                if any(k in t for k in ['rest', 'rise', 'proof', 'ferment']):
                    return 'rest'
                if any(k in t for k in ['finish', 'serve', 'garnish']):
                    return 'finish'
                if any(k in t for k in ['add', 'mix', 'combine', 'stir', 'knead', 'whisk']):
                    return 'add'
                return 'other'

        # Helper: detect explicit time/temperature; try self.has_time_or_temp else simple regex
        time_temp_re = re.compile(r'\b(\d+\s*(?:min|mins|minutes|h|hr|hour|hours)|\d+°C|\d+°F|\bfor\b.*\bminutes\b)', re.IGNORECASE)
        def _has_time_or_temp(step_text: str) -> bool:
            try:
                return self.has_time_or_temp(step_text)
            except Exception:
                return bool(time_temp_re.search(step_text))

        add_buffer = []

        def flush_add_buffer():
            if not add_buffer:
                return
            # Join adjacent add/mix steps into a single, compact instruction.
            # Keep the original phrasing but collapse into one sentence sequence.
            joined = " ".join(add_buffer).strip()
            joined = re.sub(r'\s+', ' ', joined)
            if not joined.endswith('.'):
                joined += '.'
            merged.append(self._normalize_step_text(joined))
            print("DEBUG: flushed add_buffer ->", joined)
            add_buffer.clear()

        for s in norm_steps:
            s_norm = self._normalize_step_text(s)
            phase = _classify_phase(s_norm)
            protected = (phase in {'cook', 'rest', 'finish'}) or _has_time_or_temp(s_norm)

            if phase == 'add' and not protected:
                # buffer adjacent add/mix steps (we will merge them later)
                add_buffer.append(s_norm)
                print("DEBUG: buffered add step:", s_norm)
                continue
            else:
                # flush any buffered add steps before appending a protected/non-add step
                flush_add_buffer()
                merged.append(s_norm)
                print("DEBUG: appended protected/non-add step:", s_norm)

        # flush remaining add buffer at end
        flush_add_buffer()

        # If nothing merged (shouldn't happen because flush adds), fallback to previous logic:
        if not merged:
            print("DEBUG: merged empty — returning norm_steps")
            return norm_steps

        # If earlier detection created a standalone add_step and no add content was present in norm_steps,
        # we keep the earlier behavior of adding the summary add_step (but only if it is not already present).
        if not any('add' in x.lower() for x in merged) and add_step:
            # append as a fallback summary add_step
            merged.append(self._normalize_step_text(add_step))
            print("DEBUG: merged appended fallback add_step:", add_step)

        # If cook_step was detected by 'steam' heuristic and not present in merged, append it (not overwrite).
        if cook_step and not any('steam' in x.lower() for x in merged):
            merged.append(self._normalize_step_text(cook_step))
            print("DEBUG: merged appended cook_step fallback:", cook_step)

        print("DEBUG: merged result before fallback =", merged)

        print("DEBUG: merge_semantic_steps FINAL =", merged)
        return merged



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


    def canonicalize_step_text(self, text: str) -> str:
        print(f"DEBUG: canonicalize_step_text input={repr(text)}")
        out = text
        for alias, canon in self.CANONICAL_NAMES.items():
            pattern = r'\b' + re.escape(alias) + r'\b'
            new_out = re.sub(pattern, canon.title(), out, flags=re.I)
            if new_out != out:
                print(f"DEBUG:   replaced alias '{alias}' -> '{canon.title()}'")
            out = new_out
        print(f"DEBUG: canonicalize_step_text output={repr(out)}")
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

        def __init__(self, model_name: str = 'lmsys/fastchat-t5-3b-v1.0'):
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
                self._pipe = pipeline(
                    'text2text-generation',
                    model=model,
                    tokenizer=tokenizer,
                    device=-1
                )
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

    def reorder_steps(self, steps: List[str]) -> List[str]:
        buckets: Dict[str, List[Tuple[int,str]]] = {'prep': [], 'mix': [], 'rest': [], 'cook': [], 'finish': []}
        for i, s in enumerate(steps):
            phase = self.classify_phase(s)
            buckets.setdefault(phase, []).append((i, s))
        ordered = []
        for phase in ['prep', 'mix', 'rest', 'cook', 'finish']:
            items = sorted(buckets.get(phase, []), key=lambda x: x[0])
            ordered.extend([s for _, s in items])
        result = ordered if ordered else steps
        print("DEBUG: reorder_steps() input =", steps)
        print("DEBUG: reorder_steps() output =", result)
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


    def synthesize(self, top_recipes: List[Recipe], requested_servings: int,
               llm_model: str = 'lmsys/fastchat-t5-3b-v1.0', reorder: bool = True) -> Recipe:
        print("DEBUG: synthesize() start")
        if not top_recipes:
            raise ValueError("No recipes provided for synthesis")

        merged_ings = self.merge_ingredients(top_recipes, requested_servings)
        print("DEBUG: merged_ings =", [asdict(ing) for ing in merged_ings])
        prep_from_ings = self.generate_prep_from_ingredients(merged_ings)
        print("DEBUG: prep_from_ings =", prep_from_ings)

        raw_steps = []
        for r in top_recipes:
            for s in r.steps:
                s_norm = self._normalize_step_text(s)
                s_norm = self.canonicalize_step_text(s_norm)
                raw_steps.append(s_norm)

        raw_steps = prep_from_ings + raw_steps
        print("DEBUG: raw_steps (combined) =", raw_steps)
        src = "\n".join(f"- {s}" for s in raw_steps)

        prompt = (
            f"Combine the following cooking actions into one clear, merged recipe for {requested_servings} servings.\n\n"
            f"Write 4\u20138 numbered steps. Keep steps short (one sentence each). Do NOT add new ingredients or quantities.\n"
            f"Try to include times/temperatures when they are present in the source actions.\n\n"
            f"Source actions:\n{src}\n\n"
            f"Output (begin with '1. '):\n1. "
        )
        print("DEBUG: prompt constructed (truncated):", prompt[:400].replace("\n", "\\n"))

        llm = self.FreeOpenLLM(model_name=llm_model)
        print("DEBUG: llm available?", llm.available(), "llm init error:", getattr(llm, "_init_error", None))
        if not llm.available():
            print("DEBUG: entering fallback (no llm) path")
            fallback_steps = []
            seen = set()
            for s in raw_steps:
                s_clean = re.sub(r'\s+', ' ', s).strip()
                if s_clean.lower() not in seen:
                    seen.add(s_clean.lower())
                    fallback_steps.append(s_clean)
            out_lines = fallback_steps[:6] if fallback_steps else ["Combine ingredients and cook as directed."]
            print("DEBUG: fallback initial out_lines =", out_lines)
            if reorder:
                out_lines = self.reorder_steps(out_lines)
                print("DEBUG: out_lines after reorder =", out_lines)
            out_lines = self.merge_semantic_steps(out_lines)
            print("DEBUG: out_lines after merge_semantic_steps =", out_lines)
            out_lines = self.remove_invalid_leavening_from_steps(out_lines, merged_ings)
            print("DEBUG: out_lines after remove_invalid_leavening_from_steps =", out_lines)
            # ensure prep lines survive
            if prep_from_ings:
                # prepend in original order
                prep_normed = [self._normalize_step_text(p) for p in prep_from_ings]
                for p in prep_normed[::-1]:
                    if not any(p.lower() in s.lower() for s in out_lines):
                        out_lines.insert(0, p)
                print("DEBUG: out_lines after prep prepend =", out_lines)
            out_lines = [self._collapse_repeated_words(s).strip() for s in out_lines]
            out_lines = self.ensure_ingredient_coverage(out_lines, merged_ings)
            print("DEBUG: out_lines after ensure_ingredient_coverage =", out_lines)
            # finally dedupe aggressively but preserve readable originals
            out_lines = self._dedupe_steps(out_lines)
            print("DEBUG: out_lines after _dedupe_steps =", out_lines)

            generated_text = "\n".join(out_lines)
            ai_conf = self.compute_ai_confidence(len(top_recipes), out_lines, generated_text)
            validator_conf = round(min(1.0, ai_conf * 0.8), 3)
            print(f"DEBUG: fallback ai_conf={ai_conf}, validator_conf={validator_conf}")
            title_base = top_recipes[0].title.split(':')[0].strip()
            title = f"Synthesized \u2014 {title_base} (for {requested_servings} servings)"
            meta = {
                "sources": [r.id for r in top_recipes],
                "ai_confidence": ai_conf,
                "synthesis_method": f"fallback:no-llm"
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

        gen_kwargs = {
            "max_new_tokens": 180,
            "do_sample": True,
            "temperature": 0.35,
            "top_p": 0.9,
        }
        print("DEBUG: calling llm.generate with gen_kwargs =", gen_kwargs)
        generated = llm.generate(prompt, **gen_kwargs)
        print("DEBUG: raw generated output (truncated) =", (generated[:1000] if isinstance(generated, str) else str(generated)[:1000]))

        pattern = r'^\s*(\d+)\.\s*(.+?)(?=\n\s*\d+\.|\Z)'
        matches = re.findall(pattern, generated, flags=re.S | re.M)
        print("DEBUG: regex matches found =", matches)

        out_lines = []
        for _, text in matches:
            text = text.strip()
            text = re.sub(r'\s+', ' ', text)
            if not text.endswith(('.', '!', '?')):
                text = text + '.'
            if len(text.split()) >= 3:
                out_lines.append(text)
        print("DEBUG: out_lines after parsing numbered steps =", out_lines)

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
        if reorder:
            out_lines = self.reorder_steps(out_lines)
            print("DEBUG: out_lines after reorder =", out_lines)
        out_lines = self.merge_semantic_steps(out_lines)
        print("DEBUG: out_lines after merge_semantic_steps =", out_lines)
        out_lines = self.remove_invalid_leavening_from_steps(out_lines, merged_ings)
        print("DEBUG: out_lines after remove_invalid_leavening_from_steps =", out_lines)
        # ensure prep lines survive

        if prep_from_ings:
            # prepend in original order
            prep_normed = [self._normalize_step_text(p) for p in prep_from_ings]
            for p in prep_normed[::-1]:
                if not any(p.lower() in s.lower() for s in out_lines):
                    out_lines.insert(0, p)
            print("DEBUG: out_lines after prep prepend (final) =", out_lines)

        out_lines = [self._collapse_repeated_words(s).strip() for s in out_lines]
        out_lines = self.ensure_ingredient_coverage(out_lines, merged_ings)
        print("DEBUG: out_lines after ensure_ingredient_coverage (final) =", out_lines)
        # finally dedupe aggressively but preserve readable originals
        out_lines = self._dedupe_steps(out_lines)
        print("DEBUG: out_lines after _dedupe_steps (final) =", out_lines)


        generated_text = generated if isinstance(generated, str) else str(generated)
        ai_conf = self.compute_ai_confidence(len(top_recipes), out_lines, generated_text)
        validator_conf = round(min(1.0, ai_conf * 0.8), 3)
        print(f"DEBUG: final ai_conf={ai_conf}, validator_conf={validator_conf}")

        base_title = top_recipes[0].title.split(':')[0].strip()
        title = f"Synthesized \u2014 {base_title} (for {requested_servings} servings)"

        meta = {
            "sources": [r.id for r in top_recipes],
            "ai_confidence": ai_conf,
            "synthesis_method": f"llm:{llm_model}"
        }
        print("DEBUG: returning LLM Recipe with meta =", meta)

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
        synthesized = km.request_recipe(u, 'Masala Omelette', servings=5)
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
