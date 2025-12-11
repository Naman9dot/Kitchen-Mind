#!/usr/bin/env python3
"""
Main entry point for KitchenMind system.
Contains example usage demonstrating all features.

Run:
    python main.py
"""

import pprint
from Module.controller import KitchenMind
from dataclasses import asdict


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
