"""
Token economy for rewarding contributors.
Implements RMDT token rewards for trainers and validators.
"""

from typing import Dict
from .models import User


class TokenEconomy:
    """Manages RMDT token rewards and ledger"""
    
    def __init__(self):
        self.ledger: Dict[str, float] = {}

    def reward_trainer_submission(self, trainer: User, amount: float = 1.0):
        """Reward a trainer for submitting a recipe"""
        trainer.credit(amount)
        self.ledger.setdefault(trainer.id, 0.0)
        self.ledger[trainer.id] += amount

    def reward_validator(self, validator: User, amount: float = 0.5):
        """Reward a validator for validating a recipe"""
        validator.credit(amount)
        self.ledger.setdefault(validator.id, 0.0)
        self.ledger[validator.id] += amount
