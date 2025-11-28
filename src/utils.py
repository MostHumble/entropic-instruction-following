from src.strategies import StandardStoryStrategy, IndexedListStrategy

import json
import logging

logger = logging.getLogger(__name__)


def get_strategy(strategy_name: str, model_name: str):
    """Load strategy with model-specific chat template."""
    strategies = {
        "Standard_Story": StandardStoryStrategy,
        "Indexed_List": IndexedListStrategy,
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")

    strategy_class = strategies[strategy_name]
    return strategy_class(model_name=model_name)


def load_dataset(dataset_path: str) -> list:
    """Load dataset from JSON file."""
    try:
        with open(dataset_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Dataset not found at {dataset_path}. Run generate_data.py first!")
        raise