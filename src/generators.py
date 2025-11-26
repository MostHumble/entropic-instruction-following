import random
import json
import logging
import os
import nltk
from nltk.corpus import wordnet as wn
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class WordDataGenerator:
    def __init__(self, alphanumeric_only: bool = False, seeds: Optional[List[str]] = None):
        self.alphanumeric_only = alphanumeric_only
        self.seeds = seeds or ["food.n.02", "artifact.n.01"]
        self._ensure_nltk()
        # Filter for valid alpha words > 3 chars, and optionally alphanumeric
        self.all_words = list(
            set(
                w for w in wn.words()
                if w.isalpha() and len(w) > 3 and (not alphanumeric_only or w.isalnum())
            )
        )
        logger.info(
            f"Initialized Generator with {len(self.all_words)} words. "
            f"Alphanumeric only: {self.alphanumeric_only}, Seeds: {self.seeds}"
        )

    def _ensure_nltk(self):
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Downloading WordNet...")
            try:
                nltk.download('wordnet')
                nltk.download('omw-1.4')
            except Exception as e:
                logger.error(f"Failed to download NLTK data: {e}")
                raise

    def get_coherent_list(self, n: int) -> tuple[List[str], Dict]:
        """Generates semantically related words using seeds in order, no random fallback.
        
        Returns:
            Tuple of (word_list, metadata) where metadata contains seed contribution info.
        """
        pool = []
        needed = n
        seed_contributions = []
        
        for seed in self.seeds:
            if needed <= 0:
                break
                
            try:
                seed_synset = wn.synset(seed)
            except Exception as e:
                logger.warning(f"Seed synset '{seed}' not found: {e}")
                continue
                
            words = {w.replace('_', ' ') for s in seed_synset.closure(lambda s: s.hyponyms()) for w in s.lemma_names()}
            words = [w for w in words if '_' not in w]
            if self.alphanumeric_only:
                words = [w for w in words if w.isalnum()]
            
            # Remove duplicates already in pool
            words = [w for w in words if w not in pool]
            
            # Take only what we need from this seed
            to_take = min(needed, len(words))
            pool.extend(words[:to_take])
            
            # Track contribution
            seed_contributions.append({
                "seed": seed,
                "words_contributed": to_take,
                "words_available": len(words)
            })
            
            needed -= to_take
        
        if needed > 0:
            logger.error(f"Could not generate {n} coherent words. Only {n - needed} words available from seeds.")
            raise ValueError(f"Insufficient coherent words available. Requested {n}, got {n - needed}")
        
        metadata = {
            "total_requested": n,
            "total_generated": len(pool),
            "seed_contributions": seed_contributions,
            "seeds_used": [sc["seed"] for sc in seed_contributions]
        }
        
        return pool, metadata

    def get_chaotic_list(self, n: int) -> List[str]:
        available_words = self.all_words
        if self.alphanumeric_only:
            available_words = [w for w in available_words if w.isalnum()]
        if n > len(available_words):
            logger.error(f"Requested {n} words, but only {len(available_words)} available.")
            raise ValueError(f"Insufficient words available. Requested {n}, got {len(available_words)}")
        return random.sample(available_words, n)

    def generate_dataset(self, rule_counts: List[int], trials: int) -> List[Dict]:
        """Creates a standardized dataset for experimentation."""
        dataset = []
        for count in rule_counts:
            for trial_idx in range(trials):
                # Coherent
                coherent_words, coherent_metadata = self.get_coherent_list(count)
                dataset.append({
                    "id": f"coh_{count}_{trial_idx}_{random.randint(1000,9999)}",
                    "type": "coherent",
                    "count": count,
                    "words": coherent_words,
                    "metadata": coherent_metadata
                })
                # Chaotic
                dataset.append({
                    "id": f"chao_{count}_{trial_idx}_{random.randint(1000,9999)}",
                    "type": "chaotic",
                    "count": count,
                    "words": self.get_chaotic_list(count),
                    "metadata": {
                        "generation_method": "random_sample",
                        "alphanumeric_only": self.alphanumeric_only
                    }
                })
        return dataset

    def save_dataset(self, dataset: List[Dict], filepath: str):
        """Saves the dataset to a JSON file, ensuring the directory exists."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        logger.info(f"Dataset saved to {filepath}")