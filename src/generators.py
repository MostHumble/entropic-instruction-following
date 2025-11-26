import random
import json
import logging
import nltk
from nltk.corpus import wordnet as wn
from typing import List, Dict

logger = logging.getLogger(__name__)

class WordDataGenerator:
    def __init__(self):
        self._ensure_nltk()
        # Filter for valid alpha words > 3 chars
        self.all_words = list(set(w for w in wn.words() if w.isalpha() and len(w) > 3))
        logger.info(f"Initialized Generator with {len(self.all_words)} words.")

    def _ensure_nltk(self):
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Downloading WordNet...")
            nltk.download('wordnet')
            nltk.download('omw-1.4')

    def get_coherent_list(self, n: int) -> List[str]:
        """Generates semantically related words (Food -> Artifacts -> Random)."""
        seed_synset = wn.synset('food.n.02') 
        pool = {w.replace('_', ' ') for s in seed_synset.closure(lambda s: s.hyponyms()) for w in s.lemma_names()}
        pool = [w for w in pool if '_' not in w] # clean underscores

        if len(pool) < n:
            seed_2 = wn.synset('artifact.n.01')
            more = {w.replace('_', ' ') for s in seed_2.closure(lambda s: s.hyponyms()) for w in s.lemma_names()}
            pool.extend([w for w in more if '_' not in w])
            
        # Fallback to random if still short
        if len(pool) < n:
            pool.extend(random.sample(self.all_words, n - len(pool)))
            
        return random.sample(pool, n)

    def get_chaotic_list(self, n: int) -> List[str]:
        return random.sample(self.all_words, n)

    def generate_dataset(self, rule_counts: List[int], trials: int) -> List[Dict]:
        """Creates a standardized dataset for experimentation."""
        dataset = []
        for count in rule_counts:
            for _ in range(trials):
                # Coherent
                dataset.append({
                    "id": f"coh_{count}_{random.randint(1000,9999)}",
                    "type": "coherent",
                    "count": count,
                    "words": self.get_coherent_list(count)
                })
                # Chaotic
                dataset.append({
                    "id": f"chao_{count}_{random.randint(1000,9999)}",
                    "type": "chaotic",
                    "count": count,
                    "words": self.get_chaotic_list(count)
                })
        return dataset

    def save_dataset(self, dataset: List[Dict], filepath: str):
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        logger.info(f"Dataset saved to {filepath}")