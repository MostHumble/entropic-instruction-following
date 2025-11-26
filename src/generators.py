import random
import json
import logging
import os
import nltk
from nltk.corpus import wordnet as wn
from typing import List, Dict, Optional, Tuple
import re

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

    def parse_pattern(self, pattern: str) -> Tuple[str, List[str]]:
        """
        Parse a pattern string to determine mode and components.
        
        Returns:
            Tuple of (mode, components) where mode is 'repeating' or 'blocked'
            
        Examples:
            'cr' -> ('repeating', ['c', 'r'])
            'c|r|c' -> ('blocked', ['c', 'r', 'c'])
            'c1c2' -> ('repeating', ['c1', 'c2'])
            'c1|c2' -> ('blocked', ['c1', 'c2'])
        """
        if '|' in pattern:
            return ('blocked', pattern.split('|'))
        else:
            # Parse repeating pattern with optional numbers
            components = re.findall(r'[cr]\d*', pattern)
            return ('repeating', components)

    def get_words_for_component(self, component: str, n: int) -> Tuple[List[str], Dict]:
        """
        Get words for a specific component (e.g., 'c', 'r', 'c1', 'c2').
        
        Args:
            component: Component identifier ('c', 'r', 'c1', 'c2', etc.)
            n: Number of words needed
            
        Returns:
            Tuple of (words, metadata)
        """
        if component == 'r':
            # Random words
            words = self.get_chaotic_list(n)
            metadata = {
                "type": "random",
                "count": n
            }
            return words, metadata
        elif component.startswith('c'):
            # Coherent words, possibly with seed index
            seed_idx = int(component[1:]) if len(component) > 1 else None
            
            if seed_idx is not None and seed_idx < len(self.seeds):
                # Use specific seed
                seeds = [self.seeds[seed_idx]]
            else:
                # Use all seeds
                seeds = self.seeds
                
            words, metadata = self.get_coherent_list(n, seeds=seeds)
            return words, metadata
        else:
            raise ValueError(f"Unknown component: {component}")

    def get_coherent_list(self, n: int, seeds: Optional[List[str]] = None) -> Tuple[List[str], Dict]:
        """Generates semantically related words using seeds in order, no random fallback.
        
        Args:
            n: Number of words to generate
            seeds: Optional list of seeds to use (defaults to self.seeds)
        
        Returns:
            Tuple of (word_list, metadata) where metadata contains seed contribution info.
        """
        seeds = seeds or self.seeds
        pool = []
        needed = n
        seed_contributions = []
        
        for seed in seeds:
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

    def get_mixed_list(self, n: int, pattern: str) -> Tuple[List[str], Dict]:
        """
        Generate a mixed word list based on a pattern.
        
        Args:
            n: Total number of words
            pattern: Pattern string (e.g., 'cr', 'c|r|c', 'c1c2', 'cccr')
            
        Returns:
            Tuple of (word_list, metadata)
            
        Examples:
            get_mixed_list(12, 'cr') -> alternating coherent/random
            get_mixed_list(12, 'c|r|c') -> 4 coherent, 4 random, 4 coherent
            get_mixed_list(12, 'cccr') -> 3 coherent, 1 random, repeated
        """
        mode, components = self.parse_pattern(pattern)
        
        if mode == 'repeating':
            # Cycle through components until we have n words
            result = []
            component_metadata = []
            pattern_length = len(components)
            
            for i in range(n):
                component = components[i % pattern_length]
                words, metadata = self.get_words_for_component(component, 1)
                result.extend(words)
                component_metadata.append({
                    "position": i,
                    "component": component,
                    "word": words[0]
                })
            
            final_metadata = {
                "pattern": pattern,
                "mode": "repeating",
                "total_words": n,
                "component_details": component_metadata
            }
            
        else:  # blocked mode
            # Split n into equal blocks
            num_blocks = len(components)
            block_size = n // num_blocks
            remainder = n % num_blocks
            
            result = []
            block_metadata = []
            
            for idx, component in enumerate(components):
                # Distribute remainder across first blocks
                current_block_size = block_size + (1 if idx < remainder else 0)
                words, metadata = self.get_words_for_component(component, current_block_size)
                result.extend(words)
                
                block_metadata.append({
                    "block_index": idx,
                    "component": component,
                    "start_position": len(result) - current_block_size,
                    "end_position": len(result),
                    "size": current_block_size,
                    "metadata": metadata
                })
            
            final_metadata = {
                "pattern": pattern,
                "mode": "blocked",
                "total_words": n,
                "blocks": block_metadata
            }
        
        return result, final_metadata

    def generate_dataset(self, rule_counts: List[int], trials: int, patterns: Optional[List[str]] = None) -> List[Dict]:
        """
        Creates a standardized dataset for experimentation.
        
        Args:
            rule_counts: List of word counts to test
            trials: Number of trials per count
            patterns: Optional list of mixing patterns to include (e.g., ['c', 'r', 'cr', 'c|r|c'])
        """
        patterns = patterns or ['c', 'r']  # Default: pure coherent and pure random
        dataset = []
        
        for count in rule_counts:
            for trial_idx in range(trials):
                for pattern in patterns:
                    words, metadata = self.get_mixed_list(count, pattern)
                    
                    # Determine type label
                    if pattern == 'c':
                        type_label = 'coherent'
                    elif pattern == 'r':
                        type_label = 'chaotic'
                    else:
                        type_label = f'mixed_{pattern}'
                    
                    dataset.append({
                        "id": f"{type_label}_{count}_{trial_idx}_{random.randint(1000,9999)}",
                        "type": type_label,
                        "pattern": pattern,
                        "count": count,
                        "words": words,
                        "metadata": metadata
                    })
        
        return dataset

    def save_dataset(self, dataset: List[Dict], filepath: str):
        """Saves the dataset to a JSON file, ensuring the directory exists."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        logger.info(f"Dataset saved to {filepath}")