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

    def get_coherent_list(self, n: int, seed: str) -> Tuple[List[str], Dict]:
        """Generates semantically related words from a single seed, with shuffling.
    
        Args:
            n: Number of words to generate
            seed: Single seed synset to use
    
        Returns:
            Tuple of (word_list, metadata) where metadata contains seed info.
        """
        try:
            seed_synset = wn.synset(seed)
        except Exception as e:
            logger.error(f"Seed synset '{seed}' not found: {e}")
            raise ValueError(f"Invalid seed: {seed}")
        
        # Get all words from this seed's hyponym closure
        words = {w.replace('_', ' ') for s in seed_synset.closure(lambda s: s.hyponyms()) for w in s.lemma_names()}
        words = [w for w in words if '_' not in w]
        if self.alphanumeric_only:
            words = [w for w in words if w.isalnum()]
        
        # Check if we have enough
        if len(words) < n:
            logger.warning(f"Seed '{seed}' only has {len(words)} words, need {n}. Skipping.")
            raise ValueError(f"Insufficient words for seed '{seed}': has {len(words)}, need {n}")
        
        # Shuffle the full pool
        random.shuffle(words)
        
        metadata = {
            "seed": seed,
            "words_available": len(words),
            "words_requested": n,
            "words_pool": words  # Store full shuffled pool for sampling
        }
        
        return words, metadata

    def get_chaotic_list(self, n: int) -> List[str]:
        available_words = self.all_words
        if self.alphanumeric_only:
            available_words = [w for w in available_words if w.isalnum()]
        if n > len(available_words):
            logger.error(f"Requested {n} words, but only {len(available_words)} available.")
            raise ValueError(f"Insufficient words available. Requested {n}, got {len(available_words)}")
        return random.sample(available_words, n)

    def get_mixed_list(self, n: int, pattern: str, seed: str, coherent_pool: Optional[List[str]] = None) -> Tuple[List[str], Dict]:
        """
        Generate a mixed word list based on a pattern, using a specific seed.
        
        Args:
            n: Total number of words
            pattern: Pattern string (e.g., 'cr', 'c|r|c', 'c1c2', 'cccr')
            seed: Specific seed to use for coherent parts
            coherent_pool: Pre-generated shuffled pool of coherent words (optional)
            
        Returns:
            Tuple of (word_list, metadata)
        """
        mode, components = self.parse_pattern(pattern)
        
        # Count words needed per component type
        component_counts = {}
        
        if mode == 'repeating':
            pattern_length = len(components)
            for i in range(n):
                component = components[i % pattern_length]
                component_counts[component] = component_counts.get(component, 0) + 1
                
        else:  # blocked mode
            num_blocks = len(components)
            block_size = n // num_blocks
            remainder = n % num_blocks
            
            for idx, component in enumerate(components):
                current_block_size = block_size + (1 if idx < remainder else 0)
                component_counts[component] = component_counts.get(component, 0) + current_block_size
        
        # Fetch words for each component type
        component_pools = {}
        component_metadata = {}
        
        for component, count in component_counts.items():
            if component == 'r':
                words = self.get_chaotic_list(count)
                metadata = {"type": "random", "count": count}
                component_pools[component] = words
                component_metadata[component] = metadata
            elif component.startswith('c'):
                # Use provided pool or generate new one
                if coherent_pool is None:
                    pool, metadata = self.get_coherent_list(count, seed)
                    component_pools[component] = pool[:count]
                    component_metadata[component] = metadata
                else:
                    # Sample from pre-generated pool
                    if len(coherent_pool) < count:
                        raise ValueError(f"Coherent pool has {len(coherent_pool)} words, need {count}")
                    component_pools[component] = coherent_pool[:count]
                    coherent_pool = coherent_pool[count:]  # Remove used words
                    component_metadata[component] = {
                        "type": "coherent",
                        "seed": seed,
                        "count": count
                    }
            else:
                raise ValueError(f"Unknown component: {component}")
        
        # Build the final sequence
        result = []
        component_indices = {comp: 0 for comp in component_pools}
        position_details = []
        
        if mode == 'repeating':
            pattern_length = len(components)
            for i in range(n):
                component = components[i % pattern_length]
                idx = component_indices[component]
                word = component_pools[component][idx]
                result.append(word)
                
                position_details.append({
                    "position": i,
                    "component": component,
                    "word": word
                })
                
                component_indices[component] += 1
            
            final_metadata = {
                "pattern": pattern,
                "mode": "repeating",
                "total_words": n,
                "seed": seed,
                "component_counts": component_counts,
                "component_metadata": component_metadata,
                "position_details": position_details
            }
            
        else:  # blocked mode
            block_metadata = []
            num_blocks = len(components)
            block_size = n // num_blocks
            remainder = n % num_blocks
            
            for idx, component in enumerate(components):
                current_block_size = block_size + (1 if idx < remainder else 0)
                start_pos = len(result)
                
                comp_idx = component_indices[component]
                block_words = component_pools[component][comp_idx:comp_idx + current_block_size]
                result.extend(block_words)
                component_indices[component] += current_block_size
                
                block_metadata.append({
                    "block_index": idx,
                    "component": component,
                    "start_position": start_pos,
                    "end_position": len(result),
                    "size": current_block_size,
                    "words": block_words
                })
            
            final_metadata = {
                "pattern": pattern,
                "mode": "blocked",
                "total_words": n,
                "seed": seed,
                "component_counts": component_counts,
                "component_metadata": component_metadata,
                "blocks": block_metadata
            }
        
        return result, final_metadata

    def generate_dataset(self, rule_counts: List[int], trials_per_seed: int, patterns: Optional[List[str]] = None) -> List[Dict]:
        """
        Creates a standardized dataset for experimentation.
        
        Args:
            rule_counts: List of word counts to test
            trials_per_seed: Number of trials per seed (samples from shuffled pool)
            patterns: Optional list of mixing patterns to include
        """
        patterns = patterns or ['c', 'r']
        dataset = []
        
        for count in rule_counts:
            for seed_idx, seed in enumerate(self.seeds):
                # Generate coherent pool once per seed per count
                try:
                    coherent_pool, _ = self.get_coherent_list(count * trials_per_seed * len(patterns), seed)
                except ValueError as e:
                    logger.warning(f"Skipping seed '{seed}' for count {count}: {e}")
                    continue
                
                # Shuffle once for all trials from this seed
                random.shuffle(coherent_pool)
                
                for trial_idx in range(trials_per_seed):
                    for pattern in patterns:
                        try:
                            # Sample from the shuffled pool
                            words, metadata = self.get_mixed_list(n=count, pattern=pattern, seed=seed, coherent_pool=coherent_pool.copy())
                            
                            # Determine type label
                            if pattern == 'c':
                                type_label = 'coherent'
                            elif pattern == 'r':
                                type_label = 'chaotic'
                            else:
                                type_label = f'mixed_{pattern}'
                            
                            dataset.append({
                                "id": f"{type_label}_{count}_{seed_idx}_{trial_idx}_{random.randint(1000,9999)}",
                                "type": type_label,
                                "pattern": pattern,
                                "count": count,
                                "seed": seed,
                                "seed_idx": seed_idx,
                                "trial_id": trial_idx,
                                "words": words,
                                "metadata": metadata
                            })
                        except ValueError as e:
                            logger.warning(f"Skipping seed '{seed}' trial {trial_idx} for pattern '{pattern}' count {count}: {e}")
                            continue
        
        return dataset

    def save_dataset(self, dataset: List[Dict], filepath: str):
        """Saves the dataset to a JSON file, ensuring the directory exists."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        logger.info(f"Dataset saved to {filepath}")