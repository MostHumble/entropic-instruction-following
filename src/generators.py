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
        self.min_word_length = 2  # Minimum word length filter

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

    def calculate_max_coherent_words_needed(self, rule_counts: List[int], patterns: List[str]) -> int:
        """
        Calculate the maximum number of coherent words needed for any single trial.
        
        Args:
            rule_counts: List of word counts to test
            patterns: List of patterns to test
            
        Returns:
            Maximum number of coherent words needed
        """
        max_needed = 0
        
        for count in rule_counts:
            for pattern in patterns:
                mode, components = self.parse_pattern(pattern)
                
                # Count how many coherent components we need
                coherent_count = 0
                
                if mode == 'repeating':
                    pattern_length = len(components)
                    for i in range(count):
                        component = components[i % pattern_length]
                        if component.startswith('c'):
                            coherent_count += 1
                else:  # blocked mode
                    num_blocks = len(components)
                    block_size = count // num_blocks
                    remainder = count % num_blocks
                    
                    for idx, component in enumerate(components):
                        if component.startswith('c'):
                            current_block_size = block_size + (1 if idx < remainder else 0)
                            coherent_count += current_block_size
                
                max_needed = max(max_needed, coherent_count)
        
        return max_needed

    def validate_and_get_seed_words(self, seed: str, min_words_needed: int) -> Tuple[Optional[List[str]], Dict]:
        """
        Validate that a seed has enough words and return the word pool.
        
        Args:
            seed: WordNet synset identifier (e.g., 'food.n.02')
            min_words_needed: Minimum number of words required
            
        Returns:
            Tuple of (word_list or None, metadata) where word_list is None if validation fails
        """
        try:
            seed_synset = wn.synset(seed)
        except Exception as e:
            logger.warning(f"Seed synset '{seed}' not found: {e}")
            return None, {
                "seed": seed,
                "status": "error",
                "error": str(e),
                "words_available": 0,
                "words_needed": min_words_needed
            }
        
        # Get all hyponyms (more specific terms) from this seed
        words = {w.replace('_', ' ') for s in seed_synset.closure(lambda s: s.hyponyms()) for w in s.lemma_names()}
        words = [w for w in words if '_' not in w]
        
        # Apply filters
        if self.alphanumeric_only:
            words = [w for w in words if w.isalnum()]
        
        # Filter by minimum length
        words = [w for w in words if len(w) >= self.min_word_length]
        
        words_available = len(words)
        
        metadata = {
            "seed": seed,
            "words_available": words_available,
            "words_needed": min_words_needed,
            "status": "valid" if words_available >= min_words_needed else "insufficient"
        }
        
        if words_available < min_words_needed:
            logger.warning(
                f"Seed '{seed}' has insufficient words. "
                f"Available: {words_available}, Needed: {min_words_needed}"
            )
            return None, metadata
        
        return words, metadata

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
        """Generates semantically related words using seeds in order, with random sampling.
    
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
            
            # Apply filters
            if self.alphanumeric_only:
                words = [w for w in words if w.isalnum()]
            
            # Filter by minimum length
            words = [w for w in words if len(w) >= self.min_word_length]
        
            # Remove duplicates already in pool
            words = [w for w in words if w not in pool]
            
            # Shuffle to add randomness
            random.shuffle(words)
            
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

    def get_mixed_list_from_pool(self, n: int, pattern: str, coherent_pool: List[str], 
                                  seed_name: str = "unknown") -> Tuple[List[str], Dict]:
        """
        Generate a mixed word list based on a pattern using pre-shuffled word pools.
        Samples without replacement from the coherent pool.
        
        Args:
            n: Total number of words
            pattern: Pattern string (e.g., 'cr', 'c|r|c', 'c1c2', 'cccr')
            coherent_pool: Pre-shuffled list of coherent words to sample from
            seed_name: Name of the seed for metadata tracking
            
        Returns:
            Tuple of (word_list, metadata)
            
        Examples:
            get_mixed_list_from_pool(12, 'cr', pool) -> c,r,c,r,c,r,c,r,c,r,c,r (alternating, no repeats in c)
            get_mixed_list_from_pool(12, 'c|r|c', pool) -> [4 coherent], [4 random], [4 coherent] (blocks)
        """
        mode, components = self.parse_pattern(pattern)
        
        # First, determine how many words we need from each component type
        component_counts = {}
        
        if mode == 'repeating':
            # Count how many times each component appears in the full sequence
            pattern_length = len(components)
            for i in range(n):
                component = components[i % pattern_length]
                component_counts[component] = component_counts.get(component, 0) + 1
                
        else:  # blocked mode
            # Split n into equal blocks
            num_blocks = len(components)
            block_size = n // num_blocks
            remainder = n % num_blocks
            
            for idx, component in enumerate(components):
                # Distribute remainder across first blocks
                current_block_size = block_size + (1 if idx < remainder else 0)
                component_counts[component] = component_counts.get(component, 0) + current_block_size
        
        # Now prepare word pools for each component type
        component_pools = {}
        coherent_idx = 0  # Track position in coherent pool for sampling without replacement
        
        for component, count in component_counts.items():
            if component == 'r':
                # Random words - generate fresh each time
                component_pools[component] = self.get_chaotic_list(count)
            elif component.startswith('c'):
                # Coherent words - sample from pre-shuffled pool without replacement
                if coherent_idx + count > len(coherent_pool):
                    raise ValueError(
                        f"Insufficient coherent words in pool. "
                        f"Needed: {coherent_idx + count}, Available: {len(coherent_pool)}"
                    )
                component_pools[component] = coherent_pool[coherent_idx:coherent_idx + count]
                coherent_idx += count
            else:
                raise ValueError(f"Unknown component: {component}")
        
        # Build the final sequence according to the pattern
        result = []
        component_indices = {comp: 0 for comp in component_pools}
        position_details = []
        
        if mode == 'repeating':
            # Cycle through components, taking one word at a time
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
                "seed": seed_name,
                "coherent_words_used": coherent_idx,
                "component_counts": component_counts,
                "sampling_method": "without_replacement"
            }
            
        else:  # blocked mode
            # Add words block by block
            block_metadata = []
            num_blocks = len(components)
            block_size = n // num_blocks
            remainder = n % num_blocks
            
            for idx, component in enumerate(components):
                current_block_size = block_size + (1 if idx < remainder else 0)
                start_pos = len(result)
                
                # Take the next chunk from this component's pool
                comp_idx = component_indices[component]
                block_words = component_pools[component][comp_idx:comp_idx + current_block_size]
                result.extend(block_words)
                component_indices[component] += current_block_size
                
                block_metadata.append({
                    "block_index": idx,
                    "component": component,
                    "start_position": start_pos,
                    "end_position": len(result),
                    "size": current_block_size
                })
            
            final_metadata = {
                "pattern": pattern,
                "mode": "blocked",
                "total_words": n,
                "seed": seed_name,
                "coherent_words_used": coherent_idx,
                "component_counts": component_counts,
                "blocks": block_metadata,
                "sampling_method": "without_replacement"
            }
        
        return result, final_metadata

    def get_mixed_list(self, n: int, pattern: str) -> Tuple[List[str], Dict]:
        """
        Generate a mixed word list based on a pattern.
        
        Args:
            n: Total number of words
            pattern: Pattern string (e.g., 'cr', 'c|r|c', 'c1c2', 'cccr')
            
        Returns:
            Tuple of (word_list, metadata)
            
        Examples:
            get_mixed_list(12, 'cr') -> c,r,c,r,c,r,c,r,c,r,c,r (alternating, different words)
            get_mixed_list(12, 'c|r|c') -> [4 coherent], [4 random], [4 coherent] (blocks)
            get_mixed_list(12, 'cccr') -> c,c,c,r,c,c,c,r,c,c,c,r (3:1 pulse)
        """
        mode, components = self.parse_pattern(pattern)
        
        # First, determine how many words we need from each component type
        component_counts = {}
        
        if mode == 'repeating':
            # Count how many times each component appears in the full sequence
            pattern_length = len(components)
            for i in range(n):
                component = components[i % pattern_length]
                component_counts[component] = component_counts.get(component, 0) + 1
                
        else:  # blocked mode
            # Split n into equal blocks
            num_blocks = len(components)
            block_size = n // num_blocks
            remainder = n % num_blocks
            
            for idx, component in enumerate(components):
                # Distribute remainder across first blocks
                current_block_size = block_size + (1 if idx < remainder else 0)
                component_counts[component] = component_counts.get(component, 0) + current_block_size
        
        # Now fetch all words needed for each component type
        component_pools = {}
        component_metadata = {}
        
        for component, count in component_counts.items():
            words, metadata = self.get_words_for_component(component, count)
            component_pools[component] = words
            component_metadata[component] = metadata
        
        # Build the final sequence according to the pattern
        result = []
        component_indices = {comp: 0 for comp in component_pools}
        position_details = []
        
        if mode == 'repeating':
            # Cycle through components, taking one word at a time
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
                "component_counts": component_counts,
                "position_details": position_details
            }
            
        else:  # blocked mode
            # Add words block by block
            block_metadata = []
            num_blocks = len(components)
            block_size = n // num_blocks
            remainder = n % num_blocks
            
            for idx, component in enumerate(components):
                current_block_size = block_size + (1 if idx < remainder else 0)
                start_pos = len(result)
                
                # Take the next chunk from this component's pool
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
                "component_counts": component_counts,
                "blocks": block_metadata
            }
        
        return result, final_metadata

    def generate_dataset(self, rule_counts: List[int], trials_per_seed: int, 
                        patterns: Optional[List[str]] = None, 
                        seeds: Optional[List[str]] = None) -> List[Dict]:
        """
        Creates a standardized dataset for experimentation with per-seed trials.
        
        Iteration order: seeds -> trials_per_seed -> rule_counts -> patterns
        Each trial shuffles the seed's word pool and samples without replacement.
        
        Args:
            rule_counts: List of word counts to test
            trials_per_seed: Number of trials per seed (each with fresh shuffle)
            patterns: Optional list of mixing patterns to include (e.g., ['c', 'r', 'cr', 'c|r|c'])
            seeds: Optional list of seeds to use (defaults to self.seeds)
        """
        patterns = patterns or ['c', 'r']  # Default: pure coherent and pure random
        seeds = seeds or self.seeds
        dataset = []
        
        # Calculate maximum coherent words needed
        max_coherent_needed = self.calculate_max_coherent_words_needed(rule_counts, patterns)
        logger.info(f"Maximum coherent words needed per trial: {max_coherent_needed}")
        
        # Iterate through each seed
        for seed in seeds:
            # Validate seed and get word pool
            word_pool, validation_metadata = self.validate_and_get_seed_words(seed, max_coherent_needed)
            
            if word_pool is None:
                # Skip this seed - insufficient words
                logger.warning(f"Skipping seed '{seed}': {validation_metadata['status']}")
                continue
            
            logger.info(
                f"Processing seed '{seed}': {len(word_pool)} words available, "
                f"{max_coherent_needed} needed"
            )
            
            # Iterate through trials for this seed
            for trial_idx in range(trials_per_seed):
                # Shuffle the word pool for this trial (sampling without replacement)
                shuffled_pool = word_pool.copy()
                random.shuffle(shuffled_pool)
                
                # Iterate through rule counts
                for count in rule_counts:
                    # Iterate through patterns
                    for pattern in patterns:
                        try:
                            # Generate words using the shuffled pool
                            words, metadata = self.get_mixed_list_from_pool(
                                count, pattern, shuffled_pool, seed_name=seed
                            )
                            
                            # Determine type label
                            if pattern == 'c':
                                type_label = 'coherent'
                            elif pattern == 'r':
                                type_label = 'chaotic'
                            else:
                                type_label = f'mixed_{pattern}'
                            
                            dataset.append({
                                "id": f"{seed}_{type_label}_{count}_{trial_idx}_{random.randint(1000,9999)}",
                                "type": type_label,
                                "pattern": pattern,
                                "count": count,
                                "seed": seed,
                                "trial": trial_idx,
                                "words": words,
                                "metadata": metadata,
                                "validation_metadata": validation_metadata
                            })
                        except ValueError as e:
                            logger.error(
                                f"Failed to generate pattern '{pattern}' with {count} words "
                                f"from seed '{seed}', trial {trial_idx}: {e}"
                            )
                            continue
        
        logger.info(f"Generated {len(dataset)} samples from {len([s for s in seeds if self.validate_and_get_seed_words(s, max_coherent_needed)[0] is not None])} valid seeds")
        return dataset

    def save_dataset(self, dataset: List[Dict], filepath: str):
        """Saves the dataset to a JSON file, ensuring the directory exists."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        logger.info(f"Dataset saved to {filepath}")