import hydra
import sys
from pathlib import Path
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generators import WordDataGenerator
import logging

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    logger.info("Starting Data Generation...")
    
    gen = WordDataGenerator(
        alphanumeric_only=cfg.word_data_generator.alphanumeric_only,
        seeds=cfg.word_data_generator.seeds
    )
    
    counts = cfg.word_data_generator.rule_counts
    trials_per_seed = cfg.word_data_generator.trials_per_seed
    patterns = cfg.word_data_generator.get('patterns', ['c', 'r'])
    seeds = cfg.word_data_generator.seeds
    
    logger.info(f"Generating for counts: {counts}")
    logger.info(f"Using patterns: {patterns}")
    logger.info(f"Using {len(seeds)} seeds with {trials_per_seed} trials per seed")
    
    dataset = gen.generate_dataset(
        rule_counts=counts, 
        trials_per_seed=trials_per_seed, 
        patterns=patterns,
        seeds=seeds
    )
    
    gen.save_dataset(dataset, cfg.word_data_generator.dataset_path)
    logger.info(f"Generated {len(dataset)} samples")

if __name__ == "__main__":
    main()