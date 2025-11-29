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
    
    logger.info(f"Generating for counts: {counts}")
    logger.info(f"Trials per seed: {trials_per_seed}")
    logger.info(f"Using patterns: {patterns}")
    dataset = gen.generate_dataset(rule_counts=counts, trials_per_seed=trials_per_seed, patterns=patterns)
    
    gen.save_dataset(dataset, cfg.word_data_generator.dataset_path)
    logger.info(f"Generated {len(dataset)} samples")

if __name__ == "__main__":
    main()