import hydra
from omegaconf import DictConfig
from src.generators import WordDataGenerator
import logging

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logger.info("Starting Data Generation...")
    
    gen = WordDataGenerator(
        alphanumeric_only=cfg.word_data_generator.alphanumeric_only,
        seeds=cfg.word_data_generator.seeds
    )
    
    counts = cfg.word_data_generator.rule_counts
    trials = cfg.word_data_generator.trials_per_count
    patterns = cfg.word_data_generator.get('patterns', ['c', 'r'])
    
    logger.info(f"Generating for counts: {counts}")
    logger.info(f"Using patterns: {patterns}")
    dataset = gen.generate_dataset(rule_counts=counts, trials=trials, patterns=patterns)
    
    gen.save_dataset(dataset, cfg.word_data_generator.dataset_path)
    logger.info(f"Generated {len(dataset)} samples")

if __name__ == "__main__":
    main()