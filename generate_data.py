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
    
    # Access config values using dot notation
    counts = cfg.experiment.rule_counts
    trials = cfg.experiment.trials_per_count
    
    logger.info(f"Generating for counts: {counts}")
    dataset = gen.generate_dataset(rule_counts=counts, trials=trials)
    
    # Hydra automatically handles path resolution relative to where you run the script
    # But for data, we usually want a fixed absolute path or relative to project root
    # For simplicity here, we use the string from config
    gen.save_dataset(dataset, cfg.experiment.dataset_path)

if __name__ == "__main__":
    main()