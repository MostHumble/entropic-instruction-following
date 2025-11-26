import hydra
import json
import pandas as pd
from omegaconf import DictConfig
from vllm import LLM, SamplingParams
from src.strategies import StandardStoryStrategy, IndexedListStrategy
from src.evaluator import Evaluator
import logging

logger = logging.getLogger(__name__)

def get_strategy(strategy_name: str):
    """Factory to pick strategy based on config string."""
    if strategy_name == "Standard_Story":
        return StandardStoryStrategy()
    elif strategy_name == "Indexed_List":
        return IndexedListStrategy()
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Load Strategy based on the YAML config currently selected
    strategy = get_strategy(cfg.strategy.name)
    logger.info(f"--- Running Experiment: {strategy.name} ---")
    logger.info(f"Model: {cfg.model.name}")

    # 2. Load Data
    try:
        with open(cfg.experiment.dataset_path, 'r') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        logger.error("Dataset not found. Run generate_data.py first!")
        return

    # 3. Build Prompts
    prompts = [strategy.build_prompt(case['words']) for case in dataset]

    # 4. Initialize vLLM (using params from config)
    llm = LLM(
        model=cfg.model.name, 
        max_model_len=cfg.inference.max_model_len
    )
    
    # Convert Hydra config to vLLM SamplingParams
    samp_cfg = cfg.inference.sampling
    sampling_params = SamplingParams(
        temperature=samp_cfg.temperature,
        top_p=samp_cfg.top_p,
        max_tokens=samp_cfg.max_tokens
    )

    outputs = llm.generate(prompts, sampling_params)

    # 5. Evaluate & Save (Hydra creates a specialized output folder for this run)
    results = []
    for i, output in enumerate(outputs):
        case = dataset[i]
        stats = Evaluator.score_strict(case['words'], output.outputs[0].text)
        results.append({
            "strategy": strategy.name,
            "count": case['count'],
            "score": stats['score'],
            "model": cfg.model.name
        })

    df = pd.DataFrame(results)
    # Save directly to the hydra output directory
    df.to_csv("results.csv", index=False)
    logger.info("Saved results.csv to Hydra output directory.")

if __name__ == "__main__":
    main()