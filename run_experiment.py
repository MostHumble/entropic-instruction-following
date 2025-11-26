import hydra
import json
import logging
import os
from dotenv import load_dotenv

import pandas as pd
from omegaconf import DictConfig
from huggingface_hub import login
from vllm import LLM, SamplingParams

from src.evaluator import Evaluator
from src.utils import get_strategy, load_dataset

logger = logging.getLogger(__name__)

# Load environment and authenticate
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.warning("HF_TOKEN not set. Gated models may fail to load.")
else:
    login(token=HF_TOKEN)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Load strategy
    strategy = get_strategy(cfg.strategy.name)
    logger.info(f"Strategy: {strategy.name}")
    logger.info(f"Model: {cfg.model.name}")

    # 2. Load dataset
    dataset = load_dataset(cfg.experiment.dataset_path)
    logger.info(f"Loaded {len(dataset)} samples")

    # 3. Build prompts
    prompts = [strategy.build_prompt(case['words']) for case in dataset]

    # 4. Initialize vLLM
    logger.info(f"Context window: {cfg.inference.max_model_len}")
    logger.info(f"Temperature: {cfg.inference.sampling.temperature}")
    
    llm = LLM(
        model=cfg.model.name,
        max_model_len=cfg.inference.max_model_len,
        trust_remote_code=True  # Add if using custom models
    )

    sampling_params = SamplingParams(
        temperature=cfg.inference.sampling.temperature,
        top_p=cfg.inference.sampling.top_p,
        max_tokens=cfg.inference.sampling.max_tokens
    )

    # 5. Generate outputs
    logger.info("Generating outputs...")
    outputs = llm.generate(prompts, sampling_params)

    # 6. Evaluate & save results
    results = []
    for i, output in enumerate(outputs):
        case = dataset[i]
        stats = Evaluator.score_strict(case['words'], output.outputs[0].text)
        results.append({
            "strategy": strategy.name,
            "count": case['count'],
            "score": stats['score'],
            "model": cfg.model.name,
        })

    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
    logger.info(f"Saved results to results.csv")


if __name__ == "__main__":
    main()