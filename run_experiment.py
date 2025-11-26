import hydra
import json
import logging
import os
from dotenv import load_dotenv

import torch
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
    # Check device availability
    if not torch.cuda.is_available():
        logger.warning("CUDA GPU not available. vLLM may not work or will be very slow on CPU.")
    
    # 1. Load strategy
    strategy = get_strategy(cfg.strategy.name)
    logger.info(f"Strategy: {cfg.strategy.name}")
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"Max model len: {cfg.inference.max_model_len}")
    logger.info(f"Temperature: {cfg.inference.sampling.temperature}")

    # 2. Load dataset
    dataset = load_dataset(cfg.word_data_generator.dataset_path)
    logger.info(f"Loaded {len(dataset)} samples")

    # 3. Filter dataset and build prompts
    filtered_dataset = [case for case in dataset if case['count'] in cfg.experiment.rule_counts]
    prompts = [strategy.build_prompt(case['words']) for case in filtered_dataset]
    logger.info(f"Filtered to {len(filtered_dataset)} samples for testing")

    # 4. Initialize vLLM
    logger.info(f"Context window: {cfg.inference.max_model_len}")
    logger.info(f"Temperature: {cfg.inference.sampling.temperature}")
    
    llm = LLM(
        model=cfg.model.name,
        max_model_len=cfg.inference.max_model_len,
        trust_remote_code=cfg.inference.trust_remote_code,
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
        case = filtered_dataset[i]
        generated_text = output.outputs[0].text
        stats = Evaluator.score_strict(case['words'], generated_text)
        
        results.append({
            "id": case['id'],
            "type": case['type'],
            "count": case['count'],
            "strategy": strategy.name,
            "model": cfg.model.name,
            "score": stats['score'],
            "passed_count": stats['passed_count'],
            "total_count": stats['total_count'],
            "missing_words": json.dumps(stats['missing_words']),
            "followed_positions": json.dumps(stats['followed_positions']),
            "unfollowed_positions": json.dumps(stats['unfollowed_positions']),
            "word_details": json.dumps(stats['word_details']),
            "generated_text": generated_text
        })

    # Save results
    df = pd.DataFrame(results)
    results_dir = cfg.experiment.results_dir
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"results_{strategy.name}_{cfg.model.name.replace('/', '_')}.csv")
    df.to_csv(results_path, index=False)
    logger.info(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()