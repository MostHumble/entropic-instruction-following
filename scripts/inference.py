import hydra
import json
import logging
import os
import gc
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from huggingface_hub import login
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

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


def get_available_models(conf_path: str = "conf/model") -> List[str]:
    """Get list of available model configs"""
    model_dir = Path(conf_path)
    models = [f.stem for f in model_dir.glob("*.yaml")]
    return sorted(models)


def load_model_config(model_name: str, base_cfg: DictConfig) -> DictConfig:
    """Load model-specific config and merge with base config"""
    model_cfg_path = Path(f"conf/model/{model_name}.yaml")
    
    if not model_cfg_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_cfg_path}")
    
    with open(model_cfg_path) as f:
        model_cfg = OmegaConf.load(f)
    
    # Create a mutable copy of base config
    cfg = OmegaConf.to_container(base_cfg, resolve=True)
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, False)
    
    # Merge model config
    cfg = OmegaConf.merge(cfg, model_cfg)
    
    return cfg


def run_inference_for_model(
    model_name: str,
    cfg: DictConfig,
    dataset: List[dict],
    results_dir: Path
) -> str:
    """Run inference for a single model and return results path"""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Running inference for model: {model_name}")
    logger.info(f"{'='*70}")
    
    # Load model-specific config
    model_cfg = load_model_config(model_name, cfg)

    logger.info(model_cfg)
    logger.info(f"Model: {model_cfg.model.name}")
    logger.info(f"Max model len: {model_cfg.inference.max_model_len}")
    logger.info(f"Temperature: {model_cfg.inference.sampling.temperature}")
    
    # Filter dataset by rule counts and patterns
    rule_counts_to_test = (
        model_cfg.experiment.rule_counts
        if model_cfg.experiment.rule_counts is not None
        else model_cfg.word_data_generator.rule_counts
    )
    patterns_to_test = (
        model_cfg.experiment.patterns
        if model_cfg.experiment.patterns is not None
        else model_cfg.word_data_generator.patterns
    )
    
    filtered_dataset = [
        case for case in dataset 
        if case['count'] in rule_counts_to_test
        and case.get('pattern') in patterns_to_test
    ]

    # Load strategy
    strategy = get_strategy(cfg.strategy.name, model_name=model_cfg.model.name)
    logger.info(f"Strategy: {cfg.strategy.name}")
    
    logger.info(f"Testing rule counts: {rule_counts_to_test}")
    logger.info(f"Testing patterns: {patterns_to_test}")
    prompts = [strategy.build_prompt(case['words']) for case in filtered_dataset]
    logger.info(f"Filtered to {len(filtered_dataset)} samples for testing")
    
    # Initialize vLLM
    logger.info("Initializing vLLM...")
    
    # Detect number of available GPUs
    n_gpus = torch.cuda.device_count()
    logger.info(f"Detected {n_gpus} GPU(s)")
    
    # Configure tensor parallelism for multi-GPU setup
    llm = LLM(
        model=model_cfg.model.name,
        max_model_len=model_cfg.inference.max_model_len,
        trust_remote_code=model_cfg.inference.trust_remote_code,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=n_gpus,  # Use all available GPUs
    )
    
    logger.info(f"vLLM initialized with tensor_parallel_size={n_gpus}")
    
    sampling_params = SamplingParams(
        temperature=model_cfg.inference.sampling.temperature,
        top_p=model_cfg.inference.sampling.top_p,
        max_tokens=model_cfg.inference.sampling.max_tokens
    )
    
    # Generate outputs
    logger.info("Generating outputs...")
    outputs = llm.generate(prompts, sampling_params)

    del llm
    gc.collect()
    destroy_model_parallel()
    torch.cuda.empty_cache()
    
    # Evaluate & save results
    results = []
    for i, output in enumerate(outputs):
        case = filtered_dataset[i]
        generated_text = output.outputs[0].text
        stats = Evaluator.score_strict(case['words'], generated_text)
        
        results.append({
            "id": case['id'],
            "type": case['type'],
            "pattern": case['pattern'],
            "count": case['count'],
            "trial": case['trial'],
            "seed": case['seed'],
            "strategy": cfg.strategy.name,
            "model": model_name,
            "model_name_full": model_cfg.model.name,
            "score": stats['score'],
            "passed_count": stats['passed_count'],
            "total_count": stats['total_count'],
            "missing_words": json.dumps(stats['missing_words']),
            "followed_positions": json.dumps(stats['followed_positions']),
            "unfollowed_positions": json.dumps(stats['unfollowed_positions']),
            "word_details": json.dumps(stats['word_details']),
            "generated_text": generated_text
        })
    
    # Save results for this model
    df = pd.DataFrame(results)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = results_dir / f"results_{cfg.strategy.name}_{model_name}.csv"
    df.to_csv(results_path, index=False)
    logger.info(f"Saved {len(results)} results to {results_path}")
    
    return str(results_path)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Run inference on multiple models"""
    
    # Check device availability
    if not torch.cuda.is_available():
        logger.warning("CUDA GPU not available. vLLM may be very slow on CPU.")
    
    # Determine which models to test
    available_models = get_available_models()
    
    if cfg.experiment.models is None:
        models_to_test = available_models
    else:
        models_to_test = cfg.experiment.models
    
    logger.info(f"Available models: {available_models}")
    logger.info(f"Testing models: {models_to_test}")
    
    # Load dataset once
    dataset = load_dataset(cfg.word_data_generator.dataset_path)
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Results directory
    results_dir = Path(cfg.experiment.results_dir)
    
    # Run inference for each model
    results_files = []
    for model_name in models_to_test:
        try:
            results_file = run_inference_for_model(
                model_name,
                cfg,
                dataset,
                results_dir
            )
            results_files.append(results_file)
        except Exception as e:
            logger.error(f"Failed to run inference for model {model_name}: {e}")
            continue
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"EXPERIMENT COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Models tested: {len(results_files)}/{len(models_to_test)}")
    logger.info(f"Results saved to:")
    for rf in results_files:
        logger.info(f"  - {rf}")


if __name__ == "__main__":
    main()