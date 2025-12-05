import hydra
import sys
from pathlib import Path
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.visualizations import ResultsVisualizer
from src.analysis.statistics import generate_summary_statistics
from src.analysis.comparison import MultiModelComparison
from src.analysis.plot_token_visual_test import TokenPerformanceVisualizer, get_available_models, load_model_configs

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Analyze experiment results using Hydra config"""
    
    results_dir = Path(cfg.experiment.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # First: Generate multi-model comparison
    print(f"📊 Analyzing all results in: {results_dir}")
    
    try:
        comparison = MultiModelComparison(str(results_dir))
        comparison.plot_model_comparison_comprehensive()
        summary = comparison.get_summary()
        print(summary)
        
        # Save comparison summary
        with open(results_dir / "model_comparison_summary.txt", 'w') as f:
            f.write(summary)
    
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("No results found. Run run_experiment.py first.")
        return
    
    for csv_file in results_dir.glob("results_*.csv"):
        # Extract model name from filename
        parts = csv_file.stem.split('_')
        if len(parts) >= 3:
            model_name = '_'.join(parts[2:])
        else:
            model_name = 'unknown'
        
        print(f"\n🔍 Analyzing {model_name}...")
        
        # Create model-specific output directory
        model_output_dir = results_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations
        visualizer = ResultsVisualizer(str(csv_file), str(model_output_dir))
        visualizer.create_all()
        
        # Generate statistics
        summary_path = model_output_dir / "analysis_summary.txt"
        generate_summary_statistics(str(csv_file), str(summary_path))
    
    # Second: Generate token-count-performance scatter plots
    print("\n📈 Generating token performance scatter plots...")
    # Determine which models to analyze
    available_models = get_available_models()
    
    if cfg.experiment.models is None:
        models_to_analyze = available_models
    else:
        models_to_analyze = cfg.experiment.models
            
    # Load model configurations
    model_configs = load_model_configs(models_to_analyze)

    # Initialize visualizer
    dataset_path = cfg.word_data_generator.dataset_path
    output_dir = Path(results_dir) / "token_scatter_plots"
    
    visualizer = TokenPerformanceVisualizer(results_dir, dataset_path, model_configs)
    
    # Generate all plots
    visualizer.generate_all_plots(output_dir)

if __name__ == "__main__":
    main()