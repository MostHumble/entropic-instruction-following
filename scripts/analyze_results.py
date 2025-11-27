import hydra
import sys
from pathlib import Path
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.visualizations import ResultsVisualizer
from src.analysis.statistics import generate_summary_statistics
from src.analysis.comparison import MultiModelComparison

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Analyze experiment results using Hydra config"""
    
    results_dir = Path(cfg.experiment.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if specific file is specified in config
    if cfg.experiment.get("analysis_file"):
        results_csv = results_dir / cfg.experiment.analysis_file
        if not results_csv.exists():
            raise FileNotFoundError(f"Results file not found: {results_csv}")
        
        print(f"📊 Analyzing single results file: {results_csv}")
        visualizer = ResultsVisualizer(str(results_csv), str(results_dir))
        visualizer.create_all()
        
        summary_path = results_dir / "analysis_summary.txt"
        summary_text = generate_summary_statistics(str(results_csv), str(summary_path))
        print(summary_text)
    
    else:
        # Analyze all results (multiple models)
        print(f"📊 Analyzing all results in: {results_dir}")
        
        try:
            comparison = MultiModelComparison(str(results_dir))
            comparison.plot_model_comparison()
            summary = comparison.get_summary()
            print(summary)
            
            # Save comparison summary
            with open(results_dir / "model_comparison_summary.txt", 'w') as f:
                f.write(summary)
            
            # Also analyze each model individually
            for model in comparison.all_results['model'].unique():
                model_results = comparison.all_results[comparison.all_results['model'] == model]
                model_csv = results_dir / f"results_*_{model}.csv"
                
                # Find the actual CSV file
                import glob
                matching_files = glob.glob(str(model_csv))
                if matching_files:
                    print(f"\n🔍 Analyzing {model}...")
                    visualizer = ResultsVisualizer(matching_files[0], str(results_dir / model))
                    visualizer.create_all()
        
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
            print("No results found. Run run_experiment.py first.")


if __name__ == "__main__":
    main()