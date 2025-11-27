import hydra
import sys
from pathlib import Path
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.visualizations import ResultsVisualizer
from src.analysis.statistics import generate_summary_statistics

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
    else:
        # Find the most recent results file
        results_files = list(results_dir.glob("results_*.csv"))
        if not results_files:
            raise FileNotFoundError(f"No results CSV files found in {results_dir}")
        results_csv = max(results_files, key=lambda p: p.stat().st_mtime)
    
    print(f" Analyzing results from: {results_csv}")
    
    # Generate visualizations
    visualizer = ResultsVisualizer(str(results_csv), str(results_dir))
    visualizer.create_all()
    
    # Generate statistics
    summary_path = results_dir / "analysis_summary.txt"
    summary_text = generate_summary_statistics(str(results_csv), str(summary_path))
    print(summary_text)

if __name__ == "__main__":
    main()