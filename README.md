# Entropic Instruction Following

An experimental framework for testing how semantic coherence affects Large Language Models' (LLMs) ability to follow instructions under cognitive load.

## Overview

This project investigates whether the semantic relatedness of instructions impacts a model's ability to follow them. Inspired by Alan Roth's work on vocabulary constraint tasks, we extend the analysis by introducing **entropy** as a controlled variable.

### Key Questions

- Does semantic coherence (e.g., "School," "Teacher," "Blackboard") make instruction-following easier than random words (e.g., "Azimuth," "Potato," "Carburetor")?
- How do models handle mixed patterns of coherent and random instructions?
- Do positional biases (primacy/recency effects) interact with semantic coherence?

### Experimental Design

The framework tests LLMs on vocabulary constraint tasks where they must write stories including specific words. We vary:

- **Rule Counts**: 50, 200, or 400 words to include
- **Semantic Patterns**: 
  - `c` - Pure coherent (semantically related words from WordNet)
  - `r` - Pure random (uniformly sampled from dictionary)
  - `cr` - Alternating coherent/random
  - `c|r`, `r|c` - Half-and-half splits
  - `c|r|c`, `r|c|r` - Bookended patterns
- **Semantic Seeds**: 10 WordNet synsets covering diverse domains (food, animals, abstract concepts, etc.)
- **Trials**: 10 variations per configuration to ensure statistical robustness

## Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/entropic-instruction-following.git
   cd entropic-instruction-following
   ```

3. **Sync dependencies**:
   ```bash
   uv sync
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your Hugging Face token (required for gated models):
   ```
   HF_TOKEN=your_huggingface_token_here
   ```

5. **Download NLTK data** (required for WordNet):
   ```bash
   uv run python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
   ```

## Usage

### 1. Configure Experiment

Edit `conf/config.yaml` to specify:

- **Rule counts** to test:
  ```yaml
  word_data_generator:
    rule_counts: [50, 200, 400]
  ```

- **Semantic seeds** (WordNet synsets):
  ```yaml
  word_data_generator:
    seeds: [
      "food.n.02",
      "animal.n.01",
      # ... add more seeds
    ]
  ```

- **Patterns** to test:
  ```yaml
  word_data_generator:
    patterns: ["c", "r", "cr", "c|r", "r|c", "c|r|c", "r|c|r"]
  ```

- **Number of trials** per configuration:
  ```yaml
  word_data_generator:
    trials_per_seed: 10
  ```

### 2. Generate Data

Create the experimental dataset:

```bash
uv run python scripts/generate_data.py
```

This will generate `data/raw/dataset_v1.json` containing all word lists for your experiment.

### 3. Configure Models

Edit `conf/config.yaml` to specify which models to test:

```yaml
experiment:
  models: ["mistral", "llama", "falcon"]  # or null for all models
  rule_counts: [50, 200, 400]  # null = all, or specify subset
  patterns: ["c", "r", "cr"]  # null = all, or specify subset
```

For each model, create a config file under `conf/model/` with the Hugging Face model name:

**Example: `conf/model/mistral.yaml`**
```yaml
# @package _global_

model:
  name: mistralai/Mistral-7B-Instruct-v0.3
```

**Example: `conf/model/llama.yaml`**
```yaml
# @package _global_

model:
  name: meta-llama/Llama-3.2-3B-Instruct
```

You can also override inference parameters in model configs:

```yaml
# @package _global_

model:
  name: your-model/name

inference:
  max_model_len: 4096
  sampling:
    temperature: 0.7
    top_p: 0.95
```

### 4. Run Inference

Execute the instruction-following experiments:

```bash
uv run python scripts/inference.py
```

Results will be saved to `data/results/` with CSV files for each model and configuration.

### 5. Analyze Results

Generate plots and summary statistics:

```bash
uv run python scripts/analyze_results.py
```

This creates:
- Comparison plots in `data/results/comparison_<rule_count>_rules/`
- Model-specific analyses in `data/results/Story_<model>/`
- Summary statistics in `data/results/model_comparison_summary.txt`

## Project Structure

```
entropic-instruction-following/
├── conf/                      # Hydra configuration files
│   ├── config.yaml           # Main experiment config
│   ├── model/                # Model-specific configs
│   └── strategy/             # Prompting strategies
├── data/
│   ├── raw/                  # Generated word lists
│   └── results/              # Experimental results & plots
├── scripts/
│   ├── generate_data.py      # Dataset generation
│   ├── inference.py          # Run experiments
│   └── analyze_results.py    # Generate visualizations
├── src/                      # Core implementation
│   ├── generators.py         # WordNet-based data generation
│   ├── evaluator.py          # Rule compliance checking
│   ├── strategies.py         # Prompt templates
│   └── analysis/             # Statistical analysis & plotting
└── pyproject.toml            # Project dependencies
```

## Key Findings

Initial experiments reveal:

1. **Coherence helps under load**: At 200-400 rules, coherent instructions significantly improve compliance
2. **Model-specific behaviors**: Different architectures show distinct patterns (e.g., Mistral benefits more from coherence than Olmo)
3. **Position still matters**: Primacy bias persists even with coherent instructions
4. **Pattern interactions**: Bookended patterns (`c|r|c`) can help models recover from chaotic middle sections

See the [blog post](link-to-your-blog) for detailed analysis.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{entropic_instruction_following,
  author = {Your Name},
  title = {Entropic Instruction Following: Testing Semantic Coherence in LLM Instruction Following},
  year = {2025},
  url = {https://github.com/yourusername/entropic-instruction-following}
}
```

## License

See [LICENSE](LICENSE) for details.

## Acknowledgments

This work was inspired by [Alan Roth's experiments](https://alantech.io/blog/rule-following-an-llm-benchmark) on LLM instruction following and context window utilization.
