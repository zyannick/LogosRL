# Logos RL


In this project, we will try to implement from scratch reinforcement learning algorithms for mathematical reasoning and de novo proteins generations.

## Usage

This project uses Docker and DVC to ensure a completely reproducible environment.

### Prerequisites

- [Docker](https://www.docker.com/get-started)
- [DVC](https://dvc.org/doc/install)
- [GIT](https://git-scm.com/)
- NVIDIA GPU with drivers compatible with CUDA 12.1 or higher.

### Setup & Installation

1. Clone the Repository

    ```bash
    git clone https://github.com/zyannick/moe-rl-finetune.git
    cd moe-rl-finetune
    ```


2. Create a Virtual Environment and Install Dependencies

    ```bash
    # Create a python env usin uv/anaconda (python 3.12)
    # conda create -n moe python=3.12
    # conda activate moe
    # pip install -r requirements.txt
    uv venv -p 3.12
    source .venv/bin/activate
    uv pip install -r requirements.txt
    ```

3.  **Create Local Directories:**
    Before the first run, create the necessary directories for outputs and caches. This ensures the Docker container has the correct permissions.

    ```bash
    mkdir -p data moe_outputs mlruns hf_cache mpl_config
    ```

4.  **Run the Full Pipeline:**
    This single command executes the entire DVC pipeline, from data preparation to training and evaluation.

    ```bash
    # Launch the full pipeline inside a docker
    dvc repro
    ```
    or
    ```bash
    # Launch the full pipeline inside a docker
    python src/run_pipeline.py --pipeline_stage full_pipeline
    ```

    The final model, checkpoints, and metrics will be available in the `moe_outputs/` directory, and experiment results can be viewed via the MLflow UI.

5.  **Launch MLflow UI (Optional):**
    To view the experiment results, run the MLflow UI server:
    ```bash
    mlflow server --backend-store-uri sqlite:///moe_outputs/mlflow.db --port 5000
    ```
    Then navigate to `http://localhost:5000` in your browser.
