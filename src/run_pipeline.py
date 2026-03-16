from pathlib import Path
import yaml

from data_generation import generate_dataset
from distributed_preprocessing import to_dask_dataframe, split_features_target
from distributed_training import train_distributed_model


def load_config(config_path="experiments/pipeline_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline():
    config = load_config()

    df = generate_dataset(
        n_samples=config["n_samples"],
        n_features=config["n_features"],
        n_informative=config["n_informative"],
        random_state=config["random_state"],
    )

    ddf = to_dask_dataframe(df, n_partitions=config["n_partitions"])
    X, y = split_features_target(ddf)

    _, acc = train_distributed_model(X, y)

    print("Distributed pipeline complete")
    print(f"Accuracy: {acc:.4f}")

    Path("results").mkdir(exist_ok=True)
    with open("results/pipeline_summary.txt", "w", encoding="utf-8") as f:
        f.write("Distributed pipeline complete\n")
        f.write(f"Accuracy: {acc:.4f}\n")


if __name__ == "__main__":
    run_pipeline()
