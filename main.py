import os
import json
import tempfile
from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

_ALL_STEPS = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # "test_regression_model",
]


def _component_uri(repo: str, subdir: str) -> str:
    repo = repo.strip().rstrip("/")
    if "#components" in repo:
        return f"{repo}/{subdir}"
    return f"{repo}#components/{subdir}"


def _full_artifact_ref(project: str, name_with_alias: str, entity: str = "") -> str:
    base = f"{(entity or os.environ.get('WANDB_ENTITY', '')).strip('/')}/{project.strip('/')}".replace("//", "/")
    return f"{base}/{name_with_alias}".replace("//", "/").strip("/")


@hydra.main(version_base=None, config_path=".", config_name="config")
def go(config: DictConfig) -> None:
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]
    if "wandb_entity" in config["main"]:
        os.environ["WANDB_ENTITY"] = str(config["main"]["wandb_entity"])

    steps = (
        _ALL_STEPS
        if str(config["main"]["steps"]).strip().lower() == "all"
        else [s.strip() for s in str(config["main"]["steps"]).split(",") if s.strip()]
    )

    components_repo = str(config["main"]["components_repository"]).strip()
    project_name = config["main"]["project_name"]
    wandb_entity = os.environ.get("WANDB_ENTITY", "")

    orig_cwd = Path(get_original_cwd()).resolve()
    basic_cleaning_uri = str((orig_cwd / "src" / "basic_cleaning").resolve())
    data_check_uri = str((orig_cwd / "src" / "data_check").resolve())
    train_rf_uri = str((orig_cwd / "src" / "train_random_forest").resolve())

    tmp_dir = tempfile.TemporaryDirectory()
    try:
        os.chdir(tmp_dir.name)

        if "download" in steps:
            print(">> Step: download")
            _ = mlflow.run(
                uri=_component_uri(components_repo, "get_data"),
                entry_point="main",
                env_manager="local",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": config.get("download", {}).get("artifact_name", "sample.csv"),
                    "artifact_type": config.get("download", {}).get("artifact_type", "raw_data"),
                    "artifact_description": config.get("download", {}).get(
                        "artifact_description", "Raw_file_as_downloaded"
                    ),
                },
            )

        if "basic_cleaning" in steps:
            print(">> Step: basic_cleaning")
            _ = mlflow.run(
                uri=basic_cleaning_uri,
                entry_point="main",
                env_manager="local",
                parameters={
                    "input_artifact": f"{config.get('download', {}).get('artifact_name', 'sample.csv')}:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_data",
                    "output_description": "Data_after_basic_cleaning",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_check" in steps:
            print(">> Step: data_check")
            ref = _full_artifact_ref(project_name, "clean_sample.csv:latest", wandb_entity)
            _ = mlflow.run(
                uri=data_check_uri,
                entry_point="main",
                env_manager="local",
                parameters={
                    "csv": ref,
                    "ref": ref,
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_split" in steps:
            print(">> Step: data_split")
            input_ref = _full_artifact_ref(project_name, "clean_sample.csv:latest", wandb_entity)
            _ = mlflow.run(
                uri=_component_uri(components_repo, "train_val_test_split"),
                entry_point="main",
                env_manager="local",
                parameters={
                    "input_artifact": input_ref,
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                },
            )

        if "train_random_forest" in steps:
            print(">> Step: train_random_forest")
            rf_config_path = Path(tmp_dir.name) / "rf_config.json"
            with open(rf_config_path, "w", encoding="utf-8") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)

            train_ref = _full_artifact_ref(project_name, "train.csv:latest", wandb_entity)
            val_ref = _full_artifact_ref(project_name, "val.csv:latest", wandb_entity)

            _ = mlflow.run(
                uri=train_rf_uri,
                entry_point="main",
                env_manager="local",
                parameters={
                    "train_data": train_ref,
                    "val_data": val_ref,
                    "rf_config": str(rf_config_path),
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": "random_forest_export",
                },
            )

    finally:
        os.chdir(orig_cwd)
        tmp_dir.cleanup()


if __name__ == "__main__":
    go()

