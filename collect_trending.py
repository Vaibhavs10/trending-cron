from huggingface_hub import HfApi
from datasets import Dataset, concatenate_datasets
from datetime import datetime
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
COLLECTION_DATE = datetime.utcnow().isoformat()

def get_trending_models_and_datasets():
    """Fetch top 100 trending models and datasets from Hugging Face Hub."""
    hf_api = HfApi()
    
    logger.info("Fetching top 100 trending models...")
    models = hf_api.list_models(
        sort="trendingScore",
        direction=-1,
        limit=200,
        full=True
    )
    
    logger.info("Fetching top 100 trending datasets...")
    datasets = hf_api.list_datasets(
        sort="trendingScore",
        direction=-1,
        limit=200,
        full=True
    )
    
    return models, datasets

def prepare_model_data(models):
    """Prepare model data for the dataset."""
    model_data = []
    for model in models:
        model_data.append({
            "id": model.modelId,
            "type": "model",
            "author": model.modelId.split('/')[0] if '/' in model.modelId else "",
            "downloads": model.downloads if hasattr(model, 'downloads') else 0,
            "likes": model.likes if hasattr(model, 'likes') else 0,
            "tags": model.tags if hasattr(model, 'tags') else [],
            "last_modified": model.lastModified,
            "created_at": model.createdAt if hasattr(model, 'createdAt') else None,
            "sha": model.sha,
            "collected_at": COLLECTION_DATE
        })
    return pd.DataFrame(model_data)

def prepare_dataset_data(datasets):
    """Prepare dataset data for the dataset."""
    dataset_data = []
    for dataset in datasets:
        dataset_data.append({
            "id": dataset.id,
            "type": "dataset",
            "author": dataset.id.split('/')[0] if '/' in dataset.id else "",
            "downloads": dataset.downloads if hasattr(dataset, 'downloads') else 0,
            "likes": dataset.likes if hasattr(dataset, 'likes') else 0,
            "tags": dataset.tags if hasattr(dataset, 'tags') else [],
            "last_modified": dataset.lastModified,
            "created_at": dataset.createdAt if hasattr(dataset, 'createdAt') else None,
            "sha": dataset.sha,
            "collected_at": COLLECTION_DATE
        })
    return pd.DataFrame(dataset_data)

def update_dataset(models_df, datasets_df, dataset_repo):
    """Update or create the dataset with new data."""
    from datasets import load_dataset, DatasetDict
    
    try:
        # Try to load existing dataset
        existing_ds = load_dataset(dataset_repo)
        
        # Append new data to existing splits
        existing_models = Dataset.from_pandas(existing_ds["models"].to_pandas())
        existing_datasets = Dataset.from_pandas(existing_ds["datasets"].to_pandas())
        
        new_models = concatenate_datasets([existing_models, Dataset.from_pandas(models_df)])
        new_datasets = concatenate_datasets([existing_datasets, Dataset.from_pandas(datasets_df)])
    except Exception as e:
        logger.info(f"Dataset doesn't exist or couldn't be loaded, creating new one: {e}")
        new_models = Dataset.from_pandas(models_df)
        new_datasets = Dataset.from_pandas(datasets_df)
    
    # Create dataset with splits
    dataset = DatasetDict({
        "models": new_models,
        "datasets": new_datasets
    })
    
    # Push to Hub
    dataset.push_to_hub(dataset_repo)
    logger.info(f"Successfully updated dataset at {dataset_repo}")

def main():
    # Configuration
    DATASET_REPO = "reach-vb/trending-repos"  # Change this to your repo

    # Check Hugging Face Hub login status
    hf_api_for_login_check = HfApi()
    try:
        user_info = hf_api_for_login_check.whoami()
        logger.info(f"Successfully logged in to Hugging Face Hub as {user_info['name']}.")
    except Exception:  # Catches HTTPError (e.g., 401) if not logged in, or other network issues.
        logger.error(
            "Failed to verify Hugging Face Hub login status. "
            "Please ensure you are logged in using 'huggingface-cli login'. "
            "The script needs to push data to the Hub."
        )
        print("Exiting due to authentication issue. Please run 'huggingface-cli login'.")
        return # Exit the main function if not logged in.

    # Get data
    models, datasets = get_trending_models_and_datasets()
    
    # Prepare DataFrames
    models_df = prepare_model_data(models)
    datasets_df = prepare_dataset_data(datasets)
    
    # Update dataset
    update_dataset(models_df, datasets_df, DATASET_REPO)

if __name__ == "__main__":
    main()
