import pandas as pd
from sklearn.model_selection import train_test_split

def load_ticket_data(
        csv_path: str,
        text_cols: tuple[str, str] = ("Subject", "Description"),
        topic_col: str = "Topic",
        subtopic_col: str = "Sub Topic",
        *,
        drop_na: bool = True,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a CSV file containing support ticket data, preprocess it, and split it into training and validation sets.
    Returns two labels: 'Topic' and 'Sub Topic' plus the combined text from 'Subject' and 'Description'.
    Args:
        csv_path (str): Path to the CSV file.
        text_cols (tuple[str, str]): Tuple containing the names of the text columns to combine.
        topic_col (str): Name of the column containing the main topic labels.
        subtopic_col (str): Name of the column containing the subtopic labels.
        drop_na (bool): Whether to drop rows with missing values in the specified columns.
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Training and validation DataFrames.
    """
    df = pd.read_csv(csv_path)

    # sanity check
    required_cols = list(text_cols) + [topic_col, subtopic_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in CSV: {missing_cols}")