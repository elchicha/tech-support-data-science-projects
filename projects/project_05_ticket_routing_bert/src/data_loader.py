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
    

    df["combined_text"] = (
        df[text_cols[0]].astype(str).str.strip()
        + " . "
        + df[text_cols[1]].astype(str).str.strip()
    )

    if drop_na:
        df = df.dropna(subset=["combined_text", topic_col, subtopic_col])

    if df[topic_col].value_counts().min() < 2:
        raise ValueError(f"Not enough samples in some classes of '{topic_col}' to perform stratified split.")

    n_classes = df[topic_col].nunique()
    n_samples = len(df)

    # If test_size is an int, check if it's >= n_classes
    if isinstance(test_size, int):
        if test_size < n_classes:
            print(f"[WARN] test_size={test_size} < n_classes={n_classes}. Setting test_size=n_classes.")
            test_size = n_classes
        if test_size >= n_samples:
            raise ValueError(f"test_size={test_size} is greater than total samples={n_samples}.")
    elif isinstance(test_size, float):
        # If test_size as a float would result in fewer samples than classes, adjust
        if int(test_size * n_samples) < n_classes:
            min_size = n_classes / n_samples + 0.01  # add small epsilon
            print(f"[WARN] test_size={test_size} too small for n_classes={n_classes}. Setting test_size={min_size:.2f}.")
            test_size = min_size

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[topic_col],
    )

    return train_df, val_df