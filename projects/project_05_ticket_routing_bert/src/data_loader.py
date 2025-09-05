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




    # Step 1: Move all classes with only 1 sample to train set
    class_counts = df[topic_col].value_counts()
    single_sample_classes = class_counts[class_counts == 1].index.tolist()
    multi_sample_df = df[~df[topic_col].isin(single_sample_classes)]
    single_sample_df = df[df[topic_col].isin(single_sample_classes)]

    # Step 2: Recursively move classes with <2 samples in the split set to train
    while True:
        split_class_counts = multi_sample_df[topic_col].value_counts()
        too_few_classes = split_class_counts[split_class_counts < 2].index.tolist()
        if not too_few_classes:
            break
        # Move these to train
        to_move = multi_sample_df[multi_sample_df[topic_col].isin(too_few_classes)]
        single_sample_df = pd.concat([single_sample_df, to_move], ignore_index=True)
        multi_sample_df = multi_sample_df[~multi_sample_df[topic_col].isin(too_few_classes)]

    if len(multi_sample_df) > 0:
        n_classes = multi_sample_df[topic_col].nunique()
        n_samples = len(multi_sample_df)

        # If test_size is an int, check if it's >= n_classes
        split_test_size = test_size
        if isinstance(test_size, int):
            if split_test_size < n_classes:
                print(f"[WARN] test_size={test_size} < n_classes={n_classes}. Setting test_size=n_classes.")
                split_test_size = n_classes
            if split_test_size >= n_samples:
                raise ValueError(f"test_size={test_size} is greater than total samples={n_samples}.")
        elif isinstance(test_size, float):
            if int(split_test_size * n_samples) < n_classes:
                min_size = n_classes / n_samples + 0.01
                print(f"[WARN] test_size={test_size} too small for n_classes={n_classes}. Setting test_size={min_size:.2f}.")
                split_test_size = min_size

        train_df, val_df = train_test_split(
            multi_sample_df,
            test_size=split_test_size,
            random_state=random_state,
            stratify=multi_sample_df[topic_col],
        )
    else:
        # If all classes are single-sample, put all in train, val is empty
        train_df, val_df = single_sample_df.copy(), df.iloc[0:0].copy()

    # Always add single-sample and too-few-sample classes to the training set
    if not single_sample_df.empty:
        train_df = pd.concat([train_df, single_sample_df], ignore_index=True)

    return train_df, val_df