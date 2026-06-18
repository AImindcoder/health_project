from hackthathon import interpret_labs, build_feature_vector
import pandas as pd


def interpret(labs: dict) -> dict:
    return interpret_labs(labs)


def build_features(labs: dict) -> pd.DataFrame:
    df = build_feature_vector(labs)
    return df.fillna(0).astype(float)
