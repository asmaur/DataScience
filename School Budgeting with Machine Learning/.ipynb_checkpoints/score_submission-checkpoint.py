import numpy as np
import pandas as pd
def score_submission(pred_path, holdout_path):
    # this happens on the backend to get the score
    holdout_labels = pd.get_dummies(
    pd.read_csv(holdout_path, index_col=0)
    .apply(lambda x: x.astype('category'), axis=0)
    )

    preds = pd.read_csv(pred_path, index_col=0)

    # make sure that format is correct
    assert (preds.columns == holdout_labels.columns).all()
    assert (preds.index == holdout_labels.index).all()

    return _multi_multi_log_loss(preds.values, holdout_labels.values)