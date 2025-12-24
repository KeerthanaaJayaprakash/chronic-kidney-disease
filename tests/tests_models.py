from src.preprocessing import load_and_preprocess
from src.train import models


def test_training_pipeline():
	X_train, X_test, y_train, y_test = load_and_preprocess('data/ckd.csv')
	models = train_models(X_train, y_train)
	assert len(models) == 3