from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_models(models, X_test, y_test):
	results = {}
	for name, model in models.items():
		preds = model.predict(X_test)
		results[name] = {
			'accuracy': accuracy_score(y_test, preds),
			'report': classification_report(y_test, preds),
			'confusion_matrix': confusion_matrix(y_test, preds)
		}
	return results