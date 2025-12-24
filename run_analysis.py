# run_analysis.py
from src.preprocessing import load_and_preprocess
from src.train import models
from src.evaluate import evaluate_models
from src.visualize import plot_confusion_matrix
def main():
    # Step 1: Load and preprocess the data
    data_path = "data/ckd.csv"  
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)

    # Step 2: Train models
    models = train_models(X_train, y_train)

    # Step 3: Evaluate models
    results = evaluate_models(models, X_test, y_test)

    # Step 4: Print results and visualize confusion matrices
    for name, metrics in results.items():
        print(f"{name} Accuracy: {metrics['accuracy']:.2f}")
        print(f"{name} Classification Report:\n{metrics['report']}")
        plot_confusion_matrix(metrics['confusion_matrix'], f"{name} Confusion Matrix")
