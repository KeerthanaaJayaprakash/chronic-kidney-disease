import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, title):
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
	plt.title(title)
	plt.xlabel('Predicted')
	plt.ylabel('Actual')
	plt.show()