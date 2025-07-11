from matplotlib import pyplot as plt 
from xgboost import XGBClassifier


def plot_train_test_errors(results):

    epochs = len(results['validation_0']['merror'])
    x_axis = range(epochs)

    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, results['validation_0']['merror'], label='Train Error')
    plt.plot(x_axis, results['validation_1']['merror'], label='Test Error')
    plt.xlabel('Number of Boosting Rounds')
    plt.ylabel('Classification Error Rate')
    plt.title('XGBoost Classification Error vs. Boosting Rounds')
    plt.grid()
    plt.legend()
    plt.show()
