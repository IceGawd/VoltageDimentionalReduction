from matplotlib import pyplot as plt 
from xgboost import XGBClassifier


# def plot_train_test_errors(results):

#     epochs = len(results['validation_0']['merror'])
#     x_axis = range(epochs)

#     plt.figure(figsize=(8, 6))
#     plt.plot(x_axis, results['validation_0']['merror'], label='Train Error')
#     plt.plot(x_axis, results['validation_1']['merror'], label='Test Error')
#     plt.xlabel('Number of Boosting Rounds')
#     plt.ylabel('Classification Error Rate')
#     plt.title('XGBoost Classification Error vs. Boosting Rounds')
#     plt.grid()
#     plt.legend()
#     plt.show()

def plot_train_test_errors(train_errors, test_errors=None):
    x_axis = range(1, len(train_errors) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, train_errors, label='Train Error', marker='o')

    if test_errors is not None:
        plt.plot(x_axis, test_errors, label='Test Error', marker='x')

    plt.xlabel('Batch Number')
    plt.ylabel('Classification Error Rate')
    plt.title('Error Rate Across Batches')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
