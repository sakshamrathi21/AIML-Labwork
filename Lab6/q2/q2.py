import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn import tree

# DONOT IMPORT ANY OTHER LIBRARIES

class SKLearnDecisionTree:

    def __init__(self, X_train, X_test, Y_train, Y_test):
        
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test


    def train_model(self, criterion="entropy"):

        # train the model without any regularisation
        # and check both train and test accuracies and f1 scores

        # The decision tree classifier can be found here: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

        ## Calculate the accuracies for both test and train datasets. To compute accuracy, you can use the metrics class of sklearn

        ### Student Code start
        self.model = DecisionTreeClassifier(criterion=criterion, random_state=42)
        self.model.fit(self.X_train, self.Y_train)
        train_predictions = self.model.predict(self.X_train)
        test_predictions = self.model.predict(self.X_test)
        train_accuracy = metrics.accuracy_score(self.Y_train, train_predictions)
        test_accuracy = metrics.accuracy_score(self.Y_test, test_predictions)
        train_f1 = metrics.f1_score(self.Y_train, train_predictions)
        test_f1 = metrics.f1_score(self.Y_test, test_predictions)

        results = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_f1": train_f1,
            "test_f1": test_f1
        }

        return results
        ### Student Code end


    def train_model_regularised(self, criterion="entropy"):

        # train the model using regularisation
        # and check both train and test accuracies

        ## Hint: Here are the different parameters that can be tweaked to bring in regularisation
        # 1. max_depth, 2. min_samples_split, 3. min_samples_leaf

        # Vary the max_depth parameter from 1 to 10, min_samples_split from 3 to 30 (in steps of 3 -> 3,6,9...30) and min_samples_leaf from 1 to 10 (in steps of 1 -> 1,2,3...10)
        # Now each combination of these three parameters will give you a different decision tree model.
        # There are total 1000 combinations. Hence you will get 1000 different decision tree models.
        # You need to return a list of training accuracies and test accuracies for all these 1000 models.
        # First you have to vary the min_samples_leaf, then min_samples_split and then max_depth. That means for inde 0-9, you will have max_depth=1 and min_samples_split=3, min_samples_leaf=1,2,3...10
        # Similarly for index 10-19, you will have max_depth=1 and min_samples_split=6 and min_samples_leaf=1,2,3...10. Focus on how we chose the next min_saamples_split
        # Similarly for index 100-109, you will have max_depth=2 and min_samples_split=3 and min_samples_leaf=1,2,3...10. Thus resulting in 1000 models, indexed from 0 to 999.
        # Return a dictionary of the form: {"train_accuracies": [list of training accuracies for 1000 models], "test_accuracies": [list of test accuracies for 1000 models]}
        
        train_accuracies = []
        test_accuracies = []

        ### Student Code start
        for depth in range(1, 11):
            for sample_split in range(3,31,3):
                for sample_leaf in range(1,11):
                    model = DecisionTreeClassifier(criterion=criterion, max_depth=depth, min_samples_split=sample_split, min_samples_leaf=sample_leaf, random_state=42)
                    model.fit(self.X_train, self.Y_train)
                    train_predictions = model.predict(self.X_train)
                    test_predictions = model.predict(self.X_test)
                    train_accuracy = metrics.accuracy_score(self.Y_train, train_predictions)
                    test_accuracy = metrics.accuracy_score(self.Y_test, test_predictions)
                    train_accuracies.append(train_accuracy)
                    test_accuracies.append(test_accuracy)
        # print(train_accuracies.__len__())
        results = {"train_accuracies": train_accuracies, "test_accuracies": test_accuracies}

        ### Student Code end
        
        return results

    def plot_tree(self, filename="decision_tree.png"):

        # save the decision trees

        fig = plt.figure(figsize=(25,20))
        _ = tree.plot_tree(self.tree, filled=True, feature_names=list(self.X_train.columns)) 
        fig.savefig(filename)
        plt.close()


def main():

    # files on which model will be trained, these files will be provided to you

    X_train = pd.read_csv("./data/X_train.csv")
    Y_train = pd.read_csv("./data/Y_train.csv")

    X_test = pd.read_csv("./data/X_test.csv")
    Y_test = pd.read_csv("./data/Y_test.csv")

    dt = SKLearnDecisionTree(X_train, X_test, Y_train, Y_test)

    # train the model
    results = dt.train_model()
    print(results)

    # regularise and train the model
    results_regularised = dt.train_model_regularised()

    # Construct a plot to show the effect of regularisation
    plt.plot(results_regularised["train_accuracies"], label="train")
    plt.plot(results_regularised["test_accuracies"], label="test")
    plt.legend()
    plt.xlabel("Model Index")
    plt.ylabel("Accuracy")
    plt.savefig("analysis.jpg")

    # Find the best model and plot the decision tree
    best_model_index = results_regularised["test_accuracies"].index(max(results_regularised["test_accuracies"]))

    # Find the hyperparameters of the best model
    best_max_depth = best_model_index // 100 + 1
    best_min_samples_split = (best_model_index % 100) // 10 * 3 + 3
    best_min_samples_leaf = best_model_index % 10 + 1

    print("Best Model Hyperparameters: Max Depth: {}, Min Samples Split: {}, Min Samples Leaf: {}".format(best_max_depth, best_min_samples_split, best_min_samples_leaf))

    # Train the best model
    tree = DecisionTreeClassifier(max_depth=best_max_depth, min_samples_split=best_min_samples_split, min_samples_leaf=best_min_samples_leaf, random_state=42, criterion="entropy")
    tree.fit(X_train, Y_train)

    # Find the best model accuracy
    yhat_tree = tree.predict(X_test)
    yhat_tree_train = tree.predict(X_train)
    best_model_accuracy = metrics.accuracy_score(Y_test, yhat_tree)
    best_model_train_accuracy = metrics.accuracy_score(Y_train, yhat_tree_train)

    print("Best Model Accuracy: Train: {}, Test: {}".format(best_model_train_accuracy, best_model_accuracy))

    # plot the decision tree
    # dt.plot_tree()
    

if __name__ == "__main__":
    main()
