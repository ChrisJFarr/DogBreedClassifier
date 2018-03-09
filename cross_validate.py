# from sklearn.model_selection import KFold
# from dog_fine_tuning import load_data, create_model, train_eval
#
#
# if __name__ == "__main__":
#     n_folds = 10
#     kfold = KFold(n_splits=n_folds, shuffle=True)
#
#     for train, test in kfold.split():
#             print("Running Fold", i+1, "/", n_folds)
#             model = None # Clearing the NN.
#             model = create_model()
#             train_and_evaluate_model(model, data[train], labels[train], data[test], labels[test))