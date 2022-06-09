def featureImportanceForest(X_data, y_data, n_estimators, n_jobs,  min_samples_leaf = 1, graph_title = 'The Graph', random_state = 42, only_graph = True):
    from sklearn.ensemble import RandomForestRegressor

    try:
        if only_graph == False:
            rfr = RandomForestRegressor(n_estimators = n_estimators, min_samples_leaf = min_samples_leaf, n_jobs = n_jobs, random_state = random_state)
            rfr.fit(X_data, y_data)
            importances = rfr.feature_importances_
            std = np.std([tree.feature_importances_ for tree in rfr.estimators_], axis = 0)
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize = (15,15))
            plt.title(graph_title, fontdict = {'fontsize':15})
            plt.bar(range(X_data.shape[1]), importances[indices], color = 'r', yerr = std[indices], align = 'center')
            plt.xticks(range(X_data.shape[1]), X_data.columns[indices], rotation = 90, fontsize = 13)
            plt.show()
            return np.array([importances, std, indices])

        else:
            rfr = RandomForestRegressor(n_estimators = n_estimators, min_samples_leaf = min_samples_leaf, n_jobs = n_jobs, random_state = random_state)
            rfr.fit(X_data, y_data)
            importances = rfr.feature_importances_
            std = np.std([tree.feature_importances_ for tree in rfr.estimators_], axis = 0)
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize = (15,15))
            plt.title(graph_title, fontdict = {'fontsize':15})
            plt.bar(range(X_data.shape[1]), importances[indices], color = 'r', yerr = std[indices], align = 'center')
            plt.xticks(range(X_data.shape[1]), X_data.columns[indices], rotation = 90, fontsize = 13)
            plt.show()

    except ValueError:
        print('Please check column(s) in data frame whether that includes object or NaN.')
    except NameError:
        print(
            'Please import modules \n'
            'import numpy as np \n'
            'import matplotlib.pyplot as plt'
            )

        # except pd.core.Indexing.IndexingError:    # I think the IndexingError isn't default error type in pure python.
        #     print('Please check range of y_data, if y_data is Series you never write 2nd colon that means columns')