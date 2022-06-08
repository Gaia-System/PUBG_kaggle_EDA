import pandas as pd


def loadData(data_path):
    if str(data_path).endswith('.csv'):
        return pd.read_csv(str(data_path))
    elif str(data_path).endswith('.xlsx') or str(data_path).endswith('.xls'):
        return pd.read_excel(str(data_path))
    else:
        print('Format Error, Please load ONLY xls, xlsx or csv file')

# X_y_split method will be occur the error
def X_y_split(dataset, startrow, endrow, Xstartcol, Xendcol, ystartcol, yendcol):
    train_X = dataset.iloc[startrow : endrow, Xstartcol : Xendcol]
    train_y = dataset.iloc[startrow : endrow, ystartcol : yendcol]
    return train_X, train_y

def dropData(dataset, idx_list, col_list, inplace = False):
    if inplace == False:
        dataset.drop(index = idx_list, columns = col_list)
        return dataset
    else:
        dataset = dataset.drop(index = idx_list, columns = col_list)
        return dataset
