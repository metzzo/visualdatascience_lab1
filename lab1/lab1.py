import pandas as pd
import numpy as np

import pickle

def load_dataset(path="../../nutritions.csv"):
    dataset = pd.read_csv(
        filepath_or_buffer=path,
        sep=";",  # ; used to separate
        header=0,  # 0th row has header
        doublequote=True
    )

    # drop entries with empty rows
    dataset.replace('', np.nan, inplace=True)
    dataset.dropna(inplace=True)

    def normalize_row(row):
        return row.strip()
    dataset['Shrt_Desc'] = dataset['Shrt_Desc'].apply(normalize_row)

    return dataset


def extract_categories(dataset):
    category_sizes = {}

    def extract_categories_per_row(row):
        categories = row['Shrt_Desc'].split(',')
        for category in categories:
            if category not in category_sizes:
                category_sizes[category] = 1
            else:
                category_sizes[category] += 1

    print("Extracting Categories")
    dataset.apply(extract_categories_per_row, axis=1)
    print("Extracting successful, No. of Categories: {}".format(len(category_sizes.keys())))

    #print("Filter out categories which are underrepresented")
    #threshold = 5
    #filtered_categories = {k: v for k, v in category_sizes.items() if v > threshold}
    #print("Threshold for filtering {0}, new size {1}".format(threshold, len(filtered_categories.keys())))

    categories = list(category_sizes.keys())
    print("Create labels_true")

    def create_labels(row):
        local_categories = row['Shrt_Desc'].split(',')
        data = {key: key in local_categories for key in categories}
        return pd.Series(data)

    labels_true = dataset.apply(create_labels, axis=1)
    print("Finished creating labels")

    return categories, labels_true

def statistical_clustering_by_energy(dataset, labels_true):
    from sklearn.cluster import AffinityPropagation
    clustering = AffinityPropagation().fit(
        X=dataset['Energ_Kcal'].values.reshape(-1, 1)
    )
    print(clustering.labels)



ds = load_dataset()
data = None
try:
    print("Load existing dataset")
    data = pickle.load(open("categories.p", "rb"))
except:
    print("Create dataset")
    data = extract_categories(dataset=ds)
    pickle.dump(data, open("categories.p", "wb"))

_, labels_true = data

statistical_clustering_by_energy(dataset=ds, labels_true=labels_true)