import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np

import pickle
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler


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
    category_set = set()

    def extract_categories_per_row(row):
        categories = row['Shrt_Desc'].split(',')
        for category in categories:
            if category not in category_set:
                category_set.add(category)

    print("Extracting Categories")
    dataset.apply(extract_categories_per_row, axis=1)
    print("Extracting successful, No. of Categories: {}".format(len(category_set)))

    categories = list(category_set)
    print("Create Labels")

    def create_labels(row):
        local_categories = row['Shrt_Desc'].split(',')
        data = {key: float(key in local_categories) for key in categories}
        return pd.Series(data)

    labels = dataset.apply(create_labels, axis=1)
    print("Finished creating labels")

    return categories, labels


def statistical_clustering_by_energy(dataset, labels):
    labels['Energ_Kcal'] = dataset['Energ_Kcal']
    X = StandardScaler().fit_transform(labels.values)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=['principal component 1', 'principal component 2'])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    ax.scatter(
        principalDf['principal component 1'],
        principalDf['principal component 2'],
        s=50
    )

    ax.grid()
    plt.show()
    plt.waitforbuttonpress()

    """
    clustering = AffinityPropagation().fit(
        X=X
    )
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()"""


ds = load_dataset()
data = None
try:
    print("Load existing dataset")
    data = pickle.load(open("categories.p", "rb"))
except:
    print("Create dataset")
    data = extract_categories(dataset=ds)
    pickle.dump(data, open("categories.p", "wb"))

_, labels = data

statistical_clustering_by_energy(dataset=ds, labels=labels)