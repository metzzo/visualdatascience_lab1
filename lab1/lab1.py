import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import pickle
from sklearn.cluster import KMeans
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

    return dataset


def extract_categories(dataset):
    category_set = set()

    def extract_categories_per_row(row):
        categories = row['Shrt_Desc'].split(',')
        for category in categories:
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


def calculate_average_energy_per_category(dataset, labels, categories):
    energy_data = {value: (value, 0, 0) for value in categories}
    print("Calculate Averages")

    for i in range(0, len(dataset)):
        row = dataset.iloc[i]
        labels_row = labels.iloc[i]
        energy = row['Energ_Kcal']
        for category in categories:
            if labels_row[category]:
                data = energy_data[category]
                data = (
                    data[0],
                    data[1] + energy,
                    data[2] + 1
                )
                energy_data[category] = data
    df = pd.DataFrame.from_dict(energy_data, orient='index')
    df['average'] = df[1] / df[2]
    return df


def statistical_clustering_by_energy(averages):
    print("Run Clustering")

    # this causes K-Means to perform very badly
    #X = StandardScaler().fit_transform(labels.values)
    #print(*X[0])

    X = averages['average'].values.reshape(-1,1)
    clustering = KMeans(
        init='k-means++',
        n_clusters=4,
        n_init=100,
        random_state=42
    ).fit(
        X=X
    )
    clusters = {
        2: 'Low calories',
        0: 'Medium-low calories',
        1: 'Medium-high calories',
        3: 'High calories',
    }

    for index, assigned_cluster in enumerate(clustering.labels_):
        cluster = clusters[assigned_cluster]
        original_category = averages[0].iloc[index]
        calories = averages['average'].iloc[index]
        print("Cluster {0} for {1} with calories {2}".format(cluster, original_category, calories))

def visual_clustering_by_energy(averages):
    # the histogram of the data

    X = sorted(list(averages['average']))
    print(*X)
    plt.hist(X, bins=int(len(averages)/16))


    plt.xlabel('Energy')
    plt.ylabel('Amount')
    plt.xticks(range(int(X[0]), int(X[-1]), 50))
    plt.grid(True)

    plt.show()


ds = load_dataset()
try:
    print("Load existing dataset")
    data = pickle.load(open("categories.p", "rb"))
    categories, labels, averages = data
except:
    print("Create dataset")
    data = extract_categories(dataset=ds)
    categories, labels = data
    averages = calculate_average_energy_per_category(
        dataset=ds,
        labels=labels,
        categories=categories
    )
    data = categories, labels, averages
    pickle.dump(data, open("categories.p", "wb"))

# filter out categories with only 1 entry
averages = averages[averages[2] > 1]

statistical_clustering_by_energy(averages=averages)
visual_clustering_by_energy(averages=averages)
