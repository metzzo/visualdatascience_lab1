import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import plotly.offline as offline
import plotly.graph_objs as go

import pickle
from sklearn.cluster import KMeans

nutritions = ["Water_(g)", "Energ_Kcal", "Protein_(g)", "Lipid_Tot_(g)", "Ash_(g)", "Carbohydrt_(g)", "Fiber_TD_(g)",
              "Sugar_Tot_(g)", "Calcium_(mg)", "Iron_(mg)", "Magnesium_(mg)", "Phosphorus_(mg)", "Potassium_(mg)",
              "Sodium_(mg)", "Zinc_(mg)", "Copper_mg)", "Manganese_(mg)", "Selenium_(µg)", "Vit_C_(mg)", "Thiamin_(mg)",
              "Riboflavin_(mg)", "Niacin_(mg)", "Panto_Acid_mg)", "Vit_B6_(mg)", "Folate_Tot_(µg)", "Folic_Acid_(µg)",
              "Food_Folate_(µg)", "Folate_DFE_(µg)", "Choline_Tot_ (mg)", "Vit_B12_(µg)", "Vit_A_IU", "Vit_A_RAE",
              "Retinol_(µg)", "Alpha_Carot_(µg)", "Beta_Carot_(µg)", "Beta_Crypt_(µg)", "Lycopene_(µg)",
              "Lut+Zea_ (µg)", "Vit_E_(mg)", "Vit_D_µg", "Vit_D_IU", "Vit_K_(µg)", "FA_Sat_(g)", "FA_Mono_(g)",
              "FA_Poly_(g)", "Cholestrl_(mg)"]


def load_dataset(path="../../nutritions.csv"):
    dataset = pd.read_csv(
        filepath_or_buffer=path,
        sep=";",  # ; used to separate
        header=0,  # 0th row has header
        doublequote=True
    )

    # drop entries with empty rows
    dataset.replace('', np.nan, inplace=True)
    #dataset.dropna(inplace=True)

    return dataset


def extract_categories(dataset):
    category_set = set()

    def get_single_categories(row):
        cats = [v.strip().upper() for v in row['Shrt_Desc'].split(',') if len(v) > 0]
        add_sugarfree = False
        for c in cats:
            if 'SUGAR FREE' in c and c != 'SUGAR FREE':
                add_sugarfree = True
        if add_sugarfree:
            cats.append('SUGAR FREE')
        return cats

    def extract_categories_per_row(row):
        categories = get_single_categories(row)
        for category in categories:
            category_set.add(category)

    print("Extracting Categories")
    dataset.apply(extract_categories_per_row, axis=1)
    print("Extracting successful, No. of Categories: {}".format(len(category_set)))

    categories = list(category_set)
    print("Create Labels")

    def create_labels(row):
        local_categories = get_single_categories(row)
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
        if not np.isnan(energy):
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


def statistical_clustering_by_energy(dataset):
    print("Run Clustering")

    # this causes K-Means to perform very badly
    #X = StandardScaler().fit_transform(labels.values)
    #print(*X[0])
    df = dataset[['Energ_Kcal', 'Shrt_Desc']].dropna()
    energies = df['Energ_Kcal']

    X = energies.values.reshape(-1,1)
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
        original_category = df.iloc[index]['Shrt_Desc']
        calories = energies.iloc[index]
        print("Cluster {0} for {1} with calories {2}".format(cluster, original_category, calories))

def visual_clustering_by_energy(dataset):
    # the histogram of the data

    X = sorted(list(dataset['Energ_Kcal'].dropna()))
    print(*X)
    plt.hist(X, bins=int(len(dataset)/256))


    plt.xlabel('Energy')
    plt.ylabel('Amount')
    plt.xticks(range(int(X[0]), int(X[-1]), 50))
    plt.grid(True)

    plt.show()


def find_statistical_correlations(dataset1, dataset2):
    if dataset2 is None:
        dataset2 = dataset1

    correlations = []
    already_done = set()
    for n1 in nutritions:
        for n2 in nutritions:
            if n1 != n2 and n1+n2 not in already_done and n2+n1 not in already_done:
                ds1 = dataset1[n1].copy().dropna()
                ds2 = dataset2[n2].copy().dropna()
                if len(ds1) != len(ds2):
                    if len(ds1) > len(ds2):
                        # first make sure that ds1 is the bigger ds
                        tmp = ds1
                        ds1 = ds2
                        ds2 = tmp

                    ds2 = ds2[0:len(ds1)]

                corr = np.corrcoef(ds1, ds2)[0, 1]
                if not np.isnan(corr):
                    correlations.append({
                        "corr": corr,
                        "n1": n1,
                        "n2": n2
                    })
                    print("{0} vs {1}: {2}".format(n1, n2, corr))
                already_done.add(n1+n2)
                already_done.add(n2+n1)
    correlations = sorted(correlations, key=lambda x: abs(x['corr']))
    print(*correlations[-4:-1])


def find_visual_correlations(dataset):
    data = [
        go.Parcoords(
            line=dict(color=dataset['NDB_No'],
                      colorscale=[[0, '#D7C16B'], [0.5, '#23D8C3'], [1, '#F3F10F']]),
            dimensions=list([
                dict(label='Energy', values=dataset['Energ_Kcal']),
                dict(label='Protein', values=dataset['Protein_(g)']),
                dict(label='Lipid', values=dataset['Lipid_Tot_(g)']),
                dict(label='Sugar', values=dataset['Sugar_Tot_(g)']),
                dict(label='Water', values=dataset['Water_(g)']),
                dict(label='Folate (Total)', values=dataset['Folate_Tot_(µg)']),
                dict(label='Folate (DFE)', values=dataset['Folate_DFE_(µg)']),
                dict(label='Vitamin A', values=dataset['Vit_A_IU']),
                dict(label='Beta Carot', values=dataset['Beta_Carot_(µg)']),
            ])
        )
    ]

    layout = go.Layout(
        plot_bgcolor='#E5E5E5',
        paper_bgcolor='#E5E5E5'
    )

    fig = go.Figure(data=data, layout=layout)
    offline.plot(fig, filename='parcoords-basic')


def extract_sugar_free_vs_non_sugar_free(dataset, labels, categories):
    dataset = pd.concat([dataset, labels], axis=1, sort=False)
    sugarfree_ds = dataset[dataset['SUGAR FREE'] > 0]
    try:
        non_sugarfree_ds = pickle.load(open("sugar_free.p", "rb"))
    except:
        print("Extract sugar free vs non sugar free")

        relevant_categories = []
        for c in categories:
            if c != 'SUGAR FREE' and sugarfree_ds[c].max() > 0:
                relevant_categories.append(c)
        print(relevant_categories)

        def filter_out_categories(ds, relevant_categories, categories, or_for_relevant=False):
            conditions = None
            for c in relevant_categories:
                condition = ds[c] > 0

                if conditions is None:
                    conditions = condition
                else:
                    if or_for_relevant:
                        conditions = condition | conditions
                    else:
                        conditions = condition & conditions
            """
            ds = ds[conditions]
            conditions = None
            for c in categories:
                if c not in relevant_categories:
                    condition = ds[c] < 1
    
                    if conditions is None:
                        conditions = condition
                    else:
                        conditions = condition & conditions"""
            ds = ds[conditions]
            return ds

        dataset = filter_out_categories(
            ds=dataset,
            relevant_categories=relevant_categories,
            categories=categories,
            or_for_relevant=True
        )
        print("Shape of pre-filtered dataset {0}".format(dataset.shape))

        datasets = []
        for _, sugarfree in sugarfree_ds.iterrows():
            new_dataset = dataset.copy()
            my_relevant_categories = []
            for c in categories:
                if c == 'SUGAR FREE':
                    continue

                if sugarfree[c] > 0:
                    my_relevant_categories.append(c)
            print("Relevant Categories {0}".format(my_relevant_categories))
            new_dataset = filter_out_categories(
                ds=new_dataset,
                relevant_categories=my_relevant_categories,
                categories=relevant_categories,
                or_for_relevant=True
            )
            avgs = {}
            for c in nutritions:
                avgs[c] = new_dataset[c].mean()
            datasets.append(pd.DataFrame(data=[avgs]))

        non_sugarfree_ds = pd.concat(datasets)

        pickle.dump(non_sugarfree_ds, open("sugar_free.p", "wb"))
    print("Shapes Sugarfree: {0} Non-Sugarfree: {1}".format(sugarfree_ds.shape, non_sugarfree_ds.shape))
    find_statistical_correlations(dataset1=sugarfree_ds, dataset2=non_sugarfree_ds)
    visually_compare_sugar_non_sugar(
        sugar_ds=non_sugarfree_ds,
        non_sugar_ds=sugarfree_ds
    )


def visually_compare_sugar_non_sugar(sugar_ds, non_sugar_ds):
    def compare(columns,filename):
        data_sugar = [sugar_ds[c].mean() for c in columns]
        data_non_sugar = [non_sugar_ds[c].mean() for c in columns]
        data = [
            go.Scatterpolar(
                r=data_sugar,
                theta=columns,
                fill='toself',
                name='Sugar'
            ),
            go.Scatterpolar(
                r=data_non_sugar,
                theta=columns,
                fill='toself',
                name='Non-Sugar'
            )
        ]

        layout = go.Layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 50]
                )
            ),
            showlegend=False
        )

        fig = go.Figure(data=data, layout=layout)
        offline.plot(fig, filename=filename)
    compare(
        columns=['Water_(g)', 'Lipid_Tot_(g)', 'Carbohydrt_(g)', 'Sodium_(mg)', 'FA_Mono_(g)', 'Energ_Kcal'],
        filename="correlated"
    )
    compare(
        columns=['Water_(g)', 'Sugar_Tot_(g)', 'Copper_mg)', 'Folate_DFE_(µg)', 'Choline_Tot_ (mg)', 'Alpha_Carot_(µg)'],
        filename = "uncorrelated"
    )


ds = load_dataset()
try:
    print("Load existing dataset")
    data = pickle.load(open("categories.p", "rb"))
    categories, labels, averages = data
except:
    print("Create dataset")
    data = extract_categories(dataset=ds)
    categories, labels = data
    averages = []
    #averages = calculate_average_energy_per_category(
    #    dataset=ds,
    #    labels=labels,
    #    categories=categories
    #)
    data = categories, labels, averages
    pickle.dump(data, open("categories.p", "wb"))

# filter out categories with only 1 entry
#averages = averages[averages[2] > 1]

#statistical_clustering_by_energy(dataset=ds)
#visual_clustering_by_energy(dataset=ds)
#find_statistical_correlations(dataset1=ds)
#find_visual_correlations(dataset=ds)

extract_sugar_free_vs_non_sugar_free(dataset=ds, labels=labels, categories=categories)