import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import prince


import pandas
import webbrowser
import os


# View the original dataset
def loadDataset():
    # Read the dataset into a data table using Pandas
    data_table = pandas.read_csv("primary-tumor1.csv")

    # Create a web page view of the data for easy viewing
    # Display the first 100 records
    html = data_table[0:339].to_html()

    # Save the html to a temporary file
    with open("data.html", "w") as f:
        f.write(html)

    # Open the web page in our web browser
    full_filename = os.path.abspath("data.html")
    webbrowser.open("file://{}".format(full_filename))




def get_details(data_frame):
    print(data_frame.shape)
    print(data_frame.head(10))
    print(data_frame.info())
    print(data_frame.describe())

    # Get missing datasets - https://www.kaggle.com/funkegoodvibe/comprehensive-data-exploration-with-python
    print("\n\nTotal no of missing values %d \n" % data_frame.isnull().values.sum())
    if(data_frame.isnull().values.sum() != 0):
        total = data_frame.isnull().sum().sort_values(ascending=False)
        percent = (data_frame.isnull().sum() / data_frame.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        print(missing_data.head(20))




def check_duplicates(data_frame):
    print("##############  Duplicate records  ############## ")
    duplicate_records = data_frame[data_frame.duplicated()]
    is_duplicated = pd.DataFrame(duplicate_records).size != 0
    if(is_duplicated):
        print("Size of duplicate Records ", duplicate_records.shape)
        print(duplicate_records.head())
    else:
        print("No duplicate records found")
    return is_duplicated





def visualize_class_distribution(data_frame, target_name):
    # sns.countplot(data_frame['class'], label="Count") - method 1
    plt.figure(figsize=(14,8))
    Y = data_frame[target_name]
    total = len(Y)*1
    majority_count = len(data_frame[data_frame[target_name] == '1'])
    ax=sns.countplot(x=target_name, data=data_frame)

    for p in ax.patches:
        ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.1, p.get_height()+5))

    #put 11 ticks (therefore 10 steps), from 0 to the total nusmber of rows in the dataframe
    # ax.yaxis.set_ticks(np.linspace(0, total, 11)) # gives 309 at d top of d y axis
    ax.yaxis.set_ticks(np.linspace(0, majority_count, 11))
    #adjust the ticklabel to the desired format, without changing the position of the ticks.
    # ax.set_yticklabels(map('{:.1f}%'.format, 100*ax.yaxis.get_majorticklocs()/total)) - y axis values r display as %s
    ax.set_yticklabels(map('{:.0f}'.format, ax.yaxis.get_majorticklocs()))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=360, ha="right")
    # Use a LinearLocator to ensure the correct number of ticks
    # And use a MultipleLocator to ensure a tick spacing of 10
    # Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
    # ax.legend(labels=["lung","salivary g", "Pancreas", "gall bladder", "liver"])
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of instances")
    plt.show()






# Visualizes the feature distribution with box plots.
# source - https://towardsdatascience.com/designing-a-feature-selection-pipeline-in-python-859fec4d1b12
def visualise_feature_distribution(data_frame):
    # Set graph style
    sns.set(font_scale=0.75)
    sns.set_style({'axes.facecolor': '1.0', 'axes.edgecolor': '0.85', 'grid.color': '0.85',
                   'grid.linestyle': '-', 'axes.labelcolor': '0.4', 'xtick.color': '0.4',
                   'ytick.color': '0.4', 'axes.grid': False})

    # Create box plots based on feature type
    # Set the figure size
    f, ax = plt.subplots(figsize=(9, 14))
    sns.boxplot(data=data_frame, orient="h", palette="Set2")  # X
    # Set axis label
    plt.xlabel('Feature Value')
    # Tight layout
    f.tight_layout()
    # Save figure
    f.savefig('Box Plots.png', dpi=1080)
    plt.show()







def perform_correspondence_analysis(data_frame):
    mca = prince.MCA()

    mca = prince.MCA(
            n_components=2,
            n_iter=3,
            copy=True,
            check_input=True,
            engine='auto',
            random_state=42
            )
    ptumor_mca = mca.fit(data_frame)

    ax = ptumor_mca.plot_coordinates(
            X=data_frame,
            ax=None,
            figsize=(10, 10),
            show_row_points=False,
            row_points_size=0,
            show_row_labels=False,
            show_column_points=True,
            column_points_size=30,
            show_column_labels=True,
            legend_n_cols=1
                   ).legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()



