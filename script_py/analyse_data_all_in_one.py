import pandas as pd
import numpy as np
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('data_ad_1',
                   help='Path of the volume data')
    p.add_argument('data_ad_2',
                   help='Path of the volume data')
    p.add_argument('data_cn_1',
                   help='Path of the volume data')
    p.add_argument('data_cn_2',
                   help='Path of the volume data')
    p.add_argument('bundles', type=str,
                   help='Bundles selected for the analysis')

    return p

def classification_json(data):
    df = pd.DataFrame.from_records(
        [
            (level1, level2, level3, level4, leaf)
            for level1, level2_dict in data.items()
            for level2, level3_dict in level2_dict.items()
            for level3, level4_dict in level3_dict.items()
            for level4, leaf in level4_dict.items()
        ],
        columns=['Bundles', 'Metrics', 'Sections', 'Mean/Std','value'])
    
    return(df)

def selection_json(data, bundles, metrics):
    df = (data[(data['Bundles']==bundles) & (data['Metrics']==metrics)])
    return df

def get_mean_value(data):
    df = (data.iloc[:,4])
    df = df.values.tolist()
    df_t = list(map(list, zip(*df)))
    mean=np.mean(df_t, axis=0)
    return (mean)


def get_value(data):
    df = (data.iloc[:,4])
    df = df.values.tolist()
    df_t = list(map(list, zip(*df)))
    return (df_t)




def get_graph(mean_1, mean_2, std_1, std_2, index, bundles, metrics):

    fig, ax = plt.subplots()
    ax.plot(index, mean_1, "r",  label=f"AD", linewidth=2)
    ax.fill_between( index, mean_1-std_1, mean_1+std_1,  color='lightcoral', alpha=0.2)
    ax.plot(index, mean_2, "b",  label=f"CN", linewidth=2)
    ax.fill_between( index, mean_2-std_2, mean_2+std_2,  color='royalblue', alpha=0.2)
    plt.suptitle("comparison of diffusion metrics between AD and CN patients for the {0}".format(bundles), fontsize = 12)
    ax.set_xlabel('sections')
    ax.set_ylabel(metrics)
    ax.set_xlim(1, len(index))
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_facecolor('#EBEBEB')
    [ax.spines[side].set_visible(False) for side in ax.spines]
    ax.grid(which='major', color='white', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    plt.plot(color='blue', label='AD', linewidth=2)  # Ligne bleue pour AD
    plt.plot(color='red', label='CN', linewidth=2)   # Ligne rouge pour CN
    plt.legend(loc="upper right")
    return(fig)

def get_graph_all_subject(mean_1, mean_2, index, bundles, metrics):
    fig, ax = plt.subplots()
    for line in mean_1:
        ax.plot(index, line, "r")

    for line in mean_2:
        ax.plot(index, line, "b")

    plt.title("Graphique des listes")
    plt.suptitle(bundles, fontsize = 12)
    ax.set_xlabel("Section")
    ax.set_ylabel(metrics)
    plt.grid(True)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.plot([], [], color='blue', label='AD')  # Ligne bleue pour AD
    plt.plot([], [], color='red', label='CN')   # Ligne rouge pour CN
    plt.legend(loc="upper right")

def graph_seaborn(mean_1, mean_2, mean_3, mean_4, std_1, std_2, std_3, std_4, index, bundles):

    data = pd.DataFrame({
    'section': index * 4,
    'mean FA': list(mean_1) + list(mean_2) + list(mean_3) + list(mean_4),
    'sd': list(std_1) + list(std_2) + list(std_3) + list(std_4),
    'subject': ['AD'] * 20 + ['CN'] * 20 + ['AD'] * 20 + ['CN'] * 20,
    'time': ['first session(y0)'] * 40 + ['second session(y2)'] * 40
    })

    sns.set_theme(style="whitegrid")
    g = sns.catplot(
    data=data, x="section", y="mean FA", hue="subject", kind="point", 
    ci="sd", capsize=.2, height=6, aspect=1.5, col="time"
    )
    g.despine(left=True)
    plt.suptitle("comparison of diffusion metrics between AD and CN patients for the {0}".format(bundles), fontsize = 12)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    data_AD_1 = pd.read_json(args.data_ad_1)
    df_AD_1 = classification_json(data_AD_1)   
    df_AD_1_fa = selection_json(df_AD_1, args.bundles, 'fa_metric')
    df_AD_1_fa_mean = get_value(df_AD_1_fa[(df_AD_1_fa['Mean/Std']=="mean")])
    df_AD_1_fa_mean_global = get_mean_value(df_AD_1_fa[(df_AD_1_fa['Mean/Std']=="mean")])
    df_AD_1_fa_std = get_value(df_AD_1_fa[(df_AD_1_fa['Mean/Std']=="std")])
    df_AD_1_fa_std_global = get_mean_value(df_AD_1_fa[(df_AD_1_fa['Mean/Std']=="std")])
    df_AD_1_md = selection_json(df_AD_1, args.bundles, 'md_metric')
    df_AD_1_md_mean = get_value(df_AD_1_md[(df_AD_1_md['Mean/Std']=="mean")])
    df_AD_1_md_mean_global = get_mean_value(df_AD_1_md[(df_AD_1_md['Mean/Std']=="mean")])
    df_AD_1_md_std = get_value(df_AD_1_md[(df_AD_1_md['Mean/Std']=="std")])
    df_AD_1_md_std_global = get_mean_value(df_AD_1_md[(df_AD_1_md['Mean/Std']=="std")])


    data_AD_2 = pd.read_json(args.data_ad_2)
    df_AD_2 = classification_json(data_AD_2)   
    df_AD_2_fa = selection_json(df_AD_2, args.bundles, 'fa_metric')
    df_AD_2_fa_mean = get_value(df_AD_2_fa[(df_AD_2_fa['Mean/Std']=="mean")])
    df_AD_2_fa_mean_global = get_mean_value(df_AD_2_fa[(df_AD_2_fa['Mean/Std']=="mean")])
    df_AD_2_fa_std = get_value(df_AD_2_fa[(df_AD_2_fa['Mean/Std']=="std")])
    df_AD_2_fa_std_global = get_mean_value(df_AD_2_fa[(df_AD_2_fa['Mean/Std']=="std")])
    df_AD_2_md = selection_json(df_AD_2, args.bundles, 'md_metric')
    df_AD_2_md_mean = get_value(df_AD_2_md[(df_AD_2_md['Mean/Std']=="mean")])
    df_AD_2_md_mean_global = get_mean_value(df_AD_2_md[(df_AD_2_md['Mean/Std']=="mean")])
    df_AD_2_md_std = get_value(df_AD_2_md[(df_AD_2_md['Mean/Std']=="std")])
    df_AD_2_md_std_global = get_mean_value(df_AD_2_md[(df_AD_2_md['Mean/Std']=="std")])

    data_CN_1 = pd.read_json(args.data_cn_1)
    df_CN_1 = classification_json(data_CN_1)   
    df_CN_1_fa = selection_json(df_CN_1, args.bundles, 'fa_metric')
    df_CN_1_fa_mean = get_value(df_CN_1_fa[(df_CN_1_fa['Mean/Std']=="mean")])
    df_CN_1_fa_mean_global = get_mean_value(df_CN_1_fa[(df_CN_1_fa['Mean/Std']=="mean")])
    df_CN_1_fa_std = get_value(df_CN_1_fa[(df_CN_1_fa['Mean/Std']=="std")])
    df_CN_1_fa_std_global = get_mean_value(df_CN_1_fa[(df_CN_1_fa['Mean/Std']=="std")])
    df_CN_1_md = selection_json(df_CN_1, args.bundles, 'md_metric')
    df_CN_1_md_mean = get_value(df_CN_1_md[(df_CN_1_md['Mean/Std']=="mean")])
    df_CN_1_md_mean_global = get_mean_value(df_CN_1_md[(df_CN_1_md['Mean/Std']=="mean")])
    df_CN_1_md_std = get_value(df_CN_1_md[(df_CN_1_md['Mean/Std']=="std")])
    df_CN_1_md_std_global = get_mean_value(df_CN_1_md[(df_CN_1_md['Mean/Std']=="std")])

    data_CN_2 = pd.read_json(args.data_cn_2)
    df_CN_2 = classification_json(data_CN_2)   
    df_CN_2_fa = selection_json(df_CN_2, args.bundles, 'fa_metric')
    df_CN_2_fa_mean = get_value(df_CN_2_fa[(df_CN_2_fa['Mean/Std']=="mean")])
    df_CN_2_fa_mean_global = get_mean_value(df_CN_2_fa[(df_CN_2_fa['Mean/Std']=="mean")])
    df_CN_2_fa_std = get_value(df_CN_2_fa[(df_CN_2_fa['Mean/Std']=="std")])
    df_CN_2_fa_std_global = get_mean_value(df_CN_2_fa[(df_CN_2_fa['Mean/Std']=="std")])
    df_CN_2_md = selection_json(df_CN_2, args.bundles, 'md_metric')
    df_CN_2_md_mean = get_value(df_CN_2_md[(df_CN_2_md['Mean/Std']=="mean")])
    df_CN_2_md_mean_global = get_mean_value(df_CN_2_md[(df_CN_2_md['Mean/Std']=="mean")])
    df_CN_2_md_std = get_value(df_CN_2_md[(df_CN_2_md['Mean/Std']=="std")])
    df_CN_2_md_std_global = get_mean_value(df_CN_2_md[(df_CN_2_md['Mean/Std']=="std")])

    index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    graph_1_fa_global=get_graph(df_AD_1_fa_mean_global, df_CN_1_fa_mean_global, df_AD_1_fa_std_global, df_CN_1_fa_std_global, index, args.bundles, "fa")
    plt.show()
    graph_2_fa_global=get_graph(df_AD_2_fa_mean_global, df_CN_2_fa_mean_global, df_AD_2_fa_std_global, df_CN_2_fa_std_global, index, args.bundles, "fa")
    plt.show()
    '''
    graph_1_fa_subject=get_graph_all_subject(df_AD_1_fa_mean, df_CN_1_fa_mean, index, args.bundles, "fa")
    plt.show()
    graph_1_md_global=get_graph(df_AD_1_md_mean_global, df_CN_1_md_mean_global, df_AD_1_md_std_global, df_CN_1_md_std_global, index, args.bundles, "md")
    plt.show()
    graph_1_md_subject=get_graph_all_subject(df_AD_1_md_mean, df_CN_1_md_mean, index, args.bundles, "md")
    plt.show()
    graph_2_fa_global=get_graph(df_AD_2_fa_mean_global, df_CN_2_fa_mean_global, df_AD_2_fa_std_global, df_CN_2_fa_std_global, index, args.bundles, "fa")
    plt.show()
    graph_2_fa_subject=get_graph_all_subject(df_AD_2_fa_mean, df_CN_2_fa_mean, index, args.bundles, "fa")
    plt.show()
    graph_2_md_global=get_graph(df_AD_2_md_mean_global, df_CN_2_md_mean_global, df_AD_2_md_std_global, df_CN_2_md_std_global, index, args.bundles, "fa")
    plt.show()
    graph_2_md_subject=get_graph_all_subject(df_AD_2_md_mean, df_CN_2_md_mean, index, args.bundles, "md")
    plt.show()
    '''
    graph_global=graph_seaborn(df_AD_1_fa_mean_global, df_CN_1_fa_mean_global, df_AD_2_fa_mean_global, df_CN_2_fa_mean_global,
                              df_AD_1_fa_std_global, df_CN_1_fa_std_global, df_AD_2_fa_std_global, df_CN_2_fa_std_global, index, args.bundles)
    
    difference_fa_1 = df_CN_1_fa_mean_global - df_AD_1_fa_mean_global
    difference_fa_1 = [0 if x < 0 else x for x in difference_fa_1]
    difference_fa_2 = df_CN_2_fa_mean_global - df_AD_2_fa_mean_global
    difference_fa_2 = [0 if x < 0 else x for x in difference_fa_2]
    
    with open("df_AD_1_fa_mean_global.txt", "w") as output:
        output.write(str(df_AD_1_fa_mean_global))

    with open("df_AD_2_fa_mean_global.txt", "w") as output:
        output.write(str(df_AD_2_fa_mean_global))

    with open("df_CN_1_fa_mean_global.txt", "w") as output:
        output.write(str(df_CN_1_fa_mean_global))

    with open("df_CN_2_fa_mean_global.txt", "w") as output:
        output.write(str(df_CN_2_fa_mean_global))

    with open("difference_fa_1.txt", "w") as output:
        output.write(str(difference_fa_1))

    with open("difference_fa_2.txt", "w") as output:
        output.write(str(difference_fa_2))
    

    # Spécifier le chemin du nouveau répertoire et du fichier
    bundle_dossier = args.bundles
    fichier_df_AD_1_fa_mean_global = "df_AD_1_fa_mean_global_{0}.txt".format(bundle_dossier)
    fichier_df_AD_2_fa_mean_global = "df_AD_2_fa_mean_global_{0}.txt".format(bundle_dossier)
    fichier_df_CN_1_fa_mean_global = "df_CN_1_fa_mean_global_{0}.txt".format(bundle_dossier)
    fichier_df_CN_2_fa_mean_global = "df_CN_2_fa_mean_global_{0}.txt".format(bundle_dossier)
    fichier_difference_fa_1 = "difference_fa_1_{0}.txt".format(bundle_dossier)
    fichier_difference_fa_2 = "difference_fa_2_{0}.txt".format(bundle_dossier)


    # Créer le nouveau répertoire s'il n'existe pas
    if not os.path.exists(bundle_dossier):
        os.makedirs(bundle_dossier)

    # Créer le chemin complet du fichier
    chemin_fichier_df_AD_1_fa_mean_global = os.path.join(bundle_dossier, fichier_df_AD_1_fa_mean_global)
    np.savetxt(chemin_fichier_df_AD_1_fa_mean_global, df_AD_1_fa_mean_global, fmt="%g")
    chemin_fichier_df_AD_2_fa_mean_global = os.path.join(bundle_dossier, fichier_df_AD_2_fa_mean_global)
    np.savetxt(chemin_fichier_df_AD_2_fa_mean_global, df_AD_2_fa_mean_global, fmt="%g")
    chemin_fichier_df_CN_1_fa_mean_global = os.path.join(bundle_dossier, fichier_df_CN_1_fa_mean_global)
    np.savetxt(chemin_fichier_df_CN_1_fa_mean_global, df_CN_1_fa_mean_global, fmt="%g")
    chemin_fichier_df_CN_2_fa_mean_global = os.path.join(bundle_dossier, fichier_df_CN_2_fa_mean_global)
    np.savetxt(chemin_fichier_df_CN_2_fa_mean_global, df_CN_2_fa_mean_global, fmt="%g")
    chemin_fichier_difference_fa_1 = os.path.join(bundle_dossier, fichier_difference_fa_1)
    np.savetxt(chemin_fichier_difference_fa_1, difference_fa_1, fmt="%g")
    chemin_fichier_difference_fa_2 = os.path.join(bundle_dossier, fichier_difference_fa_2)
    np.savetxt(chemin_fichier_difference_fa_2, difference_fa_2, fmt="%g")

    plt.show()

if __name__ == "__main__":
    main()