#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scilpy.io.utils import (add_overwrite_arg)

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    
    p.add_argument('data', nargs='+',
                   help='dataframe containing metrics data with differents range of NAWM')
    p.add_argument('--class', nargs='+',
                   help='dataframe containing metrics data with differents range of NAWM')



    add_overwrite_arg(p)
    return p

def extract_value_to_list(data, index):
    list_data = []
    for subject in data:
        subject_data = pd.read_json(subject)
        subject_value = subject_data[index].tolist()
        list_data.append(subject_value)
        flattened_list = [item[0][0] for item in list_data]
    return(flattened_list)

def get_list_subject_values(data):
    list_adjacency_voxels = extract_value_to_list(data, 'adjacency_voxels')
    list_dice_voxels = extract_value_to_list(data, 'dice_voxels')
    list_volume_overlap = extract_value_to_list(data, 'volume_overlap')
    list_volume_overreach = extract_value_to_list(data, 'volume_overreach')
    return (list_adjacency_voxels, list_dice_voxels, list_volume_overlap, list_volume_overreach)

def create_dataframe_complete(data, groupe):
    data = pd.DataFrame({
        'adjacency_voxels': data[0],
        'dice_voxels': data[1],
        'volume_overlap': data[2],
        'volume_overreach': data[3],
        'groupe': groupe
    })
    return(data)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    list_lt_c_l_vs_lt_c_nl = []
    for fichier in args.data:
        if "lt_c_l_vs_lt_c_nl" in fichier:
            list_lt_c_l_vs_lt_c_nl.append(fichier)
    values_lt_c_l_vs_lt_c_nl = get_list_subject_values(list_lt_c_l_vs_lt_c_nl)
    df_lt_c_l_vs_lt_c_nl = create_dataframe_complete(values_lt_c_l_vs_lt_c_nl, 'lt_c_l_vs_lt_c_nl')
    df_melted = pd.melt(df_lt_c_l_vs_lt_c_nl, id_vars=['groupe'], 
                    value_vars=['adjacency_voxels', 'dice_voxels', 'volume_overlap', 'volume_overreach'],
                    var_name='metric', value_name='value')
    print (df_melted[(df_melted['metric']=="dice_voxels")])
    #sns.set_theme(style="dark")
    #sns.violinplot(data=(df_melted[(df_melted['metric']=="dice_voxels")]), x="metric", y="value", hue="groupe",
    #          split=True, inner="quart", fill=False,
    #           palette={"lt_c_l_vs_lt_c_nl": "g", "T2_vs_T3": ".35"})
    #plt.show()
    
    sns.boxplot(data=(df_melted[(df_melted['metric']=="dice_voxels")]), x="metric", y="value", hue="groupe",
                legend=False, fill=False, width=0.80, 
                linewidth=.75, gap=.5, showfliers=False)

    
    #sns.despine(offset=10, trim=True)

    sns.stripplot(
        data=(df_melted[(df_melted['metric']=="dice_voxels")]), x="metric", y="value", hue="groupe",
        dodge=True, alpha=.25, jitter=False, )
    #sns.pointplot(
    #    data=df_melted, x="metric", y="value", hue="groupe",
    #    dodge=.8 - .8 / 3, palette="dark", errorbar=None,
    #   markers="d", markersize=4, linestyle="none",
    #)

    plt.show()


if __name__ == "__main__":
    main()
