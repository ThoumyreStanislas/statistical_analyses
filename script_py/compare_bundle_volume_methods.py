import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Compare volumes of selected bundles across methods.",
        formatter_class=argparse.RawTextHelpFormatter)
    
    p.add_argument('data', nargs='+', help='JSON files with volume for each timepoint and method')
    p.add_argument('--bundles', nargs='+', required=True, help='Bundles to compare (e.g., AF, IFOF, SLF)')
    p.add_argument('--output', type=str, help='File path to save the boxplot image (e.g., output.png)')
    
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    
    data = []

    # Parcours des fichiers JSON fournis en arguments
    for file_path in args.data:
        filename = os.path.basename(file_path)
        
        # Extraction de la méthode, timepoint, et sujet depuis le nom de fichier et le chemin
        method = file_path.split('/')[0]  # Méthode
        timepoint = filename.split('_')[4]  # Timepoint (e.g., T1, T2, T3)
        subject = '_'.join(filename.split('_')[:5])  # Sujet 
        
        # Chargement des données du fichier JSON
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            
            # Pour chaque faisceau demandé, récupérer les volumes
            for bundle in args.bundles:
                # Cas où le faisceau a des variantes L et R
                bundle_l = f"{bundle}_L"
                bundle_r = f"{bundle}_R"
                
                if bundle_l in json_data.get(f"{subject}", {}) and bundle_r in json_data.get(f"{subject}", {}):
                    # Si les variantes gauche/droite existent, on calcule la moyenne
                    volume_l = json_data[f"{subject}"][bundle_l].get("volume", None)
                    volume_r = json_data[f"{subject}"][bundle_r].get("volume", None)
                    
                    if volume_l is not None and volume_r is not None:
                        bundle_volume = (volume_l + volume_r) / 2
                        data.append({
                            'Subject': subject,
                            'Timepoint': timepoint,
                            'Method': method,
                            'Bundle': bundle,
                            'Bundle_Volume': bundle_volume
                        })
                
                # Cas où le faisceau n'a pas de variantes gauche/droite
                elif bundle in json_data.get(f"{subject}", {}):
                    bundle_volume = json_data[f"{subject}"][bundle].get("volume", None)
                    
                    if bundle_volume is not None:
                        data.append({
                            'Subject': subject,
                            'Timepoint': timepoint,
                            'Method': method,
                            'Bundle': bundle,
                            'Bundle_Volume': bundle_volume
                        })

    # Création du DataFrame
    df = pd.DataFrame(data)
    print (df)
    
    method_order = ['local_tracking_not_corrected', 'local_tracking_corrected', 'local_tracking_corrected_lesion']

    palette = {
    'local_tracking_not_corrected': '#e91212ff',
    'local_tracking_corrected': '#f3af00ff',
    'local_tracking_corrected_lesion': '#69bb47ff'
    }

    # Création du boxplot avec Seaborn
    plt.figure(figsize=(6, 6))
    sns.boxplot(data=df, x='Bundle', y='Bundle_Volume', hue='Method', hue_order=method_order,
        fill=False, width=0.80, linewidth=.75, gap=0.2, showfliers=False, legend=False, palette=palette)
    sns.stripplot(data=df, x='Bundle', y='Bundle_Volume', hue='Method', hue_order=method_order,
        legend=False, dodge=True, alpha=.25, jitter=True, palette=palette)
    plt.title('Comparison of average volume by bundles and methods', fontsize=14)
    plt.xlabel('Bundle', fontsize=14)
    plt.ylabel('Volume (mm³)', fontsize=12)
    plt.xticks(fontsize=12)

    # Sauvegarde ou affichage du plot
    if args.output:
        plt.savefig(args.output)
        print(f"Plot saved as {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()