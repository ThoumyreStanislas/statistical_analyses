import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Compare volumes of selected bundle across methods and timepoints.",
        formatter_class=argparse.RawTextHelpFormatter)
    
    p.add_argument('data', nargs='+', help='JSON files with volume for each timepoint and method')
    p.add_argument('--bundle', type=str, required=True, help='The bundle to compare (e.g., AF, IFOF, SLF)')
    p.add_argument('--output', type=str, help='File path to save the boxplot image (e.g., output.png)')
    
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    
    data = []

    # Parcours des fichiers JSON fournis en arguments
    for file_path in args.data:
        filename = os.path.basename(file_path)
        # Extraction de la méthode, timepoint et sujet depuis le nom de fichier et le chemin
        method = file_path.split('/')[0]  # Méthode
        timepoint = filename.split('_')[4]  # Timepoint (e.g., T1, T2, T3)
        subject = '_'.join(filename.split('_')[:5])  # Sujet
        

        
        # Chargement des données du fichier JSON
        with open(file_path, 'r') as file:
            json_data = json.load(file)

            bundle_volume = json_data.get(subject, {}).get(args.bundle, {}).get("volume", None)
             
            if bundle_volume is not None:
                # Ajout des données dans la liste
                data.append({
                    'Subject': subject,
                    'Timepoint': timepoint,
                    'Method': method,
                    'Bundle_Volume': bundle_volume
                })

    # Création du DataFrame
    df = pd.DataFrame(data)


    # Création du boxplot avec Seaborn
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Timepoint', y='Bundle_Volume', hue='Method')
    plt.title(f'Comparaison des volumes du faisceau {args.bundle} entre méthodes et timepoints')
    plt.xlabel('Timepoint')
    plt.ylabel(f'Volume de {args.bundle}')
    plt.legend(title='Method')
    
    # Sauvegarde ou affichage du plot
    if args.output:
        plt.savefig(args.output)
        print(f"Plot saved as {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()