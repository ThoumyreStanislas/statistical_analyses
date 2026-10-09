import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import argparse

def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('image',
                   help='Path of the volume data for timepoint 1')

    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Charger l'image NIfTI de freewater
    # Remplacez par votre fichier
    img = nib.load(args.image)
    data = img.get_fdata()

    # Aplatir les données et filtrer les valeurs valides (dans [0,1])
    data_flat = data.flatten()
    data_flat = data_flat[(data_flat >= 0) & (data_flat <= 1)]  # Filtrer valeurs aberrantes si besoin

    # Tracer l'histogramme
    plt.figure(figsize=(8, 6))
    plt.hist(data_flat, bins=200, range=(0, 1), color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel('Valeur de freewater (0-1)')
    plt.ylabel('Nombre de voxels')
    plt.title('Histogramme des valeurs de Freewater')
    plt.yscale('log')  # Échelle logarithmique pour mieux voir la distribution

    plt.savefig("histogram.png", dpi=300, bbox_inches='tight')
    plt.show()
if __name__ == "__main__":
    main()