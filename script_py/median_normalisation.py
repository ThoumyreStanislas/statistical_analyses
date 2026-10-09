import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import argparse

def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)


    p.add_argument("--input", type=str, required=True,
                    help="Chemin de l'image NIfTI à normaliser")
    p.add_argument("--mask", type=str, required=False,
                    help="Chemin du masque NIfTI")
    p.add_argument("--output", type=str, required=True,
                    help="Chemin de sauvegarde de l'image normalisée")

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Charger les images
    img = nib.load(args.input)

    # Extraire les données sous forme de tableaux numpy
    data = img.get_fdata()

    if args.mask:
        mask = nib.load(args.mask)
        mask_data = mask.get_fdata()
        masked_values = data[mask_data > 0]
        median_value = np.median(masked_values)
        normalized_data = np.zeros_like(data)
        normalized_data[mask_data > 0] = masked_data / median_value

    else:
        median_value = np.median(data[data > 0])
        normalized_data = data / median_value


    # Sauvegarder en NIfTI
    normalized_img = nib.Nifti1Image(normalized_data, img.affine, img.header)
    nib.save(normalized_img, args.output)

if __name__ == "__main__":
    main()