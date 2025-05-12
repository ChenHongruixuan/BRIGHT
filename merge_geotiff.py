import os
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import matplotlib.pyplot as plt


# Function to mosaic multiple GeoTIFF images
def mosaic_geotiffs(input_file_paths, output_path):
    # List to store open rasterio dataset objects
    src_files_to_mosaic = []

    # Open all input files and append to the list
    for fp in input_file_paths:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)

    # Merge function to combine the images
    mosaic, out_trans = merge(src_files_to_mosaic)

    # Copy the metadata of one of the files
    out_meta = src.meta.copy()

    # Update the metadata with the new dimensions, transform (affine), and CRS
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "compression": 'DEFLATE'
    })

    # Write the merged mosaic to a new file
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    print(f"Mosaic created and saved at: {output_path}")

    # # Optional: Display the mosaic
    # show(mosaic, cmap="terrain")
    # plt.show()


if __name__ == '__main__':
    #
    input_files = [
        r"path of the subarea image 1 (e.g., Mexico_Hurricane_AOI01A_pre_disaster.tif)",
        r"path of the subarea image 2 (e.g., Mexico_Hurricane_AOI01B_pre_disaster.tif)",
    ]

    # Define the output path for the mosaic
    output_file = r"saved path of the merged geotiff"

    # Call the function to mosaic the images
    mosaic_geotiffs(input_files, output_file)
