import os
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds, reproject, Resampling
import numpy as np


# Function to crop patches from a new SAR GeoTIFF image using original optical patch names from a folder
def crop_optical_using_sar_patches(whole_optical_path, sar_patch_folder, output_dir, patch_size=1024, keyword=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all SAR patches
    sar_patch_paths = [
        os.path.join(sar_patch_folder, f)
        for f in os.listdir(sar_patch_folder)
        if f.endswith('.tif') and (keyword in f if keyword else True)
    ]

    with rasterio.open(whole_optical_path) as optical_src:
        optical_crs = optical_src.crs
        optical_count = optical_src.count
        patch_count = 0

        for sar_patch_path in sar_patch_paths:
            sar_patch_name = os.path.basename(sar_patch_path)
            optical_patch_name = sar_patch_name.replace("_post_disaster", "_pre_disaster")
            patch_output_path = os.path.join(output_dir, optical_patch_name)

            with rasterio.open(sar_patch_path) as sar_patch:
                sar_bounds = sar_patch.bounds
                sar_crs = sar_patch.crs

                # Transform SAR bounds to optical image CRS if needed
                if sar_crs != optical_crs:
                    transformed_bounds = transform_bounds(sar_crs, optical_crs,
                                                          sar_bounds.left, sar_bounds.bottom,
                                                          sar_bounds.right, sar_bounds.top)
                else:
                    transformed_bounds = sar_bounds

                # Convert transformed bounds to pixel coordinates in optical image
                col_start, row_start = ~optical_src.transform * (transformed_bounds[0], transformed_bounds[3])
                col_stop, row_stop = ~optical_src.transform * (transformed_bounds[2], transformed_bounds[1])

                col_start, row_start = int(col_start), int(row_start)
                col_stop, row_stop = int(col_stop), int(row_stop)

                # Check bounds validity
                if (-512 <= col_start < optical_src.width+1 and -512 <= row_start < optical_src.height+1 and
                        col_stop <= optical_src.width + 1025 and row_stop <= optical_src.height + 1025):

                    window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
                    patch_data = optical_src.read(window=window)

                    # Resample to fixed patch size
                    resampled_patch = np.zeros((optical_count, patch_size, patch_size), dtype=patch_data.dtype)
                    reproject(
                        source=patch_data,
                        destination=resampled_patch,
                        src_transform=optical_src.window_transform(window),
                        src_crs=optical_crs,
                        dst_transform=rasterio.transform.from_bounds(*transformed_bounds, patch_size, patch_size),
                        dst_crs=optical_crs,
                        resampling=Resampling.lanczos
                    )

                    profile = optical_src.profile
                    profile.update({
                        'height': patch_size,
                        'width': patch_size,
                        'transform': rasterio.transform.from_bounds(*transformed_bounds, patch_size, patch_size),
                        'count': optical_count,
                        'compress': None
                    })

                    with rasterio.open(patch_output_path, 'w', **profile) as dst:
                        dst.write(resampled_patch)

                    patch_count += 1
                else:
                    print(f"[!] Patch {sar_patch_name} is out of bounds and skipped.")

    print(f"[✔] Generated {patch_count} optical patches and saved in {output_dir}")

# Example usage:

if __name__ == '__main__':
    # Define the absolute path to the new SAR GeoTIFF image
    whole_optical_scene_path = r'D:\Research\Dataset\BRIGHT\Myanmar_Hurricane_GoogleEarth_AOI01\pre_disaster_modified.tif'

    # Define the folder containing the original GeoTIFF optical patches
    sar_patch_folder = r'D:\Research\Dataset\BRIGHT\BRIGHT_final\post-event\post-event'

    # Define the path to the directory where the new SAR patches will be saved
    output_dir = 'D:\Research\Dataset\BRIGHT\github_test'

    crop_event = 'myanmar-hurricane'

    # Generate the SAR patches with a keyword filter
    crop_optical_using_sar_patches(whole_optical_scene_path, sar_patch_folder, output_dir, patch_size=1024,
                                             keyword=crop_event)
