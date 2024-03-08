import os
import pandas as pd
import shutil
import warnings
from pylabel import importer
from collections import Counter
from pylabel.dataset import Dataset
import argparse 

class CocoDatasetProcessor:
    """
    Class to process COCO datasets and perform various operations.
    """

    def __init__(self, home):
        """
        Initialize the CocoDatasetProcessor object.

        Parameters:
        - home (str): The path to the home directory.
        """
        self.home = home

    def rename_images_with_suffix(self, directory, suffix):
        """
        Rename images in a directory by adding a suffix to their filenames.

        Parameters:
        - directory (str): The path to the directory containing images.
        - suffix (str): The suffix to be added to the filenames.
        """
        for filename in os.listdir(directory):
            if filename.endswith(".jpg"):
                old_path = os.path.join(directory, filename)
                new_filename = f"{suffix}_{filename}"
                new_path = os.path.join(directory, new_filename)
                os.rename(old_path, new_path)

    def coco_importer(self, annots_json, path_to_images, name):
        """
        Create a COCO PyLabel dataset.

        Parameters:
        - annots_json (str): The path to the COCO format annotations JSON file.
        - path_to_images (str): The path to the directory containing images.
        - name (str): The name of the dataset.

        Returns:
        - Dataset: The dataset in COCO PyLabel format.
        """
        dataset = importer.ImportCoco(annots_json, path_to_images=path_to_images, name=name)
        return dataset

    def prepare_coco_dataset(self, path_to_annots_json, path_to_images):
        """
        Prepare a COCO dataset for further processing.

        Parameters:
        - path_to_annots_json (str): The path to the COCO format annotations JSON file.
        - path_to_images (str): The path to the directory containing images.

        Returns:
        - Dataset: The prepared Pylabel dataset.
        """
        # Create a COCO PyLabel dataset
        annots_dataset = self.coco_importer(path_to_annots_json, path_to_images, name="annots_coco")

        # Mapping of cat_id to cat_name
        cat_name_mapping = {
            '1': 'undamagedresidentialbuilding',
            '2': 'damagedresidentialbuilding',
            '3': 'undamagedcommercialbuilding',
            '4': 'damagedcommercialbuilding'
        }

        # Append cat_name based on cat_id
        annots_dataset.df['cat_name'] = annots_dataset.df['cat_id'].map(cat_name_mapping)

        print(f"Number of images: {annots_dataset.analyze.num_images}")
        print(f"Number of classes: {annots_dataset.analyze.num_classes}")
        print(f"Classes:{annots_dataset.analyze.classes}")
        print(f"Class counts:\n{annots_dataset.analyze.class_counts}")
        print(f"Path to annotations:\n{annots_dataset.path_to_annotations}")

        return annots_dataset

    def combine_datasets(self, datasets, synthetic_datasets, include_synthetic=False):
        """
        Combine multiple PyLabel datasets into a single dataset.

        Parameters:
        - datasets (list): A list of Dataset objects to be combined.
        - synthetic_datasets (list): A list of synthetic Dataset objects to be combined.
        - include_synthetic (bool): Whether to include synthetic datasets. Default is False.

        Returns:
        - Dataset: The combined Pylabel dataset.
        """
        combined_dfs = []

        # Concatenate real dataset
        for dataset in datasets:
            combined_dfs.append(dataset.df)

        print(f"Number of labels before combining synthetic datasets: {pd.concat(combined_dfs).shape[0]}")

        # Concatenate synthetic dataset if include_synthetic is True
        if include_synthetic:
            for synthetic_dataset in synthetic_datasets:
                combined_dfs.append(synthetic_dataset.df)

        print(f"Number of labels after combining synthetic datasets: {pd.concat(combined_dfs).shape[0]}")

        # Concatenate all dataframes
        combined_df = pd.concat(combined_dfs, axis=0)

        print(f"Number of labels before dropna: {combined_df.shape[0]}")

        # Drop rows with missing 'cat_name'
        combined_df = combined_df.dropna(subset=['cat_name'])

        print(f"Number of labels after dropna: {combined_df.shape[0]}")

        # Sort and reset index
        combined_df.sort_values(by='img_filename', inplace=True)
        combined_df.reset_index(drop=True, inplace=True)

        # Initialize img_id counter and previous filename
        current_img_id = 0
        previous_filename = None

        # Iterate through rows and update img_id based on filename changes
        for index, row in combined_df.iterrows():
            current_filename = row['img_filename']
            if current_filename != previous_filename:
                current_img_id += 1
            combined_df.at[index, 'img_id'] = current_img_id
            previous_filename = current_filename

        return Dataset(combined_df)


if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description='Process COCO datasets.')
    parser.add_argument('--include_synthetic', action='store_true', help='Include synthetic dataset')
    args = parser.parse_args()

    print("--------------------------\nPipeline Script \n(KFold Version) \nLast Update: 8/3/2024 \n-------------------------- \n\n")
    # Define the paths
    home = os.getcwd() # Make sure inside the home directory of repo
    destination_path = f"{home}/processed_yolo"
    temp_path = os.path.join(home, "temp")
    raw_data_path = os.path.join(home, "raw_data")

    # Copy the contents of raw_data to temp folder
    print("--------------------------\n( 1 ) Creating 'temp' Folder\n--------------------------\n")
    print("Copying raw_data to temp folder...")
    shutil.copytree(raw_data_path, temp_path, dirs_exist_ok=True)
    print("Copying completed.")

    # Suppress warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    processor = CocoDatasetProcessor(home)
    
    # Create array of JSON
    print("--------------------------\n( 2 ) Preparing PyLabel COCO Dataset\n--------------------------\n")
    users = ['user_1', 'user_2', 'user_3', 'user_4']
    suffixes = ['pre', 'post']
    
    datasets = []
    
    for user in users:
        for suffix in suffixes:
            image_directory = f"{home}/temp/{suffix}_event/{user}"
            
            # Check if the directory exists
            if not os.path.exists(image_directory):
                print(f"Directory not found: {image_directory}. Skipping...")
                continue
                
            processor.rename_images_with_suffix(image_directory, suffix)
            path_to_annots_json = f"{home}/temp/{suffix}_event/{user}-{suffix}.json"

            # Check if the annotation file exists
            if not os.path.exists(path_to_annots_json):
                print(f"\n\n{user}-{suffix}.json Not Found.\n Skipping...\n\n")
                continue

            path_to_images = image_directory
            print("\n")
            print(f"Preprocessing {user}-{suffix}...")
            dataset = processor.prepare_coco_dataset(path_to_annots_json, path_to_images)
    
            # Modify img_filename column to add suffix in front of each filename
            dataset.df['img_filename'] = suffix + '_' + dataset.df['img_filename']
            
            datasets.append(dataset)

    # Process additional datasets excluding annotations (Background)
    print("--------------------------\n( 3 ) Adding Optional Dataset\n--------------------------\n")
    if args.include_synthetic:  # Check if --include_synthetic argument is provided
        print("Adding Synthetic Dataset...")
        synthetic_classes = ['damagedresidentialbuilding', 'damagedcommercialbuilding', 'undamagedcommercialbuilding']

        synthetic_dataset = []

        for syn_class in synthetic_classes:
            path_to_syn = f"{home}/temp/synthetic/{syn_class}"  
            path_to_syn_annots_json = f"{home}/temp/synthetic/synthetic_{syn_class}.json" 

            # Check if the annotation file exists
            if not os.path.exists(path_to_syn_annots_json):
                print(f"\n\nsynthetic_{syn_class}.json Not Found.\n Skipping...\n\n")
                continue

            # Check if the directory exists
            if not os.path.exists(image_directory):
                print(f"Directory not found: {image_directory}. Skipping...")
                continue   

            print(f"\n\nPreprocessing {syn_class} synthetic images...")
            syn_dataset = processor.prepare_coco_dataset(path_to_syn_annots_json, path_to_syn ) 

            synthetic_dataset.append(syn_dataset)
    
    # Combine multiple input datasets
    print("--------------------------\n( 4 ) Creating Dataset\n--------------------------\n")
    processed_dataset = processor.combine_datasets(datasets, synthetic_dataset if args.include_synthetic else [], include_synthetic=args.include_synthetic)

    # Statistics
    print("\n\n")
    print("--------------------------\n( 5 ) Dataset Statistics\n--------------------------\n")
    print(f"Number of images: {processed_dataset.analyze.num_images}")
    print(f"Number of classes: {processed_dataset.analyze.num_classes}")
    print(f"Classes:{processed_dataset.analyze.classes}")
    print(f"Class counts:\n{processed_dataset.analyze.class_counts}")
    print(f"Path to annotations:\n{processed_dataset.path_to_annotations}")
    print("\n\n")

    # Export for YOLO
    print("--------------------------\n( 6 ) COCO to YOLO\n--------------------------\n")
    print("Exporting to YOLO format...")
    processed_dataset.export.ExportToYoloV5(output_path=f'{destination_path}/labels',
                                            yaml_file='dataset.yaml',
                                            cat_id_index = 0,
                                            copy_images=True,
                                            use_splits=True)
    
    # Remove the temporary directory
    shutil.rmtree(temp_path)
    print("--------------------------\nPipeline Script End\n--------------------------\n")
