"""
Pipeline_old.py

<Broken>
"""

import os
import pandas as pd
import shutil
import warnings
from pylabel import importer
from collections import Counter
from pylabel.dataset import Dataset

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

    def prepare_coco_dataset(self, path_to_annots_json, path_to_images, user, suffix):
        """
        Prepare a COCO dataset for further processing.

        Parameters:
        - path_to_annots_json (str): The path to the COCO format annotations JSON file.
        - path_to_images (str): The path to the directory containing images.
        - user (str): The user identifier.
        - suffix (str): The suffix indicating pre/post event.

        Returns:
        - Dataset: The prepared Pylabel dataset.
        """
        print("\n")
        print(f"Preprocessing {user}-{suffix}...")

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

    def combine_datasets(self, datasets, background_dataset):
        """
        Combine multiple PyLabel datasets into a single dataset.

        Parameters:
        - datasets (list): A list of Dataset objects to be combined.

        Returns:
        - Dataset: The combined Pylabel dataset.
        """
        combined_df = pd.concat([dataset.df for dataset in datasets], axis=0)
        print(f"Number of images before dropna: {Dataset(combined_df).analyze.num_images}")
        combined_df = combined_df.dropna(subset=['cat_name'])
        print(f"Number of images after dropna: {Dataset(combined_df).analyze.num_images}")
        combined_df = pd.concat([background_dataset, combined_df], axis=0)
        print(f"Number of images after adding background images: {Dataset(combined_df).analyze.num_images}")
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

    def check_class_fraction(self, dataset):
        """
        Check class fraction for the train and test splits of the given dataset.

        Parameters:
        - dataset (Dataset): The dataset to analyze.
        """
        splits = dataset.df.groupby('split')

        for split, df in splits:
            classes = Counter(df['cat_name'])
            total_samples = len(df)
            print(f"\nClass fraction for {split} dataset:")
            for class_name, count in classes.items():
                fraction = (count / len(df))*100
                print(f"{class_name}: {fraction:.4f}% ({count}/{total_samples} samples)")

if __name__ == "__main__":
    print("--------------------------\nPipeline Script \nLast Update: 4/3/2024 \n-------------------------- \n\n")
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
            dataset = processor.prepare_coco_dataset(path_to_annots_json, path_to_images, user, suffix)
    
            # Modify img_filename column to add suffix in front of each filename
            dataset.df['img_filename'] = suffix + '_' + dataset.df['img_filename']
            
            datasets.append(dataset)
            
    # Process additional datasets excluding annotations (Background)
    print("--------------------------\n( 3 ) Adding Optional Dataset\n--------------------------\n")
    print("Adding Background Dataset...")
    path_to_bg = f"{home}/temp/background/background"  
    path_to_bg_annots_json = f"{home}/temp/background/background_img.json" 
    bg_dataset = importer.ImportCoco(path_to_bg_annots_json, path_to_images=path_to_bg, name="annots_bg") 
    bg_dataset = bg_dataset.df[bg_dataset.df['img_filename'] != 'base.jpg']     
    
    # Combine multiple input datasets
    print("--------------------------\n( 4 ) Creating Dataset\n--------------------------\n")
    # TO-DO - Add synthetic images
    processed_dataset = processor.combine_datasets(datasets, bg_dataset)

    # Split into Train and Test
    print("--------------------------\n( 5 ) Splitting Dataset\n--------------------------\n")
    processed_dataset.splitter.StratifiedGroupShuffleSplit(train_pct=0.8, val_pct=0, test_pct=0.2, batch_size=1)

    # Check class fraction
    print("--------------------------\n( 6 ) Checking Splits Statistics\n--------------------------\n")
    processor.check_class_fraction(processed_dataset)

    # Statistics
    print("\n\n")
    print("--------------------------\n( 7 ) Dataset Statistics\n--------------------------\n")
    print(f"Number of images: {processed_dataset.analyze.num_images}")
    print(f"Number of classes: {processed_dataset.analyze.num_classes}")
    print(f"Classes:{processed_dataset.analyze.classes}")
    print(f"Class counts:\n{processed_dataset.analyze.class_counts}")
    print(f"Path to annotations:\n{processed_dataset.path_to_annotations}")
    print("\n\n")

    # Export for YOLO
    print("--------------------------\n( 8 ) COCO to YOLO\n--------------------------\n")
    print("Exporting to YOLO format...")
    processed_dataset.export.ExportToYoloV5(output_path=f'{destination_path}/labels',
                                            yaml_file='dataset.yaml',
                                            cat_id_index = 0,
                                            copy_images=True,
                                            use_splits=True)
    
    # Remove the temporary directory
    shutil.rmtree(temp_path)
    print("--------------------------\nPipeline Script End\n--------------------------\n")