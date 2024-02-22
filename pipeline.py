import os
import pandas as pd
import shutil
from pylabel import importer
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

    def coco_to_yolo_converter(self, annots_json, path_to_images, name):
        """
        Convert COCO format annotations to YOLO format.

        Parameters:
        - annots_json (str): The path to the COCO format annotations JSON file.
        - path_to_images (str): The path to the directory containing images.
        - name (str): The name of the dataset.

        Returns:
        - Dataset: The dataset in YOLO format.
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
        - Dataset: The prepared dataset.
        """
        print("\n")
        print(f"Preprocessing {user}-{suffix}...")

        # Convert COCO to YOLO annotation format
        annots_dataset = self.coco_to_yolo_converter(path_to_annots_json, path_to_images, name="annots_coco")

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

    def combine_datasets(self, datasets):
        """
        Combine multiple datasets into a single dataset.

        Parameters:
        - datasets (list): A list of Dataset objects to be combined.

        Returns:
        - Dataset: The combined dataset.
        """
        combined_df = pd.concat([dataset.df for dataset in datasets], axis=0)
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
    home = os.getcwd()
    destination_path = f"{home}/processed_yolo"
    temp_path = f"{home}/temp"

    # Check if the temp folder already exists
    if not os.path.exists(temp_path):
        # Create the temp folder
        print("Creating temp folder...")
        os.makedirs(temp_path)
        print("Temp folder created.")

        # Copy the contents of raw_data to temp folder
        print("Copying raw_data to temp folder...")
        shutil.copytree(f"{home}/raw_data", temp_path)
        print("Copying completed.")
    else:
        print("Temp folder already exists. Skipping creation.")

    processor = CocoDatasetProcessor(home)

    # Current progress:
    users = ['user_1', 'user_4']
    suffixes = ['pre']

    # If everything complete:
    # users = ['user_1', 'user_2', 'user_3', 'user_4']
    # suffixes = ['pre', 'post']

    datasets = []

    for user in users:
        for suffix in suffixes:
            image_directory = f"{home}/temp/{suffix}_event/{user}"
            processor.rename_images_with_suffix(image_directory, suffix)
            path_to_annots_json = f"{home}/temp/{suffix}_event/{user}-{suffix}.json"
            path_to_images = image_directory
            print("Path Annots:", path_to_annots_json)
            print("Path Images:", path_to_images)

            dataset = processor.prepare_coco_dataset(path_to_annots_json, path_to_images, user, suffix)

            # Modify img_filename column to add suffix in front of each filename
            dataset.df['img_filename'] = suffix + '_' + dataset.df['img_filename']
            
            datasets.append(dataset)

    # Combine multiple input datasets
    processed_dataset = processor.combine_datasets(datasets)

    # Split into Train and Test
    processed_dataset.splitter.GroupShuffleSplit(train_pct=0.8, val_pct=0, test_pct=0.2)

    # Remove NaN Class
    processed_dataset.df = processed_dataset.df.dropna(subset=['cat_name'])

    # Statistics
    print("\n\n")
    print(f"Number of images: {processed_dataset.analyze.num_images}")
    print(f"Number of classes: {processed_dataset.analyze.num_classes}")
    print(f"Classes:{processed_dataset.analyze.classes}")
    print(f"Class counts:\n{processed_dataset.analyze.class_counts}")
    print("\n\n")

    # Export for YOLO
    print("Exporting to YOLO format...")
    processed_dataset.export.ExportToYoloV5(output_path=f'{destination_path}/labels',
                                            yaml_file='dataset.yaml',
                                            cat_id_index = 0,
                                            copy_images=True,
                                            use_splits=True)
    
    # Remove the temporary directory
    shutil.rmtree(temp_path)
