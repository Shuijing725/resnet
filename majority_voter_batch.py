import majority_voter
import glob
import os

# CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
# RESULTS_FOLDER = 'results'
results_folder = 'results'
data_folder = '/home/shuijing/cancer-prediction/deep_learning/datasets'
def create_folder(output_folder_name):
    #Create a new folder with specified name
    try:
        os.makedirs(output_folder_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


if __name__ == "__main__":

	# Subroutine to find subfolders
    subfolders = []
    # for root, dirs, files in os.walk(results_folder, topdown=False):
    #     for name in dirs:
    #         subfolders.append(name)

    subfolders = ['10_3_resnet50', '10_4_resnet50', '10_4_resnet50_1', '10_5_resnet50']
    # print(subfolders)
    # # For each subfolder
    for subfolder in subfolders:
    #     # test model on train & test data of dataset 1~3
    #     for dataset in range(3):
    #         for train in range(1):
    #             # parse directory of dataset
    #             if dataset == 0:
    #                 dataset_name = 'first_dataset'
    #             elif dataset == 1:
    #                 dataset_name = 'second_dataset'
    #             else:
    #                 dataset_name = 'third_dataset'

    #             if train == 1:
    #                 train_name = 'train'
    #             else:
    #                 train_name = 'test'
                # print(os.path.join(results_folder, subfolder))
        majority_voter.vote(os.path.join(results_folder, subfolder))
 #        df = pd.read_csv(SCRIPT_LOCATION + "/" + folder_input + subfolder + "/" + "labels.csv")
	# model_file_path, weights_file_path, test_data_path, patch_accuracy, image_accuracy = majority_vote(model_file_path, weights_file_path, test_data_path)

