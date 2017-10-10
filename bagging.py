from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import re
import numpy as np
import sys
import os

'''
Function: Use majority_voter.py as a reference, Bagging loads many models (dataset and model as independent as possible)
		  Use majority vote by the many models to predict labels of test data
Parameter: a list of model paths in result folder
Output: accuracy, image labels, etc
Return value: None
'''
def bagging(model_weight_paths):
	# parameters
	img_width, img_height = 128, 128

	# model_file_path   = sys.argv[1]
	# weights_file_path = sys.argv[2]
	# test_data_path    = sys.argv[3]

	loaded_model = []
	folder_name = ''
	# load models in the list of model_weight_paths
	for i in range(len(model_weight_paths)):
		# append folder name
		folder_name = folder_name + os.path.basename(os.path.normpath(model_weight_paths[i]))

		json_path = os.path.join(model_weight_paths[i], "model.json")
		h5_path = os.path.join(model_weight_paths[i], "weights.h5")
		if (not os.path.isfile(json_path)) or (not os.path.isfile(h5_path)):
			print('cannot find model.json or weights.h5 file')
			return
		# load json and reconstruct NN model
		json_file = open(json_path, 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model.append(model_from_json(loaded_model_json))
		print(model_weight_paths[i], "Model loaded from json file")

		# load weights into new model
		print('h5path:', h5_path)
		loaded_model[i].load_weights(h5_path)
		print(model_weight_paths[i], "Weights loaded from h5 file")

	# make a folder named 'folder_name' to store result files
	if not os.path.exists(folder_name):
    	os.makedirs(folder_name)

	# test model on train & test data of dataset 1~3
	for dataset in range(3):
		# parse directory of dataset
		if dataset == 0:
			dataset_name = 'first_dataset'
		elif dataset == 1:
			dataset_name = 'second_dataset'
		else:
			dataset_name = 'third_dataset'
		for train in range(2):

			if train == 1:
				train_name = 'train'
			else:
				train_name = 'test'

			data_folder = '/home/ubuntu/cancer-prediction/deep_learning/datasets'
			test_data_path = os.path.join(data_folder, dataset_name, train_name)
			# generate test data
			test_datagen = ImageDataGenerator(rescale=1./255)
			test_generator = test_datagen.flow_from_directory(
							test_data_path,
							target_size=(img_width, img_height),
							batch_size=32,  # batch_size number of test data are feed in each time
							class_mode='binary',
							shuffle=False)

			# batch_size * steps should have enough threads to load all test data
			preds = []
			for i in loaded_model:
				preds.append(i.predict_generator(test_generator, steps=np.ceil(len(test_generator.filenames[:])/32.0)))

			# print the large volume results to a folder		
			# change from printing to console to printing to files
			if train == 1:
				output = open(os.path.join(folder_name, 'bagging_dataset'+str(dataset+1)+'_train.txt'), 'w')
			else:
				output = open(os.path.join(folder_name, 'bagging_dataset'+str(dataset+1)+'_test.txt'), 'w')
			orig_stdout = sys.stdout
			sys.stdout = output

			print("Bagging result of the following models:")
			print(model_weight_paths)
			print('')

			# store result into a dictionary
			# keys: image name, each image has multiple patches
			# value: [# of 0 class vote, # of 1 class vote, real label]

			image_dict = {}
			image_dict_list = [dict() for j in range(len(preds))]

			correct_patch = 0
			for i in range(len(test_generator.filenames[:])):
				file_name = test_generator.filenames[i].split('/')[-1]
				real_label = test_generator.classes[i]
				
				# get name of the image that current patch belongs to
				temp = re.split('_|-', file_name)
				if 'PR' in temp[0]:
					image_name = temp[1]
				elif 'TMA' in temp[0]:
					image_name = temp[2]
				else:
					image_name = temp[3]

				if image_name not in image_dict:
					image_dict[image_name] = [0, 0, real_label]

				# predict current patch
				for j in range(len(preds)):

					#if probability[0] < 0.5: # for bangqi's code
					probability = preds[j][i]
					if len(probability) == 1: # for bangqi's results
						if probability[0] < 0.5:
							image_dict_list[j][image_name][0] += 1
							curr_patch_pred = 0
						else:
							image_dict_list[j][image_name][1] += 1
							curr_patch_pred = 1
					else: # for resnet results:
						if probability[0] > probability[1]:
							image_dict[j][image_name][0] += 1
							curr_patch_pred = 0
						else:
							image_dict[j][image_name][1] += 1
							curr_patch_pred = 1


					# accumulator for single patch
					if curr_patch_pred == real_label:
						correct_patch += 1

			print("image dictionary is:")
			print(image_dict)

			print("probabilities for 2 classes are:")
			# vote by majority

			


			images = image_dict_list[0].keys()
			correct_image = 0
			correct_bagging = 0
			# for each image
			for image_name in images:
				negative = 0
				positive = 0
				true_label = image_dict_list[0][image_name][2]
				# for each model's prediction
				for j in range(len(preds)):
					result = image_dict_list[j][image_name]
					
					print(result)
					if result[0] > result[1]: # the current j-th model votes for negative
						negative += 1
						majority = 0
					else: # the current j-th model votes for postive
						positive += 1
						majority = 1
					# majority is the predict given by majority vote
					# now compare it with real label, result[2]
					if majority == result[2]:
						correct_image += 1

				# decide the bagging result of models
				if (negative > positive and true_label == 0) or (positive > negative and true_label == 1):
					correct_bagging += 1

			patch_accuracy = correct_patch*1.0/len(model_weight_paths)/len(test_generator.filenames[:])
			print("Average Patch-wide accuracy is: {}".format(patch_accuracy))
			image_accuarcy = correct_image*1.0/len(model_weight_paths)/len(images)
			print("Average Image-wide accuracy given by majority vote is: {}".format(image_accuarcy))

			bagging_acc = correct_bagging / len(images)
			print("Bagging accuracy given by input models is: {}".format(bagging_acc))
			sys.stdout = orig_stdout
			output.close()

model1_path = '/bangqi_results/bangqi_6_9_1_44pm_1/'
model2_path = 'results/8_18_resnet18_l2=1e-4/'
model3_path = 'results/10_5_resnet50/'

bagging([model1_path, model2_path, model3_path])