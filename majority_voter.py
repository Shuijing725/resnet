from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import re
import numpy as np
import sys
import os

# dataset_num = 1, 2, or 3
# train = 1 : predict on training data, train =0: predict on testing data
def vote(model_weight_path):
	# parameters
	img_width, img_height = 128, 128

	# model_file_path   = sys.argv[1]
	# weights_file_path = sys.argv[2]
	# test_data_path    = sys.argv[3]

	# check whether both json and h5 file exist
	json_path = os.path.join(model_weight_path, "model.json")
	h5_path = os.path.join(model_weight_path, "weights.h5")
	if (not os.path.isfile(json_path)) or (not os.path.isfile(h5_path)):
		return
	# load json and reconstruct NN model
	json_file = open(json_path, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	print(model_weight_path)
	print("Model loaded from json file")

	# load weights into new model
	print('h5path:', h5_path)
	loaded_model.load_weights(h5_path)
	print("Weights loaded from h5 file")

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
			preds = loaded_model.predict_generator(test_generator, steps=np.ceil(len(test_generator.filenames[:])/32.0))

			# change from printing to console to printing to files
			if train == 1:
				output = open(os.path.join(model_weight_path, 'majority_vote_dataset'+str(dataset+1)+'_train.txt'), 'w')
			else:
				output = open(os.path.join(model_weight_path, 'majority_vote_dataset'+str(dataset+1)+'_test.txt'), 'w')
			orig_stdout = sys.stdout
			sys.stdout = output

			# store result into a dictionary
			# keys: image name, each image has multiple patches
			# value: [# of 0 class vote, # of 1 class vote, real label]
			image_dict = {}
			correct_patch = 0
			for i in range(len(test_generator.filenames[:])):
				file_name = test_generator.filenames[i].split('/')[-1]
				real_label = test_generator.classes[i]
				probability = preds[i]
				print(probability)

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
				#if probability[0] < 0.5: # for bangqi's code
				if probability[0] > probability[1]:
					image_dict[image_name][0] += 1
					curr_patch_pred = 0
				else:
					image_dict[image_name][1] += 1
					curr_patch_pred = 1

				# accumulator for single patch
				if curr_patch_pred == real_label:
					correct_patch += 1

			print("image dictionary is:")
			print(image_dict)

			print("probabilities for 2 classes are:")
			# vote by majority
			images = image_dict.keys()
			correct_image = 0
			for image_name in images:
				result = image_dict[image_name]
				
				print(result)
				if result[0] > result[1]:
					majority = 0
				else:
					majority = 1

				# majority is the predict given by majority vote
				# now compare it with real label, result[2]
				if majority == result[2]:
					correct_image += 1

			patch_accuracy = correct_patch*1.0/len(test_generator.filenames[:])
			print("Patch-wide accuracy is: {}".format(patch_accuracy))
			image_accuarcy = correct_image*1.0/len(images)
			print("Image-wide accuracy given by majority vote is: {}".format(image_accuarcy))

			sys.stdout = orig_stdout
			output.close()