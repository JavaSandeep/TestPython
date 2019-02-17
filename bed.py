import numpy as np
import pandas as pd

layers = [
	"layer1":{
		"number_of_inputs":3,
		"nuerons_count":2
	},
	"layer2":{
		"number_of_inputs":2,
		"nuerons_count":1
	}
]

def feedforward(input_matrix, weights, bais):
	"""
	Function takes input column for a column of neurons
	Take weights assoicated to them
	Calculate output for next column of neurons
	y = sigmoid(W.X + b)
	"""
	multiplied_mat = np.matmul(weights, input_matrix) + bais
	result_mat = sigmoid(bias_added)
	return result_mat

def sigmoid(x):
	"""
	Mathematically:
	A = W.X + b
			1
	y = ----------
		 1 + e^-A
	"""
	return 1/(1+np.exp(-x))

def pd_of_sigmoid(x):
	"""
	Mathematically, Partial derivatives:
	 dy    
	---- = y * (1 - y) <--- Since, y is an output. It's partial derivative shows it depends on itself, but not input.
	 dA
	"""
	ret_list = []
	for elem in x:
		ret_list.append([elem[0]*(1-elem[0])])
	ret_list = np.asarray(ret_list, dtype=np.float32)
	return ret_list

def initialize_weights():
	layer_weights = []
	for each_layer in layers:
		intial_weights = np.random.randn(each_layer.get("nuerons_count"), each_layer.get("number_of_inputs"))
		layer_weights.append(intial_weights) 
	return layer_weights

def initialize_bias():
	layer_bias = []
	for each_layer in layers:
		intial_bias = np.random.randn(each_layer.get("nuerons_count"), 1)
		layer_bias.append(intial_weights)
	return layer_bias	

def get_training_set():
	file_to_read = "D:\\AI\\Bed_nn\\bed.csv"

	input_cols = ["Height", "Width", "Thickness"]
	output_cols = ["output"]

	dataframe = pd.read_csv(file_to_read)
	dataframe["output"] = np.where(
		dataframe["Bed"] == "Small Bed",
		0,
		1
	)
	# SPLIT THE DATAFRAME HERE
	return dataframe[input_cols].as_matrix(), dataframe[output_cols].as_matrix()

def train_neural_network():
	__weights = initialize_weights()
	__bias = initialize_bais()
	__training_set, result = get_training_set()

	__number_of_samples = len(__training_set)
	__number_of_iteration = 50

	# STOCHASTIC GRADIENT SAMPLE
	__sample_size = 20

	__start_idx = 0
	__end_idx = __sample_size - 1

	while (__end_idx < __number_of_samples):
		# BELOW HERE STARTS STOCHASTIC TRAIING
		# NEURAL NETWORK IS TRAINED OVER A SUBSET OF TRAINING SET
		# ERROR IN EACH PASS IS CALCULATED
		# delta. FOR WEIGHTS AND BAIS ARE CALCULATE FOR EACH PASS
		# AGGREGATED delta is calculated for subset of training set
		# delta is applied to wieghts and bais
		deltas_W = []
		deltas_b = []
		for i in range(__number_of_iteration):
			select_idx = np.random.randint(__start_idx, __end_idx)
			input_values = __training_set[i]
			desired_output = result[i]

			# FORWARD FEED LOOP
			output_vals = []
			input_vals = []
			for idx, each_layer in layers:
				input_vals.append(input_values)
				output_values = feedforward(input_values, __weights[idx], __bias[idx])
				output_vals.append(output_values)
				input_values = output_values

			# FEED FORWARDING PROVIDES US WITH FOR A SINGLE INPUT EXAMPLE
			# 1. INPUT AT EACH COLUMN OF N.N.
			# 2. OUTPUT AT EACH COLUMN OF N.N.
			# 3. OUTPUT OF NEURAL NETWORK FOR A TRAINING SET
			# EACH OF THEM IS REQUIRED TO BACK-PROPAGATE ERROR


			# BACKPROPAGATE LOGIC
			temp_deltas_W = []
			temp_deltas_b = []
			for idx, each_layer in layers:
				



		__start_idx += __sample_size
		__end_idx += __sample_size



