#include "neuron.h"
#include <time.h>
#include <math.h>
#include <iostream>

using namespace std;

MLP::MLP(int noLayers, int* layerSizes, double learnRate, double momentum) : learnRate_(learnRate), momentum_(momentum), noLayers_(noLayers) {
	srand(time(NULL));
	
	//allocate base memory
	layerSizes_ = new int[noLayers];
	outputs_ = new double*[noLayers];
	errors_ = new double*[noLayers];
	weights_ = new double**[noLayers];
	deltaWeights_ = new double**[noLayers];
	
	//allocate layer-dependant memory
	for (int i = 0; i < noLayers; i++) {
		layerSizes_[i] = layerSizes[i];
		outputs_[i] = new double[layerSizes[i]];
		
		if (i > 0) {
			errors_[i] = new double[layerSizes[i]];
			weights_[i] = new double*[layerSizes[i]];
			deltaWeights_[i] = new double*[layerSizes[i]];
			
			//allocate additional memory for weights_
			for (int j = 0; j < layerSizes[i]; j++) {
				//number of weights is equal to number of outputs of previous layer plus 1 (bias weight)
				int inputSize = layerSizes[i-1] + 1;
				weights_[i][j] = new double[inputSize];
				deltaWeights_[i][j] = new double[inputSize];
				
				//initialize weights in range [-1.0, 1.0]
				for (int k = 0; k < inputSize; k++) {
					weights_[i][j][k] = 2.0 * (double)rand() / (double)RAND_MAX - 1.0;
					deltaWeights_[i][j][k] = 0.0;
				}
			}
		}
	}
}

MLP::~MLP() {
	//deallocate memory
	// for (int i = 0; i < noLayers_; i++)
		// delete[] outputs_[i];
	// delete[] outputs_;
	
	// for (int i = 1; i < noLayers_; i++)
		// delete[] errors_[i];
	// delete[] errors_;
	
	// for (int i = 1; i < noLayers_; i++)
		// for (int j = 0; j < layerSizes_[i]; j++)
			// delete[] weights_[i][j];
	// for (int i = 1; i < noLayers_; i++)
		// delete[] weights_[i];
	// delete[] weights_;
	
	// for (int i = 1; i < noLayers_; i++)
		// for (int j = 0; j < layerSizes_[i]; j++)
			// delete[] deltaWeights_[i][j];
	// for (int i = 1; i < noLayers_; i++)
		// delete[] deltaWeights_[i];
	// delete[] deltaWeights_;
	
	// delete[] layerSizes_;
}

void MLP::backPropagate(double* inputs, double* answers) {
	//calculates outputs
	feedForward(inputs);
	
	//process output layers
	int lastLayerIndex = noLayers_ - 1;
	for (int i = 0; i < layerSizes_[lastLayerIndex]; i++) {
		//derivative of activation function; f'(x)
		double sigmoidDerivative = outputs_[lastLayerIndex][i] * (1.0 - outputs_[lastLayerIndex][i]);
		//how much output value differs from real value
		double deltaAnswer = answers[i] - outputs_[lastLayerIndex][i];
		errors_[lastLayerIndex][i] = sigmoidDerivative * deltaAnswer;
	}	

	//process every hidden layer
	for (int i = noLayers_-2; i > 0; i--) {
		//process every neuron in hidden layer
		for (int j = 0; j < layerSizes_[i]; j++) {
			double value = 0.0;
			for (int k = 0; k < layerSizes_[i+1]; k++)
				value += errors_[i+1][k] * weights_[i+1][k][j];
			
			//apply sigmoid derivative
			errors_[i][j] = outputs_[i][j] * (1.0 - outputs_[i][j]) * value;
		}
	}
	
	//adjust weights based on momentum and previous weights
	for (int i = 1; i < noLayers_; i++) {
		for (int j = 0; j < layerSizes_[i]; j++) {
			//adjust normal weights
			for (int k = 0; k < layerSizes_[i-1]; k++)
				weights_[i][j][k] += momentum_ * deltaWeights_[i][j][k];
			
			//adjust bias weights
			weights_[i][j][layerSizes_[i-1]] += momentum_ * deltaWeights_[i][j][layerSizes_[i-1]];
		}
	}
	
	//adjust weights using calculated errors_ and learnRate
	for (int i = 1; i < noLayers_; i++) {
		for (int j = 0; j < layerSizes_[i]; j++) {
			//adjust normal weights
			for (int k = 0; k < layerSizes_[i-1]; k++) {
				deltaWeights_[i][j][k] = learnRate_ * errors_[i][j] * outputs_[i-1][k];
				weights_[i][j][k] += deltaWeights_[i][j][k];
			}
			
			//adjsut bias weight
			deltaWeights_[i][j][layerSizes_[i-1]] = learnRate_ * errors_[i][j];
			weights_[i][j][layerSizes_[i-1]] += deltaWeights_[i][j][layerSizes_[i-1]];
		}
	}
}

double* MLP::feedForward(double* inputs) {
	//first layer is an input layer
	for (int i = 0; i < layerSizes_[0]; i++)
		outputs_[0][i] = inputs[i];
	
	//calculate output for each layer except input layer
	for (int i = 1; i < noLayers_; i++) {
		//calculates output for each neuron
		for (int j = 0; j < layerSizes_[i]; j++) {
			double output = 0.0;
			//calculates output based on all weights inside neuron; number of weights in neuron is equal to number of neurons in previous layer
			for (int k = 0; k < layerSizes_[i-1]; k++) {
				//when it comes to one neuron, output is equal to (inputs * weights)
				output += outputs_[i-1][k] * weights_[i][j][k];
			}
			
			//add bias which is on last position inside weights_[i][j] array of weights for j-th neuron in i-th layer
			output += weights_[i][j][layerSizes_[i-1]];
			
			//apply sigmoid activation function (1.0 / (1.0 + e^-x); possible to replace with pointer to function
			double sigmoidValue = 1.0 / (1.0 + exp(-output));
			outputs_[i][j] = sigmoidValue;
		}
	}
	
	//return outputs from last layer
	return outputs_[noLayers_-1];
}

double MLP::getMSError(double* answers) {
	double ret = 0.0;
	for (int i = 0; i < layerSizes_[noLayers_-1]; i++) {
		double delta = answers[i] - outputs_[noLayers_-1][i];
		ret += delta * delta;
	}

	ret *= 0.5;
	return ret;
}