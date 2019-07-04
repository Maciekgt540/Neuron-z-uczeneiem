#include <string>

class MLP
{
public:
	MLP(int noLayers, int* layerSizes, double learnRate, double momentum);
	~MLP();
	
	//adjust weights for learning purpose
	void backPropagate(double* inputs, double* answers);
	
	//provides array of outputs from neural network
	double* feedForward(double* inputs);
	
	//calculates mean-squared error
	double getMSError(double* answers);

private:
	//how much weights are changed in each iteration
	double learnRate_;
	
	//prevents weights from changing too quickly
	double momentum_;
	
	//number of layers, including input layer
	int noLayers_;
	
	//layerSizes_[i] means number of neurons in i-th layer
	int* layerSizes_;
	
	//outputs_[i][j] means output value from j-th neuron in i-th layer
	double** outputs_;
	
	//errors_[i][j] means local error of j-th neuron in i-th layer
	double** errors_;
	
	//weight_[i][j][k] means k-th weight in i-th layer and j-th neuron in this layer
	double*** weights_;
	
	//values added to weights in previous iteration; deltaWeights_[i][j][k] interpretation similiar to weights_
	double*** deltaWeights_;
};

