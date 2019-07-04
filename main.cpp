#include <iostream>
#include <fstream>
#include <time.h>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "neuron.h"

using namespace std;

struct BoardState
{
	double* state;
	double* answer;
};

void learn(MLP network, ifstream& learnFile, ofstream& outputFile) {
	vector<BoardState> states;
	while (!learnFile.eof()) {
		int* state = new int[9];
		int answer;
		for (int i = 0; i < 9; i++)
			learnFile >> state[i];
		learnFile >> answer;
		
		BoardState board;
		board.state = new double[9];
		board.answer = new double[1];
		for (int i = 0; i < 9; i++)
			board.state[i] = state[i];
		board.answer[0] = answer;
		states.push_back(board);
	}
	
	for (int i = 0; i < 1000; i++) {
		random_shuffle(states.begin(), states.end());
		double mse = 0.0;
		
		for (int j = 0; j < states.size(); j++) {
			network.backPropagate(states[j].state, states[j].answer);
			mse += network.getMSError(states[j].answer);
		}
	
		outputFile << "[Learn" << i << "] Error = " << (mse / (double)states.size()) << "\n";
	}
	
	outputFile << "\n\n";
}

void test(MLP network, ifstream& testFile, ofstream& outputFile) {
	int index = 1;
	int correct = 0;
	while (!testFile.eof()) {
		double* state = new double[9];
		int answer;
		for (int i = 0; i < 9; i++)
			testFile >> state[i];
		testFile >> answer;
		
		double* solution = network.feedForward(state);
		if ((answer == 1 && solution[0] >= 0.5) || (answer == 0 && solution[0] < 0.5)) {
			outputFile << "[GOOD]";
			correct++;
		} else {
			outputFile << "[FAIL]";
		}

		outputFile << "[Test " << index << "] " << solution[0] << "\n";
		index++;
	}
	
	outputFile << "\nSuccess rate: " << correct << "/" << (index + 1) << " (" << ((double)correct / (double)(index + 1) * 100.0) << "%)\n";
}

int main(int argc, char** argv) {
	srand(time(NULL));
	
	//streams to read and write data
	ifstream learnFile;
	learnFile.open("learn.txt");
	ifstream testFile;
	testFile.open("test.txt");
	ofstream outputFile;
	outputFile.open("output.txt", ios::trunc);
	
	//when specified, assign custom learn rate; otherwise assign 0.1
	double learnRate = (argc < 2) ? 0.1 : atof(argv[1]);
	double momentum = (argc < 3) ? 0.01 : atof(argv[2]);
	int noLayers;
	int* layerSizes;
	if (argc >= 4) {
		noLayers = atoi(argv[3]) + 2;
		layerSizes = new int[noLayers];
		layerSizes[0] = 9;
		layerSizes[noLayers-1] = 1;
		for (int i = 1; i < noLayers - 1; i++) {
			layerSizes[i] = atoi(argv[4+i-1]);
		}
	} else {
		noLayers = 4;
		layerSizes = new int[4] = { 9, 6, 3, 1 };
	}

	MLP network(noLayers, layerSizes, learnRate, momentum);
	
	learn(network, learnFile, outputFile);
	test(network, testFile, outputFile);
	
	//close all files
	learnFile.close();
	testFile.close();
	outputFile.close();
	
	return 0;
}
