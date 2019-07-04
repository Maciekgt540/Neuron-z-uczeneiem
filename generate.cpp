#include <iostream>
#include <fstream>
#include <string>

using namespace std;

ofstream outputFile;
int winning[8][3] = {
	{0, 1, 2}, {3, 4, 5}, {6, 7, 8},
	{0, 3, 6}, {1, 4, 7}, {2, 5, 8},
	{0, 4, 8}, {2, 4, 6}
};

//Check whether tictactoe board is "cross"-player win
int compareWinning(int array[9]) {
	for (int i = 0; i < 8; i++) {
		bool allSet = true;
		for (int j = 0; j < 3; j++) {
			if (array[winning[i][j]] == 0) {
				allSet = false;
				break;
			}
		}
		if (allSet) {
			return 1;
		}
	}
	
	return 0;
}

//Generate all final board states to file
void generate(int n, bool isLast) {
	int array[9];
	string ret = "";
	int bit = 1 << 9 - 1;
	int index = 0;
	while (bit) {
		int value = (n & bit) ? 1 : 0;
		array[index] = value;
		outputFile << value << " ";
		bit >>= 1;
		index++;
	}
	
	outputFile << compareWinning(array);
	if (!isLast)
		outputFile << "\n";
}

int main() {
	outputFile.open("learn.txt", ios::trunc);
	int n = 1 << 9;
	for (int i = 0; i < n - 1; i++) {
		generate(i, false);
	}

	generate(n - 1, true);
	outputFile.close();
}