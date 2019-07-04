Plik compile.bat moze byc uruchomiony by skompilowaæ kod. Wymagane g++ na komputerze!

Plik generate.cpp oraz jego binarka generator.exe s³u¿¹ do generacji pliku learn.txt.

URUCHOMIENIE solver.exe, aby wpisaæ inne paramtetry niz domyœlne:
1) Otworzyæ konsolê (logo windows + R i wpisaæ cmd, albo dowolna inna metoda)
2) Przejœæ do katalogu, gdzie znajduje siê solver.exe (polecenie cd)

solver.exe learnRate momentumRate noHiddenLayers noNeuronsInHidden1 noNeuronsInHidden2 ...

Na przyk³ad:
solver.exe 0.1 0.0 1 3
oznacza to 3 warstwy sieci, z czego 1 jest ukryta i ta ukryta ma 3 neurony (warstwa 1 ma zawsze 9 neuronów a ostatnia zawsze 1 neuron, zgodnie z wymogami zadania)

solver.exe 0.01 0.001 2 6 3
2 ukryte warstwy, pierwsza z nich ma 6 neuronoó, druga 3

solver.exe
domyslne parametry zostan¹ wczytane równowazne poleceniu: solver.exe 0.1 0.0 2 6 3