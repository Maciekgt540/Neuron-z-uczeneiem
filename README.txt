Plik compile.bat moze byc uruchomiony by skompilowa� kod. Wymagane g++ na komputerze!

Plik generate.cpp oraz jego binarka generator.exe s�u�� do generacji pliku learn.txt.

URUCHOMIENIE solver.exe, aby wpisa� inne paramtetry niz domy�lne:
1) Otworzy� konsol� (logo windows + R i wpisa� cmd, albo dowolna inna metoda)
2) Przej�� do katalogu, gdzie znajduje si� solver.exe (polecenie cd)

solver.exe learnRate momentumRate noHiddenLayers noNeuronsInHidden1 noNeuronsInHidden2 ...

Na przyk�ad:
solver.exe 0.1 0.0 1 3
oznacza to 3 warstwy sieci, z czego 1 jest ukryta i ta ukryta ma 3 neurony (warstwa 1 ma zawsze 9 neuron�w a ostatnia zawsze 1 neuron, zgodnie z wymogami zadania)

solver.exe 0.01 0.001 2 6 3
2 ukryte warstwy, pierwsza z nich ma 6 neurono�, druga 3

solver.exe
domyslne parametry zostan� wczytane r�wnowazne poleceniu: solver.exe 0.1 0.0 2 6 3