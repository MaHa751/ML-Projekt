/* NeuronalesNetz.cpp: Implementierung einer Lösung mit einem neuronalen Netze für die Vorlesung Maschinelles Lernen und Computational Intelligence
Ziel: Bewertung von Eigenschaften einer Wochnungssuche und Bestimmung des "Geschmacks"
Verwendung (mit Änderungen) der "backpropagation.cpp" und "backpropagation.h" von Dirk Reichhardt

WICHTIG: Aufgrund der Größe von Arrays ist es ggf. notwendig die Stapelreservierungsgröße zu vergrößern
Anleitung für Visual Studio: 
1. Rechtsklick auf die Projektmappe -> Eingenschaften
2. Reiter Linker -> System
3. Vergrößern der Stapelreservierungsgröße im entsprechenden Eintrag. Getestet mit 8000000 -> funktioniert
*/


#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stdlib.h>
#include "backpropagation.h"

using namespace std;

class EigeneMethoden
{
private:
	feedForwardNetwork NN; //Erzeugen einer Instanz vom NN
	int nrSpalte = 0, nrZeile = 0; //Zählvariablen Arrays 
	int correctClassifications = 0; //Anzahl der Korrekt zugeordneten Datensätze
	double error, total_error = 0.0f; //Variablen zur Fehlerbeurteilung
	double o[MAX_OUTPUT_LAYER_SIZE]; // berechneter Output
	double t[MAX_OUTPUT_LAYER_SIZE]; // Output gemaes des Lehrers
	bool  learned = false; //Zustandsvariable als Abbruchkriterium, falls der Fehler klein genug ist
	int correct = 0; //Anzahl der erforderlichen Korrekten gelernten Werte
	int iterations = 0; //Anzahl der Iterationen
	int maxZeilen = 0, maxSpalten = 0;

public:
	void initNN(double epsilon, double learningRate, int korrekte, int in, int mid, int out)
	{
		// Parameter	
		correct = korrekte; //setzen der Anzahl der erforderlichen korrekten Datensätze

		// Initialisierung des NN
		NN.configure(in, mid, out); //Setzen der Neuronen
		NN.init(); //Aufruf der Init-Funktion
		NN.setEpsilon(epsilon); //Setzen von Epsilon (max. Fehler)
		NN.setLearningRate(learningRate); //Setzen der Lernraten
	}

	template<int x, int y> //Template zur vereinfachten Übergabe der 2D Arrays
	void CSVeinlesen(string dateipfad, string(&arr)[x][y]) //Funktion zum Einlesen einer *.csv-Daten und speichern in einem String-Array
	{	
		nrSpalte = 0, nrZeile = 0; //Setzen der Zählvariablen für Spalte und zeile auf 0 als Initialwert
		maxSpalten = sizeof(arr) / sizeof arr[0]; //Bestimmen der Spaltengröße des übergebenen Arrays
		ifstream lesen; //Lokaler stream in den die *.csv-Datei geladen wird
		lesen.open(dateipfad, ios::in);
		if (lesen)
		{
			//Datei bis Ende zeilenweise einlesen 
			string  einzelZeile = "";
			string zelle = "";
			while (getline(lesen, einzelZeile))
			{
				//Zeilenstring bei ';' in Einzelstrings trennen, welche einer Zelle entsprechen
				istringstream zeileLesen(einzelZeile);
				while (getline(zeileLesen, zelle,';'))
				{
					if (nrSpalte > maxSpalten-1)
					{
						nrSpalte = 0;
						nrZeile++;
					}
					arr[nrSpalte][nrZeile] = zelle;
					//cout << "Zeile: " << nrZeile << " " << "Spalte: " << nrSpalte << "   " << arr[nrSpalte][nrZeile] << endl; //alle Strings getrennt ausgeben
					nrSpalte++;
				}	
			}
			lesen.close();
		}
		else 
		{
			cerr << "Fehler beim Lesen!" << endl;
		}
	}

	template<int x, int y, int u, int v> //Template zur vereinfachten Übergabe der 2D Arrays
	void datenDig(string(&text)[x][y], string(&attribute)[u][v],float(&zahl)[x][y])
	{
		nrSpalte = 0, nrZeile = 0;
		maxSpalten = sizeof(zahl) / sizeof zahl[0];
		maxZeilen = sizeof zahl[0] / sizeof (float);
		for (nrSpalte = 0; nrSpalte < maxSpalten; nrSpalte++)
		{
			int imax = stoi(attribute[nrSpalte][0])+1; 
			for (nrZeile = 1; nrZeile < maxZeilen; nrZeile++)
			{
				string zelle = text[nrSpalte][nrZeile];
				zahl[nrSpalte][nrZeile] = 0;
				if (zelle == "ja")
					zahl[nrSpalte][nrZeile] = 1;
				else if (zelle == "nein")
					zahl[nrSpalte][nrZeile] = 0;
				else
				{
					for (int i = 1; i < imax;i++)
					{
						string attribut = attribute[nrSpalte][i];
						if (zelle == attribut)
						{
							zahl[nrSpalte][nrZeile] = (float)i / ((float)imax - 1);
						}
					}
				}
				//cout << "Zeile: " << nrZeile << " " << "Spalte: " << nrSpalte << "   " << zahl[nrSpalte][nrZeile] << endl; //ausgeben
			}
		}
	}

	template<int x, int y> //Template zur vereinfachten Übergabe der 2D Arrays
	void lernen(float(&arr)[x][y])
	{
		nrSpalte = 0, nrZeile = 0;
		maxSpalten = sizeof(arr) / sizeof arr[0];
		maxZeilen = sizeof arr[0] / sizeof(float);
		correctClassifications = 0;
		while (correctClassifications < correct)
		{
			// lernen alle Datensätze
			for (nrZeile = 1; nrZeile < maxZeilen; nrZeile++) 
			{
				iterations++;

				for (nrSpalte = 0; nrSpalte < maxSpalten - 1; nrSpalte++) 
					NN.setInput(nrSpalte, arr[nrSpalte][nrZeile]); //Setzen der Eingangsneuronen

				//lernen für einen Datensatz solange bis der Fehler kleiner epsilon ist
				learned = false;
				t[0] = arr[maxSpalten - 1][nrZeile]; // setzen des teachers, 1 Neoron
				while (!learned)
				{
					NN.apply();
					o[0] = NN.getOutput(0); // Berechnen des Ausgangs, 1. Neuron
					error = NN.energy(t, o, 1); //1 = 1 ausgangsneuron

					if (error > NN.getEpsilon())
						NN.backpropagate(t);
					else
						learned = true;
				}
			}

			// get status of learning
			correctClassifications = 0;
			total_error = 0.0f;
			for (nrZeile = 1; nrZeile< maxZeilen; nrZeile++) 
			{
				for (nrSpalte = 0; nrSpalte < maxSpalten - 1; nrSpalte++) 
					NN.setInput(nrSpalte, arr[nrSpalte][nrZeile]); //Setzen der Eingangsneuronen

				NN.apply();
				o[0] = NN.getOutput(0); // Berechnen des Ausgangs, 1. Neuron
				t[0] = arr[maxSpalten - 1][nrZeile]; // setzen des teachers, 1 Neoron

				error = NN.energy(t, o, 1); //1 = 1 ausgangsneuron
				total_error += error;

				if (error < NN.getEpsilon())
					correctClassifications++;
			}

			// Ausgabe jeder Iteration
			printf("[%4d]>> Korrekte: %2d Gesamtfehler : %5.5f\n", iterations / maxZeilen, correctClassifications, total_error); // Ausgabe des Lernvorgangs
		}
		printf("Iterationen: %d\n ", iterations / maxZeilen);
	}

	template<int x, int y> //Template zur vereinfachten Übergabe der 2D Arrays
	void testen(float(&arr)[x][y])
	{
		nrSpalte = 0, nrZeile = 0;
		maxSpalten = sizeof(arr) / sizeof arr[0];
		maxZeilen = sizeof arr[0] / sizeof(float);
		correctClassifications = 0;
		
		for (nrZeile = 1; nrZeile < maxZeilen; nrZeile++)
		{
			printf("IN %1.0i: ",nrZeile);
			float zelle = 0;
			for (nrSpalte = 0; nrSpalte < maxSpalten - 1; nrSpalte++)
			{
				zelle = arr[nrSpalte][nrZeile];
				NN.setInput(nrSpalte, zelle); //Setzen der Eingangsneuronen
				printf("%1.1f ", zelle);
			}
			NN.apply();

			double out = NN.getOutput(0);
			printf("OUT: %1.0f ", out);
			bool korr = false;
			if ((out > 0.9 && arr[maxSpalten - 1][nrZeile] == 1) || (out < 0.9 && arr[maxSpalten - 1][nrZeile] == 0))
			{
				correctClassifications++;
				korr = true;
			}
			if (korr)
				printf("RICHTIG\n");
			else
				printf("FALSCH\n");
		}
		printf("\nAnzahl Richtige: %1.0i \n", correctClassifications);
		printf("Anzahl Falsche: %1.0i \n", maxZeilen - correctClassifications - 1);

	}
};

int main() 
{
	EigeneMethoden meineMethoden;

	const static int maxSpalten = 23, maxZeilenLerndaten = 401, maxZeilenTestdaten = 1002, maxZeilenAttribute = 19; //Dimensionen der Datensätze
	string lerndaten[maxSpalten][maxZeilenLerndaten]; //Array mit Lerndaten als Zeichenkette
	float lerndatenZahl[maxSpalten][maxZeilenLerndaten]; //Array mit Lerndaten als skalierte Zahlenwerte
	string testdaten[maxSpalten][maxZeilenTestdaten]; //Array mit Testdaten als Zeichenkette
	float testdatenZahl[maxSpalten][maxZeilenTestdaten]; //Array mit Testdaten als skalierte Zahlenwerte
	string attribute[maxSpalten][maxZeilenAttribute]; //Array mit Attributen als Zeichenkette

	//Einlesen von Daten
	cout << "Einlesen von Trainingsdaten" << endl;
	meineMethoden.CSVeinlesen("C:\\Testdaten\\Wohnungskartei_Muster_Master_4_S_teach.csv", lerndaten);
	cout << "Einlesen von Testdaten" << endl;
	meineMethoden.CSVeinlesen("C:\\Testdaten\\Wohnungskartei_Muster_Master_5_S.csv", testdaten);
	cout << "Einlesen von Attributen" << endl;
	meineMethoden.CSVeinlesen("C:\\Testdaten\\Attribute.csv", attribute);

	//Digitalisieren der Daten
	cout << "Ueberfuehren der Trainigsdaten in skalierte Zahlen" << endl;
	meineMethoden.datenDig(lerndaten, attribute, lerndatenZahl);
	cout << "Ueberfuehren der Testdaten in skalierte Zahlen" << endl;
	meineMethoden.datenDig(testdaten, attribute, testdatenZahl);

//******************************************************************************************************
#pragma region Schritt 1: Training mit Lerndaten, Test mit Testdaten
	//Schritt 1: Training mit Lerndaten, Test mit Testdaten
	meineMethoden.initNN(0.005f, 0.02f, maxZeilenLerndaten - 1, maxSpalten - 1, (maxSpalten - 1) * 2, 1); //22 Neuronen Eingang, 44 Neuronen Mitte, 1 Neuronen Ausgang
	//NN Trainieren mit Lerndaten
	cout << "\nWeiter mit Training mit Lerndaten -> ENTER" << endl;
	cin.get(); // Warte auf Eingabe, damit Konsole nicht gleich verschwindet
	meineMethoden.lernen(lerndatenZahl);

	//NN Test mit Lerndaten
	cout << "\nWeiter mit Test mit Lerndaten -> ENTER" << endl;
	cin.get(); // Warte auf Eingabe, damit Konsole nicht gleich verschwindet
	meineMethoden.testen(lerndatenZahl);

	//NN Test mit Testdaten
	cout << "\nWeiter mit Test mit Testdaten -> ENTER" << endl;
	cin.get(); // Warte auf Eingabe, damit Konsole nicht gleich verschwindet
	meineMethoden.testen(testdatenZahl);
#pragma endregion

//******************************************************************************************************
#pragma region Schritt 2: Training mit Testdaten, Test mit Lerndaten
	//Schritt 2: Training mit Testdaten, Test mit Lerndaten
	meineMethoden.initNN(0.005f, 0.02f, maxZeilenTestdaten- 1, maxSpalten - 1, (maxSpalten - 1) * 2, 1); //22 Neuronen Eingang, 44 Neuronen Mitte, 1 Neuronen Ausgang
	//NN Trainieren mit Lerndaten
	cout << "\nWeiter mit Training mit Testdaten -> ENTER" << endl;
	cin.get(); // Warte auf Eingabe, damit Konsole nicht gleich verschwindet
	meineMethoden.lernen(testdatenZahl);

	//NN Test mit Lerndaten
	cout << "\nWeiter mit Test mit Testdaten -> ENTER" << endl;
	cin.get(); // Warte auf Eingabe, damit Konsole nicht gleich verschwindet
	meineMethoden.testen(testdatenZahl);

	//NN Test mit Testdaten
	cout << "\nWeiter mit Test mit Lerndaten -> ENTER" << endl;
	cin.get(); // Warte auf Eingabe, damit Konsole nicht gleich verschwindet
	meineMethoden.testen(lerndatenZahl);
#pragma endregion

//******************************************************************************************************
#pragma region Beenden des Programms
	//Beenden des Programms
	cout << "Programm beenden -> ENTER" << endl;
	cin.get(); // Warte auf Eingabe, damit Konsole nicht gleich verschwindet
	return 0;
#pragma endregion

	
}
