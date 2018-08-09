// Versionenraum.cpp : Defines the entry point for the console application.
//

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
	int nrSpalte = 0, nrZeile = 0; //Zählvariablen Arrays 

public:
	template<int x, int y>
	void CSVeinlesen(string dateipfad, string(&arr)[x][y], int maxSpalten)
	{	
		nrSpalte = 0, nrZeile = 0;
		ifstream lesen;
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

	template<int x, int y, int u, int v>
	void datenDig(string(&text)[x][y], string(&attribute)[u][v],float(&zahl)[x][y], int const maxSpalten, int const maxZeilen)
	{
		for (nrSpalte = 0; nrSpalte < maxSpalten; nrSpalte++)
		{
			int imax = stoi(attribute[nrSpalte][0])+1; 
			for (nrZeile = 1; nrZeile < maxZeilen; nrZeile++)
			{
				string zelle = text[nrSpalte][nrZeile];
				for (int i = 1; i < imax;i++)
				{
					string attribut = attribute[nrSpalte][i];
					zahl[nrSpalte][nrZeile] = 0;
					if (zelle == attribut)
					{
						zahl[nrSpalte][nrZeile] = ((float)i - 1) / ((float)imax - 1);
						//cout << "Zeile: " << nrZeile << " " << "Spalte: " << nrSpalte << "   " << zahl[nrSpalte][nrZeile] << endl; //ausgeben
					}
				}
			}
		}
	}
};

int main() 
{
	int const maxSpalten = 23, maxZeilenLerndaten = 401, maxZeilenTestdaten = 1002;
	string lerndaten[maxSpalten][maxZeilenLerndaten];
	float lerndatenZahl[maxSpalten][maxZeilenLerndaten];
	string testdaten[maxSpalten][maxZeilenTestdaten];
	float testdatenZahl[maxSpalten][maxZeilenTestdaten];
	
	int const maxX = 23, maxY = 19;
	string attribute[maxX][maxY];
	
	EigeneMethoden meineMethoden;
	feedForwardNetwork *NN = new feedForwardNetwork(22, 44, 1); // Erzeugen einer Instanz vom NN

	//Einlesen von Daten
	cout << "Einlesen von Trainingsdaten" << endl;
	meineMethoden.CSVeinlesen("C:\\Testdaten\\Wohnungskartei_Muster_Master_4_S_teach.csv", lerndaten, maxSpalten);
	cout << "Einlesen von Testdaten" << endl;
	meineMethoden.CSVeinlesen("C:\\Testdaten\\Wohnungskartei_Muster_Master_5_S.csv", testdaten, maxSpalten);
	cout << "Einlesen von Attributen" << endl;
	meineMethoden.CSVeinlesen("C:\\Testdaten\\Attribute.csv", attribute, maxX);

	//Digitalisieren der Daten
	cout << "Digitalisieren der Trainigsdaten" << endl;
	meineMethoden.datenDig(lerndaten, attribute, lerndatenZahl, maxSpalten, maxZeilenLerndaten);
	cout << "Digitalisieren der Testdaten" << endl;
	meineMethoden.datenDig(testdaten, attribute, testdatenZahl, maxSpalten, maxZeilenTestdaten);

	// Trainingsdaten
	double in[10][15] =    {{ 0.0f,0.0f,1.0f,0.0f,0.0f,1.0f,0.0f,0.0f,1.0f,0.0f,0.0f,1.0f,0.0f,0.0f,1.0f },
							{ 1.0f,1.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f,1.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f },
							{ 1.0f,1.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f,1.0f },
							{ 1.0f,0.0f,1.0f,1.0f,0.0f,1.0f,1.0f,1.0f,1.0f,0.0f,0.0f,1.0f,0.0f,0.0f,1.0f },
							{ 1.0f,1.0f,1.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f,1.0f },
							{ 1.0f,1.0f,1.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f,1.0f,0.0f,1.0f,1.0f,1.0f,1.0f },
							{ 1.0f,1.0f,1.0f,0.0f,0.0f,1.0f,0.0f,0.0f,1.0f,0.0f,0.0f,1.0f,0.0f,0.0f,1.0f },
							{ 1.0f,1.0f,1.0f,1.0f,0.0f,1.0f,1.0f,1.0f,1.0f,1.0f,0.0f,1.0f,1.0f,1.0f,1.0f },
							{ 1.0f,1.0f,1.0f,1.0f,0.0f,1.0f,1.0f,1.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f,1.0f },
							{ 1.0f,1.0f,1.0f,1.0f,0.0f,1.0f,1.0f,0.0f,1.0f,1.0f,0.0f,1.0f,1.0f,1.0f,1.0f }};

	// Lehrer zu den Trainingsdaten
	double teach[10][10] =     {{ 0.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f },
								{ 0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f },
								{ 0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f },
								{ 0.0f,0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f },
								{ 0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,0.0f },
								{ 0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f },
								{ 0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,0.0f,0.0f },
								{ 0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,0.0f },
								{ 0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f },
								{ 1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f } };

	//Testdaten
	double test[15] = { 1.0f,1.0f,1.0f,1.0f,0.0f,1.0f,1.0f,1.0f,1.0f,1.0f,0.0f,1.0f,1.0f,1.0f,1.0f };

	// Parameter
	int correctClassifications = 0;
	int i, j;
	static double last_error = 1000.0f;
	double o[MAX_OUTPUT_LAYER_SIZE];
	double t[MAX_OUTPUT_LAYER_SIZE];
	double error, total_error = 0.0f;
	bool  learned = false;
	char  buffer[50];
	int number = 1000;//xxx wert anpassen
	int iterations = 0;

	// Initialisierung des NN
	NN->configure(22, 44, 1); //22 Neuronen Eingang, 44 Neuronen Mitte, 1 Neuronen Ausgang
	NN->init();
	NN->setEpsilon(0.0005f);
	NN->setLearningRate(0.2f);

	//trainieren
	printf("Start Training:\n");
	while (correctClassifications < number)
	{
		for (i = 0; i < number; i++)
		{
			iterations++;

			for (j = 0; j < 15; j++)
			{
				NN->setInput(j, in[i][j]);
			}

			learned = false;

			while (!learned)
			{
				NN->apply();

				for (j = 0; j < 10; j++)
				{
					o[j] = NN->getOutput(j);
				}

				for (j = 0; j < 10; j++)
					t[j] = teach[i][j];

				error = NN->energy(t, o, 10);

				if (error > NN->getEpsilon())
				{
					NN->backpropagate(t);
				}
				else
					learned = true;
			}
		}
		
		// get status of learning
		correctClassifications = 0;
		total_error = 0.0f;
		for (i = 0; i< number; i++)
		{
			for (j = 0; j<15; j++)
			{
				NN->setInput(j, in[i][j]);
			}

			NN->apply();

			for (j = 0; j<10; j++)
			{
				o[j] = NN->getOutput(j);
			}

			for (j = 0; j<10; j++)
				t[j] = teach[i][j];

			error = NN->energy(t, o, 10);
			total_error += error;

			if (error < NN->getEpsilon())
			{
				correctClassifications++;
			}
		}

		// total error
		last_error = total_error;
		printf("[%4d]>> Korrekte: %2d Fehler : %5.4f\n", iterations / 10, correctClassifications, total_error);
	}

	// Test 
	printf("Iterationen: %d\n ", iterations / 10);
	printf("\nTest:\n");
	for (i = 0; i<10; i++)
	{
		printf("[");
		for (j = 0; j<15; j++)
		{
			NN->setInput(j, in[i][j]);
			printf("%1d ", (int)in[i][j]);
		}
		printf("] :");
		NN->apply();
		for (j = 0; j<10; j++)
		{
			printf("%3.1f ", NN->getOutput(j));
		}
		printf("\n");
	}

	// Test mit eigenen Daten
	cout << endl << "Test Nr. 2 mit eigenen Daten" << endl;
	printf("[");
	for (j = 0; j<15; j++)
	{
		NN->setInput(j, test[j]);
		printf("%1d ", (int)test[j]);
	}
	printf("] :");
	NN->apply();
	for (j = 0; j<10; j++)
	{
		printf("%3.1f ", NN->getOutput(j));
	}
	printf("\n");

	delete NN;

	cin.get(); // Warte auf Eingabe, damit Konsole nicht gleich verschwindet
	return 0;
}
