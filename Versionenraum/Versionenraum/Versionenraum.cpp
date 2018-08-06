// Versionenraum.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "backpropagation.h"

using namespace std;

class EigeneMethoden
{
private:
	int nrSpalte=0, nrZeile=0; //Zählvariablen für das Array
	const static int maxSpalten = 23, maxZeilen = 1001;
public:
	string testdaten [maxSpalten][maxZeilen];

	public: void einlesen()
	{
		ifstream lesen;
		lesen.open("C:\\Testdaten\\Wohnungskartei_Muster_Master_4_S_teach.csv", ios::in);
		if (lesen)
		{
			//Datei bis Ende einlesen und bei ';' strings trennen
			string  einzelZeile = "";
			string zelle = "";
			while (getline(lesen, einzelZeile))
			{
				istringstream zeileLesen(einzelZeile);
				while (getline(zeileLesen, zelle,';'))
				{
					if (nrSpalte > maxSpalten-1)
					{
						nrSpalte = 0;
						nrZeile++;
					}
					testdaten[nrSpalte][nrZeile] = zelle;
					cout << "Zeile: " << nrZeile << " ";
					cout << "Spalte: " << nrSpalte << "   ";
					cout <<  zelle << endl; //alle Strings getrennt ausgeben
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
};

int main() 
{
	EigeneMethoden meineMethoden;
	feedForwardNetwork *NN = new feedForwardNetwork(15, 1, 10); // Erzeugen einer Instanz vom NN

	//Einlesen von Trainingsdaten
	meineMethoden.einlesen();

	// Trainingsdaten
	double in[10][15] = { { 0.0f,0.0f,1.0f,0.0f,0.0f,1.0f,0.0f,0.0f,1.0f,0.0f,0.0f,1.0f,0.0f,0.0f,1.0f },
	{ 1.0f,1.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f,1.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f },
	{ 1.0f,1.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f,1.0f },
	{ 1.0f,0.0f,1.0f,1.0f,0.0f,1.0f,1.0f,1.0f,1.0f,0.0f,0.0f,1.0f,0.0f,0.0f,1.0f },
	{ 1.0f,1.0f,1.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f,1.0f },
	{ 1.0f,1.0f,1.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f,1.0f,0.0f,1.0f,1.0f,1.0f,1.0f },
	{ 1.0f,1.0f,1.0f,0.0f,0.0f,1.0f,0.0f,0.0f,1.0f,0.0f,0.0f,1.0f,0.0f,0.0f,1.0f },
	{ 1.0f,1.0f,1.0f,1.0f,0.0f,1.0f,1.0f,1.0f,1.0f,1.0f,0.0f,1.0f,1.0f,1.0f,1.0f },
	{ 1.0f,1.0f,1.0f,1.0f,0.0f,1.0f,1.0f,1.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f,1.0f },
	{ 1.0f,1.0f,1.0f,1.0f,0.0f,1.0f,1.0f,0.0f,1.0f,1.0f,0.0f,1.0f,1.0f,1.0f,1.0f },
	};

	// Lehrer zu den Trainingsdaten
	double teach[10][10] = { { 0.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f },
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
	static float last_error = 1000.0f;
	double o[MAX_OUTPUT_LAYER_SIZE];
	double t[MAX_OUTPUT_LAYER_SIZE];
	double error, total_error = 0.0f;
	bool  learned = false;
	char  buffer[50];
	int number = 10;
	int iterations = 0;

	// Initialisierung des NN
	NN->configure(15, 30, 10); //15 Neuronen Eingang, 30 Neuronen Mitte, 10 Neuronen Ausgang
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

	cin.get(); // Warte auf Eingabe, damit Konsole nicht gleich verschwindet
	return 0;
}
