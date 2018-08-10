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

	// Parameter
	int correctClassifications = 0;
	int nrZeile, nrSpalte; //i=nrZeile, j=nrSpalte
	static double last_error = 1000.0f;
	double o[MAX_OUTPUT_LAYER_SIZE]; //output
	double t[MAX_OUTPUT_LAYER_SIZE]; //teacher
	double error, total_error = 0.0f;
	bool  learned = false;
	int correct = maxZeilenLerndaten-1;//xxx wert anpassen
	int iterations = 0;

	// Initialisierung des NN
	NN->configure(maxSpalten-1, (maxSpalten - 1)*2, 1); //22 Neuronen Eingang, 44 Neuronen Mitte, 1 Neuronen Ausgang
	NN->init();
	NN->setEpsilon(0.005f);
	NN->setLearningRate(0.02f);

	//trainieren mit Lerndaten
	cout << "Weiter mit Training mit Lerndaten -> ENTER" << endl;
	cin.get();
	while (correctClassifications < correct)
	{
		for (nrZeile = 1; nrZeile < maxZeilenLerndaten; nrZeile++) //ehemals i
		{
			iterations++;

			for (nrSpalte = 0; nrSpalte < maxSpalten-1; nrSpalte++) //ehemals j 
				NN->setInput(nrSpalte, lerndatenZahl[nrSpalte][nrZeile]); //Setzen der Eingangsneuronen

			learned = false;

			while (!learned)
			{
				NN->apply();

				//for (nrSpalte = 0; nrSpalte < maxSpalten - 1; nrSpalte++) // 
				o[0] = NN->getOutput(0); // Berechnen des Ausgangs, 1. Neuron
				
				//for (nrSpalte = 0; nrSpalte < maxSpalten - 1; nrSpalte++) //
				t[0] = lerndatenZahl[maxSpalten-1][nrZeile]; // setzen des teachers, 1 Neoron

				error = NN->energy(t, o, 1); //1 = 1 ausgangsneuron

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
		for (nrZeile = 1; nrZeile< maxZeilenLerndaten; nrZeile++) // ehemals i
		{
			for (nrSpalte = 0; nrSpalte < maxSpalten - 1; nrSpalte++) //ehemals j
				NN->setInput(nrSpalte, lerndatenZahl[nrSpalte][nrZeile]); //Setzen der Eingangsneuronen
			
			NN->apply();

			//for (j = 0; j<10; j++)
			o[0] = NN->getOutput(0); // Berechnen des Ausgangs, 1. Neuron

			//for (j = 0; j<10; j++)
			t[0] = lerndatenZahl[maxSpalten - 1][nrZeile]; // setzen des teachers, 1 Neoron

			error = NN->energy(t, o, 1); //1 = 1 ausgangsneuron
			total_error += error;

			if (error < NN->getEpsilon())
			{
				correctClassifications++;
			}
		}

		// total error
		last_error = total_error;
		//if (iterations / maxZeilenLerndaten > 1000)
		//	correctClassifications = 1000;
		printf("[%4d]>> Korrekte: %2d Fehler : %5.5f\n", iterations / maxZeilenLerndaten, correctClassifications, total_error); // Ausgabe des Lernvorgangs
	}
	printf("Iterationen: %d\n ", iterations / maxZeilenLerndaten);
	
	// Test mit Lerndaten
	cout << "Weiter mit Test mit Lerndaten -> ENTER" << endl;
	cin.get();
	correctClassifications = 0;
	for (nrZeile = 1; nrZeile < maxZeilenLerndaten; nrZeile++)
	{
		float zelle = 0;
		for (nrSpalte = 0; nrSpalte < maxSpalten - 1; nrSpalte++)
		{
			zelle = lerndatenZahl[nrSpalte][nrZeile];
			NN->setInput(nrSpalte, zelle); //Setzen der Eingangsneuronen
			printf("%1.2f ", zelle);
		}
		NN->apply();
		
		float out = NN->getOutput(0);
		if ((out > 0.9 && lerndatenZahl[maxSpalten-1][nrZeile]==1)|| (out < 0.9 && lerndatenZahl[maxSpalten - 1][nrZeile] == 0))
			correctClassifications++;
		
		printf("%1.0f \n", out);
	}
	printf("\nAnzahl Richtige: %1.0i \n", correctClassifications);
	printf("Anzahl Falsche: %1.0i \n", maxZeilenLerndaten-correctClassifications-1);

	// Test mit Testdaten
	cout << "Weiter mit Test mit Testdaten -> ENTER" << endl;
	cin.get();
	correctClassifications = 0;
	for (nrZeile = 1; nrZeile < maxZeilenTestdaten; nrZeile++)
	{
		float zelle = 0;
		for (nrSpalte = 0; nrSpalte < maxSpalten - 1; nrSpalte++)
		{
			zelle = testdatenZahl[nrSpalte][nrZeile];
			NN->setInput(nrSpalte, zelle); //Setzen der Eingangsneuronen
			printf("%1.2f ", zelle);
		}
		NN->apply();

		float out = NN->getOutput(0);
		if ((out > 0.9 && testdatenZahl[maxSpalten - 1][nrZeile] == 1) || (out < 0.9 && testdatenZahl[maxSpalten - 1][nrZeile] == 0))
			correctClassifications++;

		printf("%1.0f \n", out);
	}
	printf("\nAnzahl Richtige: %1.0i \n", correctClassifications);
	printf("Anzahl Falsche: %1.0i \n", maxZeilenTestdaten - correctClassifications - 1);

	cout << "Programm beenden -> ENTER" << endl;
	cin.get(); // Warte auf Eingabe, damit Konsole nicht gleich verschwindet
	delete NN;
	return 0;
}
