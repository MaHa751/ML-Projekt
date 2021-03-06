/* NeuronalesNetz.cpp: Implementierung einer Lösung mit einem neuronalen Netz für die Vorlesung Maschinelles Lernen und Computational Intelligence
Ziel: Vorhersage, ob eine Wohnung bei einer Wohnungssuche dem Suchenden gefällt oder nicht
Verwendung (mit Änderungen) der "backpropagation.cpp" und "backpropagation.h" von Dirk Reichhardt
Die Konsoleneinstellung ist so gewählt, dass beim Test die skalierten Attribute (Eingangsneuronen), berechneter Ausgang, Teacher und Klassifizierung
möglichst in einer Zeile dargestellt werden.

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
#include <windows.h>

using namespace std;

//Klasse, die selbst geschriebene Methoden enthält
class EigeneMethoden
{
//Private, nach außen nicht sichtbare Variablen
private:
	feedForwardNetwork NN; //Erzeugen einer Instanz vom NN
	int nrSpalte = 0, nrZeile = 0; //Zählvariablen für die Arrays 
	int correctClassifications = 0; //Anzahl der Korrekt zugeordneten Datensätze
	double error, total_error = 0.0f; //Variablen zur Fehlerbeurteilung
	double o[MAX_OUTPUT_LAYER_SIZE]; //berechneter Output
	double t[MAX_OUTPUT_LAYER_SIZE]; //Output gemäs des Lehrers
	bool  learned = false; //Zustandsvariable als Abbruchkriterium, falls der Fehler klein genug ist
	int correct = 0; //Anzahl der erforderlichen Korrekten gelernten Werte
	int iterations = 0; //Anzahl der Iterationen
	int maxZeilen = 0, maxSpalten = 0; //maximale Anzahl der Zeilen und Spalten in einem 2D Array

//******************************************************************************************************
//Nach außen sichtbare, selbst erstellte Methoden
public:
	//Methode zur Initialisierung des Neuronalen Netzes und Setzen von Parametern. Muss vor dem Lernvorgang aufgerufen werden
	//Übergabeparameter sind: epsilon=der max. Fehler, learningRate=die Lernrate, korrekte=Anzahl der notwendigen korrekten gelernten Beispiele
	//in=Anzahl der Neuronen im Eingangslayer, mid=Anzahl der Neuronen im Hiddenlayer, out=Anzahl der Neuronen im Ausgangslayer
	void initNN(double epsilon, double learningRate, int korrekte, int in, int mid, int out)
	{
		// Parameter	
		correct = korrekte; //setzen der Anzahl der erforderlichen korrekten Datensätze

		// Initialisierung des NN
		NN.configure(in, mid, out); //Setzen der Anzahl der Neuronen
		NN.init(); //Aufruf der Init-Funktion des neuronalen Netzes
		NN.setEpsilon(epsilon); //Setzen von Epsilon (max. Fehler)
		NN.setLearningRate(learningRate); //Setzen der Lernrate
	}

	template<int x, int y> //Template zur vereinfachten Übergabe der 2D Arrays
	//Funktion zum Einlesen einer *.csv-Daten und speichern in einem String-Array
	//Übergabeparameter: dateipfad=der Pfad der einzulesenden Datei
	//arr=ein 2D string Array, in dem die eingelesenen Werte gespeichert werden. Das Array muss die Größe (Zeilen,Spalten) der CSV-Datei haben
	void CSVeinlesen(string dateipfad, string(&arr)[x][y]) 
	{	
		nrSpalte = 0, nrZeile = 0; //Setzen der Zählvariablen für Spalte und zeile auf 0 als Initialwert
		maxSpalten = sizeof(arr) / sizeof arr[0]; //Bestimmen der Spaltengröße des übergebenen Arrays
		ifstream lesen; //Lokaler stream in den die *.csv-Datei geladen wird
		lesen.open(dateipfad, ios::in); //Laden der CSV-Datei in den Stream
		if (lesen) //Falls die Datei richtig geöfnet wurde, dann...
		{
			//Datei bis Ende zeilenweise einlesen 
			string  einzelZeile = ""; //Lokale Variable, die den Wert einer Einzelzeile beinhaltet
			string zelle = ""; //Lokale Variable, die den Wert einer Zelle beinhaltet
			while (getline(lesen, einzelZeile)) //solange noch Zeilen gelesen werden können, wird folgendes ausgeführt...
			{
				istringstream zeileLesen(einzelZeile); //ein neuer Stream, der den Wert der Einzelzeile beinhaltet
				while (getline(zeileLesen, zelle,';')) //Zeilenstring bei ';' in Einzelstrings trennen, welche einer Zelle entsprechen und Speichern ind Variable Zelle
				{
					if (nrSpalte > maxSpalten-1) //Falls alle Spalten in einer zeile des Arrays gefüllt wurden...
					{
						nrSpalte = 0; //...wird die Spalte wieder auf Anfang (0) gesetzt
						nrZeile++; //...und die Zeilennummer inkrementiert.
					}
					arr[nrSpalte][nrZeile] = zelle; //Speichern des ausgelesenen Wertes in das Array unter der jweiligen Spalte und Zeile
					//cout << "Zeile: " << nrZeile << " " << "Spalte: " << nrSpalte << "   " << arr[nrSpalte][nrZeile] << endl; //Ausgabe: alle Strings getrennt ausgeben
					nrSpalte++; //inkrementieren der Spalte, damit die nächste Spalte in das richtige Feld gespeicehrt wird.
				}	
			}
			lesen.close(); //Schließen des Streams
		}
		else //Falls die csv-Datei nicht korrekt geöfnet wurde...
		{
			cerr << "Fehler beim Lesen!" << endl; //...wird eine Fehler ausgegeben.
		}
	}

	template<int x, int y, int u, int v> //Template zur vereinfachten Übergabe der 2D Arrays
	//Funktion zum Überführen von Testdaten in skalierte nummerische Werte im Bereich zwischen 0 und 1. Diese Werte sollen als Eingangswerte für das NN sein
	//Übergabeparameter: text=2D Array, welches die aus der csv-Daeti eingelesenen Werte beinhaltet
	//attribute=2D Array, welches die spaltenweise Auflistung der möglichen Attributmerkmale je Attribut enthält und di eAnzahl der Auswahlmöglichkeiten. Die Spalten müssen mit den Spalten der Testdaten übereinstimmen
	//zahl=2D-Array, welches die in Zahlenwerte überführten Daten enthält
	void datenDig(string(&text)[x][y], string(&attribute)[u][v],float(&zahl)[x][y])
	{
		nrSpalte = 0, nrZeile = 0; //Setzen der Zählvariablen für Spalte und zeile auf 0 als Initialwert
		maxSpalten = sizeof(zahl) / sizeof zahl[0]; //Bestimmen der Spaltengröße des übergebenen Arrays
		maxZeilen = sizeof zahl[0] / sizeof (float); //Bestimmen der Zeilengröße des übergebenen Arrays
		for (nrSpalte = 0; nrSpalte < maxSpalten; nrSpalte++) //Zählschleife, die das Stringarray spaltenweise/attributweise durchgeht
		{
			int imax = stoi(attribute[nrSpalte][0])+1; //Einlesen, könvertieren (int) und speichern des Zahlenwertes (=Anzahl der Attributmerkmale), der in der ersten Zeile des Attributearrays steht
			for (nrZeile = 1; nrZeile < maxZeilen; nrZeile++)  //Zählschleife, die das Stringarray zeilenweise/tupelweise durchgeht
			{
				string zelle = text[nrSpalte][nrZeile]; //Lokale Variable zur Speicherung des Textwertes einer Zelle
				zahl[nrSpalte][nrZeile] = 0; //Nullsetzen des Wertes einer Zelle im Zahlenarray
				if (zelle == "ja") //Falls in der Zelle ein "ja" steht...
					zahl[nrSpalte][nrZeile] = 1; //...wird eine 1 gesetzt
				else if (zelle == "nein") //Falls in der Zelle ein "nein" steht...
					zahl[nrSpalte][nrZeile] = 0; //...wird eine 0 gesetzt
				else //sonst wird folgendes ausgeführt:
				{
					for (int i = 1; i < imax;i++) //Zählschleife, die alle möglichen Attributmerkmale eines Attributs durchgeht bis die maximale Anzahl imax erreicht wurde
					{
						string attribut = attribute[nrSpalte][i]; //Auslesen und speichern des Attributmerkmales
						if (zelle == attribut) //Falls das Attribut dem Wert der Zelle entspricht wird...
						{
							//...Konvertieren (float), linear Umrechnen und Speichern des Zahlenwertes in das Zahlenarray
							zahl[nrSpalte][nrZeile] = (float)i / ((float)imax);
						}
					}
				}
				//cout << "Zeile: " << nrZeile << " " << "Spalte: " << nrSpalte << "   " << zahl[nrSpalte][nrZeile] << endl; //Aushgabe der verarbeiteten Werte
			}
		}
	}

	template<int x, int y> //Template zur vereinfachten Übergabe der 2D Arrays
	//Funktion zum Trainieren des NN mit Hilfe vorgelegter Lerndaten
	//Übergabeparameter: arr=Zahlenarray, welches die Lerndaten/Beispiele beinhaltet
	void lernen(float(&arr)[x][y])
	{
		nrSpalte = 0, nrZeile = 0; //Setzen der Zählvariablen für Spalte und zeile auf 0 als Initialwert
		maxSpalten = sizeof(arr) / sizeof arr[0]; //Bestimmen der Spaltengröße des übergebenen Arrays
		maxZeilen = sizeof arr[0] / sizeof(float); //Bestimmen der Zeilengröße des übergebenen Arrays
		correctClassifications = 0; //Anzahl der korrekt klassifizierten Beispile wird auf 0 gesetzt
		while (correctClassifications < correct) //Solange nicht die Anzahl der richtig zu klassifizierenden Beispiele gelernt wurde, wird folgendes ausgeführt:
		{
			// lernen aller Datensätze, Zeile um Zeile mit den Gewichten der zeilen, die bereits gelernt wurden
			for (nrZeile = 1; nrZeile < maxZeilen; nrZeile++) //Zählschleife, die das Zahlenarray mit den Beispielen zeilenweise/nach Beispielen durchgeht
			{
				iterations++; //Erhöhen der Zählvariable für die Anzahl der Iterationen

				for (nrSpalte = 0; nrSpalte < maxSpalten - 1; nrSpalte++) //Zählschleife, die das Zahlenarray mit den Beispielen spaltenweise/nach Attributen durchgeht
					NN.setInput(nrSpalte, arr[nrSpalte][nrZeile]); //Setzen der Eingangsneuronen auf den Wert des beispiels un Attributwert

				//lernen für einen Datensatz solange bis der Fehler kleiner epsilon ist
				learned = false;
				t[0] = arr[maxSpalten - 1][nrZeile]; // setzen des teachers, 1 Neoron
				while (!learned) //solange dieses eine Bespile nicht gelernt wurde...
				{
					NN.apply(); //Feed-forward Durchrechnen des NN
					o[0] = NN.getOutput(0); // Berechnen des Ausgangs und Speichern im Array "o" welches die Werte der Ausgangsneuronen beinhaltet, 1. Neuron
					error = NN.energy(t, o, 1); //Berechnen des Fehlers. 1 = 1 Ausgangsneuron
					
					// Falls der Fehler größer als epsilon ist, wird der Backpropagation-Algorhitmus angewandt, sonst wird "learned" auf wahr gesetzt -> Dieses Beispiel wurde gelernt
					if (error > NN.getEpsilon())
						NN.backpropagate(t);
					else
						learned = true;
				}
			}

			// Bewertung, ob in Summe die Anzahl der richtig zu klassifizirenden Beispiele erreicht wurde. Falls nicht wird noch einmal Trainiert für alle Beispiele (Einstieg oben)
			correctClassifications = 0; //Anzahl der korrekt klassifizierten Beispile wird auf 0 gesetzt
			total_error = 0.0f; //Gesamtfehler (für alle Beispile) wird auf 0 gesetzt
			for (nrZeile = 1; nrZeile< maxZeilen; nrZeile++) //Zählschleife, die das Zahlenarray mit den Beispielen zeilenweise/nach Beispielen durchgeht
			{
				for (nrSpalte = 0; nrSpalte < maxSpalten - 1; nrSpalte++) //Zählschleife, die das Zahlenarray mit den Beispielen spaltenweise/nach Attributen durchgeht
					NN.setInput(nrSpalte, arr[nrSpalte][nrZeile]); //Setzen der Eingangsneuronen auf den Wert des beispiels un Attributwert

				NN.apply(); //Feed-forward Durchrechnen des NN
				o[0] = NN.getOutput(0); // Berechnen des Ausgangs und Speichern im Array "o" welches die Werte der Ausgangsneuronen beinhaltet, 1. Neuron
				t[0] = arr[maxSpalten - 1][nrZeile]; // Setzen des Lehrers welcher in der letzten Spalte des Zahlenarrays steht und Spiechern im Array t (=teacher), 1. Neuron

				error = NN.energy(t, o, 1); //Berechnen des Fehlers. 1 = 1 Ausgangsneuron
				total_error += error; //Aufsummieren des Einzelfehlers zu einem Gesamtfehler

				// Falls der Fehler des aktuellen Beispiels zu groß ist, wird es als nicht richtig klassifiziert betrachtet
				if (error < NN.getEpsilon()) 
					correctClassifications++;
			}

			// Ausgabe jeder Iteration
			printf("[%4d]>> Korrekte: %2d Gesamtfehler : %5.5f\n", iterations / maxZeilen, correctClassifications, total_error); // Ausgabe des Lernvorgangs
		}
		printf("Iterationen: %d\n ", iterations / maxZeilen); //Ausgabe der insgesamt benötigten Iterationen
	}

	template<int x, int y> //Template zur vereinfachten Übergabe der 2D Arrays
	// Funktion zum Testen des trainierten NN
	// Übergabeparameter: arr=2D-Zahlenarray, welches die zu testenden Daten beinhaltet
	void testen(float(&arr)[x][y])
	{
		nrSpalte = 0, nrZeile = 0; //Setzen der Zählvariablen für Spalte und zeile auf 0 als Initialwert
		maxSpalten = sizeof(arr) / sizeof arr[0]; //Bestimmen der Spaltengröße des übergebenen Arrays
		maxZeilen = sizeof arr[0] / sizeof(float); //Bestimmen der Zeilengröße des übergebenen Arrays
		correctClassifications = 0; //Anzahl der korrekt klassifizierten Beispile wird auf 0 gesetzt
		
		for (nrZeile = 1; nrZeile < maxZeilen; nrZeile++) //Zählschleife, die das Zahlenarray mit den Beispielen zeilenweise/nach Beispielen durchgeht
		{
			printf("IN %1.0i: ",nrZeile); //Ausgabe der Zeilennummer
			float zelle = 0; // Lokale Variable, die den Wert der aktuellen Zelle speichert
			for (nrSpalte = 0; nrSpalte < maxSpalten - 1; nrSpalte++) //Zählschleife, die das Zahlenarray mit den Beispielen spaltenweise/nach Attributen durchgeht
			{
				zelle = arr[nrSpalte][nrZeile]; //Auslesen und Speichern des aktuellen Zellenwertes bzw. Attributmerkmals
				NN.setInput(nrSpalte, zelle); //Setzen der Eingangsneuronen
				printf("%1.2f ", zelle); //Ausgabe des Wertes der Zelle
			}
			NN.apply(); //Feed-forward Durchrechnen des NN

			double out = NN.getOutput(0); // Berechnen des Ausgangs und Speichern ind er lokalen Variable "out"
			printf("OUT: %1.0f ", out); //Ausgabe des berechneten Wertes des Ausgangneurons
			bool korr = false; //Lokale boolsche Variable, die besat, ob das aktuelle Besipiel aus den Testdaten richtig klassifiziert wurde

			//Falls das Beispile richtig klassifiziert wurde, wird die Anzahl der richtig klassifizierten erhöht und "korr" auf wahr gesetzt
			if ((out > 0.9 && arr[maxSpalten - 1][nrZeile] == 1) || (out < 0.9 && arr[maxSpalten - 1][nrZeile] == 0))
			{
				correctClassifications++;
				korr = true;
			}
			if (korr) 
				printf("-> RICHTIG\n"); //Falls das Beispiel richtig klassifiziert wurde, wird "RICHTIG" ausgegeben
			else
				printf("-> FALSCH\n"); //Falls das Beispiel flasch klassifiziert wurde, wird "FALSCH" ausgegeben
		}
		//Am Ende wird die Gesamtanzahl der richtig und falsch Klasifizierten Beispiele und die Quote ausgegeben
		printf("\nAnzahl Richtige: %1.0i \n", correctClassifications);
		printf("Anzahl Falsche: %1.0i \n", maxZeilen - correctClassifications - 1);
		printf("Klassifizierungsquote: %2.2f %%\n", (float)correctClassifications / ((float)maxZeilen - 1)*100);
	}
};

int main() //Hauptroutine 
{
	//Konsoleneinstellung
	HWND console = GetConsoleWindow();
	RECT r;
	GetWindowRect(console, &r); //stores the console's current dimensions
	MoveWindow(console, r.left, r.top, 1100, 500, TRUE); // 1100 breite, 500 höhe
	
	EigeneMethoden meineMethoden; //Erstelln eines Objektes mit eigenen Methoden

	// Anlegen der Arrays, die die Testdaten und lerndaten und Attribute beinhalten als Test und die Testdaten und Lerndaten als Zahlenwert
	// Zur Initialisierung werden Konstanten verwendet, die die Größe der Array bestimmen
	const static int maxSpalten = 23, maxZeilenLerndaten = 401, maxZeilenTestdaten = 1002, maxZeilenAttribute = 19; //Dimensionen der Datensätze
	string lerndaten[maxSpalten][maxZeilenLerndaten]; //Array mit Lerndaten als Zeichenkette
	float lerndatenZahl[maxSpalten][maxZeilenLerndaten]; //Array mit Lerndaten als skalierte Zahlenwerte
	string testdaten[maxSpalten][maxZeilenTestdaten]; //Array mit Testdaten als Zeichenkette
	float testdatenZahl[maxSpalten][maxZeilenTestdaten]; //Array mit Testdaten als skalierte Zahlenwerte
	string attribute[maxSpalten][maxZeilenAttribute]; //Array mit Attributen als Zeichenkette

	//Einlesen von Daten in die jweiligen String-Arrays
	cout << "Einlesen von Trainingsdaten" << endl;
	meineMethoden.CSVeinlesen("..\\Testdaten\\Wohnungskartei_Muster_Master_4_S_teach.csv", lerndaten);
	cout << "Einlesen von Testdaten" << endl;
	meineMethoden.CSVeinlesen("..\\Testdaten\\Wohnungskartei_Muster_Master_5_S.csv", testdaten);
	cout << "Einlesen von Attributen" << endl;
	meineMethoden.CSVeinlesen("..\\Testdaten\\Attribute.csv", attribute);

	//Überführen der Daten aus den String-Arrays in skalierte Zahlenwerte
	cout << "Ueberfuehren der Trainigsdaten in skalierte Zahlen" << endl;
	meineMethoden.datenDig(lerndaten, attribute, lerndatenZahl);
	cout << "Ueberfuehren der Testdaten in skalierte Zahlen" << endl;
	meineMethoden.datenDig(testdaten, attribute, testdatenZahl);

//******************************************************************************************************
#pragma region Schritt 1: Training mit Lerndaten, Test mit Testdaten
	//Schritt 1: Training mit Lerndaten, Test mit Testdaten
	//Übergabewerte: 0.005f=Fehler, 0.02f=Lernrate, maxZeilenLerndaten - 1=Anzal der richtig zu klassifizierendne Beispiele
	//maxSpalten - 1=Anzahl der Eingangsneuronen (22), (maxSpalten - 1) * 2=Anzahl der Eingangsneuronen mal zwei (44), 1=1 Ausgangsneuron
	meineMethoden.initNN(0.005f, 0.02f, maxZeilenLerndaten - 1, maxSpalten - 1, 30, 1); //22 Neuronen Eingang, 44 Neuronen Mitte, 1 Neuronen Ausgang
	cout << "\nWeiter mit Training mit Lerndaten -> ENTER" << endl; 
	cin.get(); // Warte auf Eingabe, damit Konsole nicht gleich verschwindet
	meineMethoden.lernen(lerndatenZahl); //NN Trainieren mit Lerndaten

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
	//Übergabewerte: 0.005f=Fehler, 0.02f=Lernrate, maxZeilenTestdaten- 1=Anzal der richtig zu klassifizierendne Beispiele
	//maxSpalten - 1=Anzahl der Eingangsneuronen (22), (maxSpalten - 1) * 2=Anzahl der Eingangsneuronen mal zwei (44), 1=1 Ausgangsneuron
	meineMethoden.initNN(0.005f, 0.02f, maxZeilenTestdaten- 1, maxSpalten - 1, (maxSpalten - 1) * 2, 1); //22 Neuronen Eingang, 44 Neuronen Mitte, 1 Neuronen Ausgang
	cout << "\nWeiter mit Training mit Testdaten -> ENTER" << endl;
	cin.get(); // Warte auf Eingabe, damit Konsole nicht gleich verschwindet
	meineMethoden.lernen(testdatenZahl); //NN Trainieren mit Testdaten

	//NN Test mit Testdaten
	cout << "\nWeiter mit Test mit Testdaten -> ENTER" << endl;
	cin.get(); // Warte auf Eingabe, damit Konsole nicht gleich verschwindet
	meineMethoden.testen(testdatenZahl);

	//NN Test mit Lerndaten
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
