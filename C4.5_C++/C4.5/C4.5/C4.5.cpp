// ML_Projekt_MH2.cpp: Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>

using namespace std;

class MatrixCls
{
private:
	vector < vector < string > > Matrix;
public:
	MatrixCls(string Data_File)
	{
		Matrix.erase(Matrix.begin(), Matrix.end());
		ifstream Data(Data_File);
		string Line;
		string Item;
		vector < string > Row;
		while (!Data.eof())
		{
			getline(Data, Line);
			istringstream Iss(Line);
			while (Iss.good())
			{
				//Edit MH//
				//getline(Iss, Item, ',');
				getline(Iss, Item, ';');

				//Edit MH//
				//Item.erase(remove(Item.begin(), Item.begin() + 2, ' '), Item.begin() + 2);
				//Item.erase(remove(Item.end() - 1, Item.end(), ' '), Item.end());

				Row.push_back(Item);
			}

			if (Line.length())
			{
				Matrix.push_back(Row);
				Row.erase(Row.begin(), Row.end());
			}
		}
		Data.close();
	}

	MatrixCls() { };
	~MatrixCls() { };

	string Element(int i, int j)
	{
		return Matrix[i][j];
	}

	int SizeX()		//Anzahl der Attribute (Spalten) der Trainingsdaten
	{
		return Matrix[0].size();
	}

	int SizeY()
	{
		return Matrix.size();
	}

	vector < string > GetVarKinds()
	{
		vector < string > Kinds;
		int j;
		for (j = 0; j < SizeX() - 1; j++)
		{
			Kinds.push_back(Matrix[0][j]);
		}
		return Kinds;
	}

	vector < string > GetAttributes()
	{
		vector < string > Attributes;
		int j;
		for (j = 0; j < SizeX() - 1; j++)
		{
			Attributes.push_back(Matrix[1][j]);
		}
		return Attributes;
	}

	vector < string > GetScores()
	{
		vector < string > Scores;
		for (int i = 2; i < SizeY(); i++)
		{
			Scores.push_back(Matrix[i][SizeX() - 1]);
		}
		return Scores;
	}

	int GetAttributeIndex(string The_Attribute)
	{
		{
			int Index = 0;
			for (int j = 0; j < SizeX(); j++)
			{
				if (Matrix[1][j].compare(The_Attribute) == 0)
				{
					Index = j;
					break;
				}
			}
			return Index;
		}
	}

	vector < string > GetAttributeValues(string The_Attribute)
	{
		vector < string > Values;
		int Index = GetAttributeIndex(The_Attribute);
		for (int i = 2; i < SizeY(); i++)
		{
			Values.push_back(Matrix[i][Index]);
		}
		return Values;
	}

	vector < string > GetUniqueAttributeValues(string The_Attribute)
	{
		vector < string > Values = GetAttributeValues(The_Attribute);
		sort(Values.begin(), Values.end());
		Values.erase(unique(Values.begin(), Values.end()), Values.end());
		return Values;
	}

	map < string, vector < string > > GetAttributeValuesScores(string The_Attribute)
	{
		int i, k;
		int Index = GetAttributeIndex(The_Attribute);
		map < string, vector < string > > Attribute_Values_Scores;
		vector < string > Attribute_Values = GetUniqueAttributeValues(The_Attribute);
		vector < string > Row;
		for (k = 0; k < Attribute_Values.size(); k++)
		{
			for (i = 2; i < SizeY(); i++)
			{
				if (Matrix[i][Index].compare(Attribute_Values[k]) == 0)
				{
					Row.push_back(Matrix[i][SizeX() - 1]);
				}
			}
			Attribute_Values_Scores[Attribute_Values[k]] = Row;
			Row.erase(Row.begin(), Row.end());
		}
		return Attribute_Values_Scores;
	}

	vector < string > SortAttributeValues(string The_Attribute)
	{
		vector < string > Values = GetAttributeValues(The_Attribute);
		string Temp;
		for (int i = 0; i < Values.size() - 1; i++)
		{
			for (int j = i + 1; j < Values.size(); j++)
			{
				if (stod(Values[i]) - stod(Values[j]) > 1.e-8)
				{
					Temp = Values[i];
					Values[i] = Values[j];
					Values[j] = Temp;
				}
			}
		}
		return Values;
	}

	vector < string > SortScoreValues(string The_Attribute)
	{
		vector < string > Values = GetAttributeValues(The_Attribute);
		vector < string > Scores = GetScores();
		string Temp;

		for (int i = 0; i < Values.size() - 1; i++)
		{
			for (int j = i + 1; j < Values.size(); j++)
			{
				if (stod(Values[i]) - stod(Values[j]) > 1.e-8)
				{
					Temp = Values[i];
					Values[i] = Values[j];
					Values[j] = Temp;

					Temp = Scores[i];
					Scores[i] = Scores[j];
					Scores[j] = Temp;
				}
			}
		}
		return Scores;
	}

	vector < string > GetBisectNodes(string The_Attribute)
	{
		vector < string > Bisect_Nodes;
		vector < string > SortedValues = SortAttributeValues(The_Attribute);
		vector < string > SortedScores = SortScoreValues(The_Attribute);
		for (int i = 0; i < SortedValues.size() - 1; i++)
		{
			if (abs(stod(SortedValues[i]) - stod(SortedValues[i + 1])) > 1.e-8 & SortedScores[i].compare(SortedScores[i + 1]) != 0)
			{
				Bisect_Nodes.push_back(to_string((stod(SortedValues[i]) + stod(SortedValues[i + 1])) / 2.));
			}
		}
		return Bisect_Nodes;
	}

	map < string, vector < string > > GetAttributeBisectParts(string The_Attribute, string Bisect_Node)
	{
		map < string, vector < string > > Bisect_Parts;
		vector < string > SortedValues = SortAttributeValues(The_Attribute);
		vector < string > SortedScores = SortScoreValues(The_Attribute);
		vector < string > Row_1, Row_2, Row_3, Row_4;
		for (int i = 0; i < SortedValues.size(); i++)
		{
			if (stod(SortedValues[i]) - stod(Bisect_Node) < -1.e-8)
			{
				Row_1.push_back(SortedScores[i]);
				Row_3.push_back(SortedValues[i]);
			}
			else {
				Row_2.push_back(SortedScores[i]);
				Row_4.push_back(SortedValues[i]);
			}
		}
		Bisect_Parts["Lower_Scores"] = Row_1;
		Bisect_Parts["Upper_Scores"] = Row_2;
		Bisect_Parts["Lower_Values"] = Row_3;
		Bisect_Parts["Upper_Values"] = Row_4;
		return Bisect_Parts;
	}

	MatrixCls operator()(MatrixCls A_Matrix, string The_Attribute, string The_Value, string Bisect_Node = "")
	{
		Matrix.erase(Matrix.begin(), Matrix.end());
		int i, j;
		int Index = A_Matrix.GetAttributeIndex(The_Attribute);
		vector < string > Kinds = A_Matrix.GetVarKinds();
		vector < string > Row;

		if (Kinds[Index].compare("Discrete") == 0)
		{
			for (i = 0; i < 2; i++)
			{
				for (j = 0; j < A_Matrix.SizeX(); j++)
				{
					if (A_Matrix.Element(1, j).compare(The_Attribute) != 0)
					{
						Row.push_back(A_Matrix.Element(i, j));
					}
				}
				if (Row.size() != 0)
				{
					Matrix.push_back(Row);
					Row.erase(Row.begin(), Row.end());
				}
			}

			for (i = 2; i < A_Matrix.SizeY(); i++)
			{
				for (j = 0; j < A_Matrix.SizeX(); j++)
				{
					if (A_Matrix.Element(1, j).compare(The_Attribute) != 0 & A_Matrix.Element(i, Index).compare(The_Value) == 0)
					{
						Row.push_back(A_Matrix.Element(i, j));
					}
				}

				if (Row.size() != 0)
				{
					Matrix.push_back(Row);
					Row.erase(Row.begin(), Row.end());
				}
			}
			return *this;
		}

		else if (Kinds[Index].compare("Continuous") == 0)
		{
			for (i = 0; i < 2; i++)
			{
				for (j = 0; j < A_Matrix.SizeX(); j++)
				{
					Row.push_back(A_Matrix.Element(i, j));
				}
				if (Row.size() != 0)
				{
					Matrix.push_back(Row);
					Row.erase(Row.begin(), Row.end());
				}
			}

			if (The_Value.compare("Lower_Values") == 0)
			{
				for (i = 2; i < A_Matrix.SizeY(); i++)
				{
					for (j = 0; j < A_Matrix.SizeX(); j++)
					{
						if (stod(A_Matrix.Element(i, Index)) - stod(Bisect_Node) < -1.e-8)
						{
							Row.push_back(A_Matrix.Element(i, j));
						}
					}

					if (Row.size() != 0)
					{
						Matrix.push_back(Row);
						Row.erase(Row.begin(), Row.end());
					}
				}
			}
			else if (The_Value.compare("Upper_Values") == 0)
			{
				for (i = 2; i < A_Matrix.SizeY(); i++)
				{
					for (j = 0; j < A_Matrix.SizeX(); j++)
					{
						if (stod(A_Matrix.Element(i, Index)) - stod(Bisect_Node) > 1.e-8)
						{
							Row.push_back(A_Matrix.Element(i, j));
						}
					}

					if (Row.size() != 0)
					{
						Matrix.push_back(Row);
						Row.erase(Row.begin(), Row.end());
					}
				}
			}
			return *this;
		}
	}

	void Display()
	{
		int i, j;
		for (i = 0; i < Matrix.size(); i++)
		{
			for (j = 0; j < Matrix[0].size(); j++)
			{
				cout << Matrix[i][j] << "    \t";
			}
			cout << endl;
		}
	}
};

vector < string > UniqueValues(vector < string > A_String)
{
	sort(A_String.begin(), A_String.end());
	A_String.erase(unique(A_String.begin(), A_String.end()), A_String.end());
	return A_String;
}

string FrequentValues(vector < string > A_String)
{
	vector < string > Unique_Values = UniqueValues(A_String);
	//Edit MH //int Count[Unique_Values.size()] = { 0 };
	vector < int > Count(Unique_Values.size());
	for (int i = 0; i < A_String.size(); i++)
	{
		for (int j = 0; j < Unique_Values.size(); j++)
		{
			if (A_String[i].compare(Unique_Values[j]) == 0)
			{
				Count[j] = Count[j] + 1;
			}
		}
	}

	int Max_Count = 0, Max_Index;
	for (int i = 0; i < Unique_Values.size(); i++)
	{
		if (Count[i] > Max_Count)
		{
			Max_Count = Count[i];
			Max_Index = i;
		}
	}
	return Unique_Values[Max_Index];
}

double ComputeScoreEntropy(vector < string > Scores)
{
	vector < string > Score_Range = UniqueValues(Scores);
	if (Score_Range.size() == 0)
	{
		return 0.;
	}
	else
	{
		double TheEntropy = 0.;
		int i, j;
		//Edit MH // int Count[Score_Range.size()] = { 0 };
		vector< int > Count(Score_Range.size());

		for (i = 0; i < Scores.size(); i++)
		{
			for (j = 0; j < Score_Range.size(); j++)
			{
				if (Scores[i].compare(Score_Range[j]) == 0)
				{
					Count[j] = Count[j] + 1;
				}
			}
		}

		double Temp_Entropy;
		double Temp_P;
		for (j = 0; j < Score_Range.size(); j++)
		{
			Temp_P = (double)Count[j] / (double)(Scores.size());
			Temp_Entropy = -Temp_P * log(Temp_P) / log(2.);
			TheEntropy = TheEntropy + Temp_Entropy;
		}
		return TheEntropy;
	}
}

double ComputeAttributeEntropy(MatrixCls Remain_Matrix, string The_Attribute)
{
	vector < string > Values = Remain_Matrix.GetAttributeValues(The_Attribute);
	return ComputeScoreEntropy(Values);
}

double ComputeAttributeEntropyGain(MatrixCls Remain_Matrix, string The_Attribute, string Bisect_Node = "")
{
	int Index = Remain_Matrix.GetAttributeIndex(The_Attribute);
	vector < string > Kinds = Remain_Matrix.GetVarKinds();
	double Original_Entropy = 0., Gained_Entropy = 0.;
	vector < string > Scores = Remain_Matrix.GetScores();
	Original_Entropy = ComputeScoreEntropy(Scores);

	if (Kinds[Index].compare("Discrete") == 0)
	{
		map < string, vector < string > > Values_Scores = Remain_Matrix.GetAttributeValuesScores(The_Attribute);
		vector < string > Values = Remain_Matrix.GetUniqueAttributeValues(The_Attribute);

		double After_Entropy = 0.;
		double Temp_Entropy;
		vector < string > Temp_Scores;
		int i, j;
		for (i = 0; i < Values.size(); i++)
		{
			Temp_Scores = Values_Scores[Values[i]];
			Temp_Entropy = ComputeScoreEntropy(Temp_Scores)*(double)Temp_Scores.size() / (double)Scores.size();
			After_Entropy = After_Entropy + Temp_Entropy;
		}
		Gained_Entropy = Original_Entropy - After_Entropy;
		return Gained_Entropy;
	}

	if (Kinds[Index].compare("Continuous") == 0)
	{
		map < string, vector < string > > Parts = Remain_Matrix.GetAttributeBisectParts(The_Attribute, Bisect_Node);
		double LowerLen = Parts["Lower_Scores"].size();
		double UpperLen = Parts["Upper_Scores"].size();
		double Len = LowerLen + UpperLen;
		double After_Entropy, Gained_Entropy;
		After_Entropy = LowerLen / Len * ComputeScoreEntropy(Parts["Lower_Scores"]) + UpperLen / Len * ComputeScoreEntropy(Parts["Upper_Scores"]);
		Gained_Entropy = Original_Entropy - After_Entropy;
		return Gained_Entropy;
	}
}

double GainRatio(MatrixCls Remain_Matrix, string The_Attribute, string Bisect_Node = "")
{
	double Attribute_Entropy = ComputeAttributeEntropy(Remain_Matrix, The_Attribute);
	double Attribute_Entropy_Gain = ComputeAttributeEntropyGain(Remain_Matrix, The_Attribute, Bisect_Node);
	return Attribute_Entropy_Gain / Attribute_Entropy;
}

class TreeCls
{
public:
	string Node;
	string Branch;
	vector < TreeCls * > Child;
	TreeCls();
	TreeCls * BuildTree(TreeCls * Tree, MatrixCls Remain_Matrix);
	void Display(int Depth);
	string Temp_TestTree(vector < string > Kinds, vector < string > Attributes, vector < string > Value, vector < string > Score_Range);
	vector < string > TestTree(MatrixCls Data_Matrix);
};


class MyMethods
{
public:
	double GetLearnQuality(MatrixCls TestMatrix, vector <string> resultsFromTree, string valueInterpretationForUndefined) {
		int i;
		int correctCount = 0;
		double correctPercent = 0.0;
		string temp;

		for (i = 0; i < resultsFromTree.size(); i++) {

			//Wenn 'notDefined' lt. Baum (Attribut im Baum defininert, aber Attributwert nicht aufgeführt) wird der Fall lt. Angabe in 'valueInterpretationForUndefined' gehandhabt
			if (resultsFromTree[i].compare("notDefined") == 0) {
				temp = valueInterpretationForUndefined;
			}
			else {
				temp = resultsFromTree[i];
			}

			if (temp.compare(TestMatrix.Element(i + 2, 22)) == 0) {
				correctCount++;
			}
			
			temp = "";
		}

		correctPercent = 100 * correctCount / resultsFromTree.size();
		
		return correctPercent;
	}
};


string TreeCls::Temp_TestTree(vector < string > Kinds, vector < string > Attributes, vector < string > Value, vector < string > Score_Range)
{
	//Testen ob der Node des Baums schon direkt mit der Bewertung übereinstimmt (z.B. wenn Node 'Yes' wäre)
	for (int i = 0; i < Score_Range.size(); i++)
	{
		if (this->Node.compare(Score_Range[i]) == 0)		
		{
			return this->Node;
		}
	}

	//Mit restlichem Baum prüfen: Was wäre das Ergebnis für den aktuellen Testfall lt. dem Baum?
	for (int i = 0; i < Attributes.size(); i++)
	{
		if (this->Node.compare(Attributes[i]) == 0)
		{
			if (Kinds[i].compare("Discrete") == 0)
			{
				for (int j = 0; j < this->Child.size(); j++)
				{
					if ((this->Child[j])->Branch.compare(Value[i]) == 0)
					{
						for (int k = 0; k < Score_Range.size(); k++)
						{
							if ((this->Child[j])->Node.compare(Score_Range[k]) == 0)	//Ist das Ergebnis lt. Baum auch wirklich ein valider Score?
							{
								return (this->Child[j])->Node;
							}
						}

						vector < string > New_Kinds;
						vector < string > New_Attributes;
						vector < string > New_Value;
						for (int l = 0; l < Attributes.size(); l++)
						{
							if (l != i)
							{
								New_Kinds.push_back(Kinds[l]);
								New_Attributes.push_back(Attributes[l]);
								New_Value.push_back(Value[l]);
							}
						}
						return (this->Child[j])->Temp_TestTree(New_Kinds, New_Attributes, New_Value, Score_Range);
					}
					else {
						//EditMH // Für den Fall, dass der Wert aus den Tests nicht im Baum vorkommen, jedoch das Attribut, wird standardmäßig "notDefined" zurückgegeben.
						if (j == this->Child.size()-1) {
							return "notDefined";
						}
					}
				}
			}

			else if (Kinds[i].compare("Continuous") == 0)
			{
				string Threshold = (this->Child[0])->Branch;
				Threshold.erase(remove(Threshold.begin(), Threshold.begin() + 3, '<'), Threshold.begin() + 3);
				Threshold.erase(remove(Threshold.begin(), Threshold.begin() + 3, ' '), Threshold.begin() + 3);
				if (stod(Value[i]) - stod(Threshold) < -1.e-8)
				{
					for (int k = 0; k < Score_Range.size(); k++)
					{
						if ((this->Child[0])->Node.compare(Score_Range[k]) == 0)
						{
							return (this->Child[0])->Node;
						}
					}
					return (this->Child[0])->Temp_TestTree(Kinds, Attributes, Value, Score_Range);
				}
				else if (stod(Value[i]) - stod(Threshold) > 1.e-8)
				{
					for (int k = 0; k < Score_Range.size(); k++)
					{
						if ((this->Child[1])->Node.compare(Score_Range[k]) == 0)
						{
							return (this->Child[1])->Node;
						}
					}
					return (this->Child[1])->Temp_TestTree(Kinds, Attributes, Value, Score_Range);
				}
			}
		}
	}
}

vector < string > TreeCls::TestTree(MatrixCls Data_Matrix)
{

	int Lines_Number = Data_Matrix.SizeY() - 2;		//Anzahl der Daten
	vector < string > Test_Scores;
	int i, j, k, l, m;
	vector < string > Kinds = Data_Matrix.GetVarKinds();	//Attributwertearten aus Trainingsdaten lesen (z.B. discrete / continuous)
	vector < string > Attributes = Data_Matrix.GetAttributes();		//Attributkategorien aus Trainingsdaten lesen (z.B. Zimmerzahl, Stockwerk, etc.)
	vector < string > Attributes_Value;
	vector < string > Score_Range = UniqueValues(Data_Matrix.GetScores());

	for (i = 0; i < Lines_Number; i++)		//Iteration über alle Datensätze
	{
		for (j = 0; j < Data_Matrix.SizeX() - 1; j++)		//Iteration über alle Attribute
		{
			Attributes_Value.push_back(Data_Matrix.Element((i + 2), j));	//Attributwerte von aktuellem Trainingsbsp. auslesen
		}

		string Temp_Score;
		Temp_Score = Temp_TestTree(Kinds, Attributes, Attributes_Value, Score_Range);		//Was ist der Attributes_Value lt. Baum für aktuellen Testfall?
		Test_Scores.push_back(Temp_Score);								//Ergebnis aus Baum merken
		Attributes_Value.erase(Attributes_Value.begin(), Attributes_Value.end());		//Attributes_Value leeren
	}
	return Test_Scores;
}

TreeCls::TreeCls()
{
	Node = "";
	Branch = "";
}

TreeCls * TreeCls::BuildTree(TreeCls * Tree, MatrixCls Remain_Matrix)
{
	if (Tree == NULL)
	{
		Tree = new TreeCls();
	}

	vector < string > Unique_Scores = UniqueValues(Remain_Matrix.GetScores());
	if (Unique_Scores.size() == 1)
	{
		Tree->Node = Unique_Scores[0];
		return Tree;
	}

	//vector < string > Scores = Remain_Matrix.GetScores();
	//if(Scores.size() <= 5)
	//{
	//  Tree->Node = FrequentValues(Scores);
	//  return Tree;
	//}

	double Gain_Ratio = 0, Entropy_Gain = 0;
	double Temp_Gain_Ratio, Temp_Entropy_Gain;
	string Max_Attribute;
	string Max_Bisect_Node = "";
	int Max_Attribute_Index = 0;
	//string Max_Bisect_Node_Index = 0;
	vector < string > Attributes = Remain_Matrix.GetAttributes();
	vector < string > Kinds = Remain_Matrix.GetVarKinds();
	int i, j;
	for (i = 0; i < Attributes.size(); i++)
	{
		//Berechne Temp_Gain_Ratio für aktuelles Attribut (Unterscheidung Discrete / Continuous Werte)
		if (Kinds[i].compare("Discrete") == 0)
		{
			Temp_Gain_Ratio = GainRatio(Remain_Matrix, Attributes[i]);
		}
		else if (Kinds[i].compare("Continuous") == 0)
		{
			vector < string > Bisect_Nodes = Remain_Matrix.GetBisectNodes(Attributes[i]);
			for (j = 0; j < Bisect_Nodes.size(); j++)
			{
				Temp_Entropy_Gain = ComputeAttributeEntropyGain(Remain_Matrix, Attributes[i], Bisect_Nodes[j]);
				if (Temp_Entropy_Gain - Entropy_Gain > 1.e-8)
				{
					Entropy_Gain = Temp_Entropy_Gain;
					Max_Bisect_Node = Bisect_Nodes[j];
				}
			}
			Temp_Gain_Ratio = GainRatio(Remain_Matrix, Attributes[i], Max_Bisect_Node);
		}

		//Vgl. alle Temp_Gain_Ration mit Gain_Radio: Suche das Attribut mit dem größten Gain_Ratio
		if (Temp_Gain_Ratio - Gain_Ratio > 1.e-8)
		{
			Gain_Ratio = Temp_Gain_Ratio;
			Max_Attribute = Attributes[i];
			Max_Attribute_Index = i;
		}
	}

	//Attribut mit höchstem Gain_Ratio als Node in Baum eintragen
	Tree->Node = Max_Attribute;
	vector < string > Values, Branch_Values;
	if (Kinds[Max_Attribute_Index].compare("Discrete") == 0)	//Wenn Attribut diskrete Werte beinhaltet
	{
		//Alle Elemente des Attributs mit dem höchsten Gain_Ratio holen (ohne Doublikate)
		Values = Remain_Matrix.GetUniqueAttributeValues(Max_Attribute);
		Branch_Values = Values;
	}
	else if (Kinds[Max_Attribute_Index].compare("Continuous") == 0)	//Wenn Attribut continuous Werte beinhaltet
	{
		Values = { "Lower_Values", "Upper_Values" };
		string Left_Branch = "< " + Max_Bisect_Node;
		string Right_Branch = "> " + Max_Bisect_Node;
		Branch_Values = { Left_Branch, Right_Branch };
	}

	int k;
	MatrixCls New_Matrix;
	for (k = 0; k < Values.size(); k++)
	{
		New_Matrix = New_Matrix.operator()(Remain_Matrix, Max_Attribute, Values[k], Max_Bisect_Node);
		TreeCls * New_Tree = new TreeCls();
		New_Tree->Branch = Branch_Values[k];
		vector < string > New_Unique_Scores = UniqueValues(New_Matrix.GetScores());
		if (New_Unique_Scores.size() == 1)
		{
			New_Tree->Node = New_Unique_Scores[0];
		}
		else
		{
			BuildTree(New_Tree, New_Matrix);
		}
		Tree->Child.push_back(New_Tree);
	}
	return Tree;
}

void TreeCls::Display(int Depth = 0)
{
	for (int i = 0; i < Depth; i++)
	{
		cout << "\t";
	}
	if (this->Branch.compare("") != 0)
	{
		cout << this->Branch << endl;
		for (int i = 0; i < Depth + 1; i++)
		{
			cout << "\t";
		}
	}
	cout << this->Node << endl;
	for (int i = 0; i < this->Child.size(); i++)
	{
		(this->Child[i])->Display(Depth + 1);
	}
}

void DisplayVector(vector < string > The_Vector)
{
	for (int i = 0; i < The_Vector.size(); i++)
	{
		cout << The_Vector[i] << "\n";
	}
	cout << endl;
}


int main()
{
	/*
	TODO:
	- Testfälle mit Baum durchrechnen und vergleichen: Ergebnis %-Wert korrekt/falsch
	- Überprüfung ob Testdatensatz zu Teachdatensatz/Baum passt
	- Mit Testfällen lernen und Teachfälle pfüfen
	- Main auslagern in eigene cpp
	- in main(): Matrix zu Matrix1 und MatrixTest zu Matrix2 umbenennen, um dann später verschiedenartig zuweisen zu können (z.B. Matrix1 zu Test, usw.)
	*/
	MyMethods MyOwnMethods;

	cout << "********************************" << endl << "**** C4.5 Entscheidungsbaum ****" << endl << "********************************" << endl << endl;
	cout << "** Lerndaten werden eingelesen **";

	//MatrixCls Matrix("C:\\Users\\Marcel\\source\\repos\\ML_Projekt_MH2\\Golf.dat");
	MatrixCls Matrix1("..\\Testfiles\\Wohnungskartei_Muster_Master_4_S_teach_forC45_c++.csv");		//Trainingsdaten
	MatrixCls Matrix2("..\\Testfiles\\Wohnungskartei_Muster_Master_5_S_test_forC45_c++.csv");		//Testdaten
	cout << "   <<Erfolgreich>>" << endl << endl;

#pragma region V1: Mit Trainingsdaten lernen und mit Testdaten testen
	cout << "** V1 ** Mit Trainingsdaten lernen und mit Testdaten testen **" << endl << endl;
	cout << "* Erzeuge Baum" << endl <<endl;

	TreeCls * Tree1 = NULL;
	Tree1 = Tree1->BuildTree(Tree1, Matrix1);						//Berechnen des Baumes auf Basis der Trainingsdaten
	
	cout << "Entscheidungsbaum auf Basis der Trainingsdaten:" << endl << endl;
	Tree1->Display();							//Baum anzeigen
	
	cout << endl << "* Teste Baum mit Testdaten" << endl;
	vector < string > Test_Scores12 = Tree1->TestTree(Matrix2);		//Testen des Baums mit Testdaten
	
	double learnQualityInPercentage1;
	learnQualityInPercentage1 = MyOwnMethods.GetLearnQuality(Matrix2, Test_Scores12,"nein");		//'nein': Pessimistisches Vorgehen
	cout << learnQualityInPercentage1 << " % der Testdaten erfolgreich bewertet." << endl << endl;
	//DisplayVector(Test_Scores12);					//Testergebnisse lt. Baum anzeigen
	cout << "Weiter mit beliebigem Tastendruck" << endl << endl;
	cin.get();		//Warten auf Tastendruck

#pragma endregion

	
#pragma region V2: Mit Testdaten lernen und mit Trainingsdaten testen
	cout << "** V2 ** Mit Testdaten lernen und mit Trainingsdaten testen **" << endl << endl;
	cout << "* Erzeuge Baum" << endl << endl;
	
	TreeCls * Tree2 = NULL;
	Tree2 = Tree2->BuildTree(Tree2, Matrix2);		//Berechnen des Baumes auf Basis der Testdaten aus "MatrixTest"

	cout << endl << "Entscheidungsbaum auf Basis der Testdaten:" << endl << endl;
	Tree2->Display();	

	cout << endl << "* Teste Baum mit Trainingsdaten" << endl;
	vector < string > Test_Scores21 = Tree2->TestTree(Matrix1);		//Testen des Baums mit Testdaten

	double learnQualityInPercentage2;
	learnQualityInPercentage2 = MyOwnMethods.GetLearnQuality(Matrix1, Test_Scores21,"nein");
	cout << learnQualityInPercentage2 << " % der Testdaten (Trainingsdaten) erfolgreich bewertet." << endl << endl;
	//DisplayVector(Test_Scores21);					//Testergebnisse lt. Baum anzeigen


#pragma endregion
	

	
	/*
	vector < string > Test_String = { "a","b","c","d" };	//Test_String erstellen als Bsp. für Vektor
	DisplayVector(Test_String);						//Test_String anzeigen
	Test_String.erase(Test_String.begin() + 0);		//Erste 'Stelle' von Test_String löschen
	//cout << sizeof(Test_String.begin()) << endl;
	DisplayVector(Test_String);
	*/

	cout << "Klicken zum Beenden.";
	cin.get();					// Warte auf Eingabe, damit Konsole nicht gleich verschwindet
}
