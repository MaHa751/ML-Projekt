//package myc45;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import javax.swing.plaf.synth.SynthSplitPaneUI;

import java.lang.Math;

public class MyC45 {

	public static void main(String[] args) throws IOException {
		// .CSV Dateien definieren
		//String files[] = {"C:/Testdaten/Wohnungskartei_Muster_Master_4_S_teach_Testedit.csv"};
		String files[] = {"C:/Users/Marcel/Dropbox/machine learning/C4.5_Java/C4.5-master_alt/data_sets/lenses.csv"};
		
		Scanner scan;
		
		// Files einlesen
		scan = new Scanner(new File(files[0]));
		String headerLine = scan.nextLine();
		System.out.println("Line 1: " + headerLine);
		//String headers[]  = headerLine.split(";");
		String headers[]  = headerLine.split(",");
		
		// Class Index (Klassenzuordnung) wird in letzter Spalte erwartet
		int classIndex    = headers.length - 1;		//Spalte von class index
		int numAttributes = headers.length - 1;		//Anzahl der Attribute
		
		// store data set attributes
		Attribute attributes[] = new Attribute[numAttributes];
		for(int x = 0; x < numAttributes; x++) {
			attributes[x] = new Attribute(headers[x]);
		}
		
		// for storing classes and class count
		List<String>  classes      = new ArrayList<String>();
		List<Integer> classesCount = new ArrayList<Integer>();
		
		// store are values into respected attributes
		// along with respected classes
		int q = 2;
		while(scan.hasNextLine()){
			Val data = null;
			String inLine = scan.nextLine();
			System.out.println("Line " + q + ": " + inLine);

			q++;
			//String lineData[] = inLine.split(";");	//gewählte Zeile aufsplitten und in Array schreiben
			String lineData[] = inLine.split(",");	//gewählte Zeile aufsplitten und in Array schreiben
			
			// class in classes list einsortieren
			if(classes.isEmpty()){		//erster Eintrag in classes
				classes.add(lineData[classIndex]);
				classesCount.add(classes.indexOf(lineData[classIndex]), 1);
			}
			else{
				if(!classes.contains(lineData[classIndex])){	//wenn classes den class index von aktuell ausgewählter Zeile noch nicht enthält
					classes.add(lineData[classIndex]);
					classesCount.add(classes.indexOf(lineData[classIndex]), 1);
				}
				else {   //wenn classes den class index von aktuell ausgewählter Zeile schon enthält
					classesCount.set(classes.indexOf(lineData[classIndex]),classesCount.get(classes.indexOf(lineData[classIndex])) + 1);
				}
			}
			
			// insert data into attributes
			// Für jedes Attribut der aktuell ausgeählten Zeile mit der jew. Klasse (der Zeile) als Val anlegen und in data schreiben 
			for(int x = 0; x < numAttributes; x++){
				data = new Val(lineData[x], lineData[classIndex]);
				attributes[x].insertVal(data);
			}
		}
		int totalNumClasses = 0;
		for(int i : classesCount){
			totalNumClasses += i;
		}
		
		//Entropie IofD berechnen;  classesCount: Menge der einzelnen Klassen aus Übungsdaten (z.B. 5xYes 3xNo)
		double IofD = calcIofD(classesCount); // Set information criteria
		
		// TESTING DATA 
		Attribute age = new Attribute("age");
		
		Val inV = new Val("30","yes"); age.insertVal(inV);
		inV = new Val("30","yes"); age.insertVal(inV);
		inV = new Val("30","no"); age.insertVal(inV);
		inV = new Val("30","no"); age.insertVal(inV);
		inV = new Val("30","no"); age.insertVal(inV);
		inV = new Val("35","yes"); age.insertVal(inV);
		inV = new Val("35","yes"); age.insertVal(inV);
		inV = new Val("35","yes"); age.insertVal(inV);
		inV = new Val("35","yes"); age.insertVal(inV);
		inV = new Val("40","yes"); age.insertVal(inV);
		inV = new Val("40","yes"); age.insertVal(inV);
		inV = new Val("40","yes"); age.insertVal(inV);
		inV = new Val("40","no"); age.insertVal(inV);
		inV = new Val("40","no"); age.insertVal(inV);
		
		System.out.println(age.toString());

		List<Integer> testCount = new ArrayList<Integer>();
		testCount.add(9);
		testCount.add(5);

		double testIofD = calcIofD(testCount);
		age.setGain(testIofD,14);

		System.out.println("I of D: " + testIofD);
		System.out.println("age: " + age.gain);
		
		/*
		for(Attribute a : attributes){
			System.out.println(a.toString());
		}
		*/
		
	}
	
	public static double calcIofD(List<Integer> classesCount){
		double IofD = 0.0;
		double temp = 0.0;
		
		int totalNumClasses = 0;
		
		// Gesamtanzahl der Attribute aller Klassen berechnen (z.B. 5xYes & 3xNo => 8)
		for(int i : classesCount){
			totalNumClasses += i;
		}
		
		//Berechnen der Entropie (lt. Skript MLCI_004_Decision_Trees.pdf F.13)
		for(double d : classesCount){	// d: Anzahl Elemente von Klasse x
			temp = (-1 * (d/totalNumClasses)) * (Math.log((d/totalNumClasses)) / Math.log(2));		
			IofD += temp;
		}
		return IofD;
	}
}
