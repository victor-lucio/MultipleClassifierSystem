/* 
	@author Victor Lúcio
	Federal University of São Paulo - ICT/UNIFESP
	"A Classifiers Fusion System Applied to Fenology"

	Multiple Classifier System
	from: Faria, Fabio "A Framework for Pattern Classifier Selection and Fusion", 2014
*/

import java.util.*;

public class IndexSort implements Comparator<Integer>{
	private ArrayList<ClassifierPairStruct> array;

	public IndexSort(ArrayList<ClassifierPairStruct> array){
		this.array = array;
	}

	public int compare(Integer o1, Integer o2) {
		return (array.get(o1).compareTo(array.get(o2)));
	}
}