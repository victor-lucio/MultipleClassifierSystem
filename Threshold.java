/* 
	@author Victor Lúcio
	Federal University of São Paulo - ICT/UNIFESP
	"A Classifiers Fusion System Applied to Fenology"

	Multiple Classifier System
	from: Faria, Fabio "A Framework for Pattern Classifier Selection and Fusion", 2014
*/

import java.util.*;

public class Threshold<T extends Comparable> extends AbstractSelectionMethod<T>{

	private T t;
	private ArrayList<Boolean> out;
	
	public Threshold(T t){
		this.t = t;
	}


	public void setArray(ArrayList<T> array){
		out = new ArrayList<Boolean>();
		for(T element : array){
			if(element.compareTo(t) >= 0)
				out.add(true);
			else
				out.add(false);
		}
	}


	public ArrayList<Boolean> select(){
		return out;
	}
}