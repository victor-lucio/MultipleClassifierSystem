/* 
	@author Victor Lúcio
	Federal University of São Paulo - ICT/UNIFESP
	"A Classifiers Fusion System Applied to Fenology"

	Multiple Classifier System
	from: Faria, Fabio "A Framework for Pattern Classifier Selection and Fusion", 2014
*/

import java.util.*;

public abstract class AbstractFilter<T>{

	public AbstractFilter(){
	
	}

	public abstract ArrayList<T> filtrate(ArrayList<T> in); // como comparar?
}