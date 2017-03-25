/* 
	@author Victor Lúcio
	Federal University of São Paulo - ICT/UNIFESP
	"A Classifiers Fusion System Applied to Fenology"

	Multiple Classifier System
	from: Faria, Fabio "A Framework for Pattern Classifier Selection and Fusion", 2014
*/

import java.util.*;

public abstract class AbstractThreshold{

	public AbstractThreshold(){
	
	}

	public abstract ArrayList<Boolean> check(ArrayList<Integer> in); // como comparar?
}