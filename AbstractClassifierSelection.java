/* 
	@author Victor Lúcio
	Federal University of São Paulo - ICT/UNIFESP
	"A Classifiers Fusion System Applied to Fenology"

	Multiple Classifier System
	from: Faria, Fabio "A Framework for Pattern Classifier Selection and Fusion", 2014
*/

import weka.classifiers.*;
import weka.core.*;
import java.util.*;

public abstract class AbstractClassifierSelection implements SelectionMethod{

	public abstract ArrayList<Boolean> select() throws Exception;

	public abstract void setClassifiers(ArrayList<AbstractClassifier> classifiers);

}