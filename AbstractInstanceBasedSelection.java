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

public abstract class AbstractInstanceBasedSelection extends AbstractClassifierSelection{

	public int baseClassifiersSize; // baseClassifiersSize = classifiers.size()/instances.size()
	public ArrayList<Instances> instancesArray;
	public ArrayList<AbstractDiversityMeasure> dms;
	public ArrayList<AbstractClassifier> classifiers;

	public AbstractInstanceBasedSelection(ArrayList<AbstractDiversityMeasure> dms){ //can be null
		this.dms = dms;
		instancesArray = null;
		classifiers = null;
	}

	public abstract Boolean[] select() throws Exception;

	public void setClassifiers(ArrayList<AbstractClassifier> classifiers){
		this.classifiers = classifiers;
	}

	public void setInstances(ArrayList<Instances> instancesArray){
		this.instancesArray = instancesArray;
	}

	public Boolean check(){
		if(classifiers != null && instancesArray != null)
			return true;
		else
			return false;
	}

}