/* 
	@author Victor Lúcio
	Federal University of São Paulo - ICT/UNIFESP
	"A Classifiers Fusion System Applied to Fenology"

	Multiple Classifier System
	from: Faria, Fabio "A Framework for Pattern Classifier Selection and Fusion", 2014
*/

import weka.core.Instance;
import weka.core.*;
import weka.classifiers.*;
import weka.core.Instances;
import weka.classifiers.Evaluation;
import java.util.*;
import weka.classifiers.evaluation.*;

import java.util.*;

public abstract class AbstractDiversityMeasure{
	
	public abstract double run(double a, double b, double c, double d); 

	public double[] ClassifiersCorrelation(ArrayList<Double> c1, ArrayList<Double> c2){
		int i;
		double ca = 0, cb = 0, cc = 0, cd = 0, a, b, c, d, total, aux;
		
		for(i=0;i<c1.size();i++){
			if(c1.get(i) == 1 && c1.get(i) == c2.get(i))
				ca++;
			else if(c1.get(i) == 1)
				cb++;
			else if(c2.get(i) == 1)
				cc++;
			else
				cd++;
		}

		total = ca+cb+cc+cd;
		aux = 100 / total;
		a = ca * aux;
		b = cb * aux;
		c = cc * aux;
		d = cd * aux;
		double[] r = {a,b,c,d};
		return(r);
	}

}
// a - acerto duplo, b - c1 acertou e c2 errou, c - c2 acertou e c1 errou, d - erro duplo}