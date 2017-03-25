/* 
	@author Victor Lúcio
	Federal University of São Paulo - ICT/UNIFESP
	"A Classifiers Fusion System Applied to Fenology"

	Multiple Classifier System
	from: Faria, Fabio "A Framework for Pattern Classifier Selection and Fusion", 2014
*/

import java.util.*;

public class InterraterAgreement extends AbstractDiversityMeasure{
	
	public InterraterAgreement(){
		super();
	}

	public double measure(ArrayList<Boolean> c1, ArrayList<Boolean> c2){
		double[] v = new double[4];
		v = this.correlation(c1,c2);
		return(2*(v[0]*v[2]-v[1]*v[3])/(((v[0]+v[1])*(v[2]+v[3]))+((v[0]+v[2])*(v[1]+v[3]))));
	}

	public int compare(Double o1, Double o2){ //quanto menor, maior a diversidade
		if(o1 > o2){
			return -1;
		}else if(o1 < o2){
			return	1;
		}else{
			return 0;
		}
	}
}