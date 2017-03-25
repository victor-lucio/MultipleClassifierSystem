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

public class Concensus extends AbstractInstanceBasedSelection{
	
	private ArrayList<ArrayList<Boolean>> matrixV;
	private ArrayList<ArrayList<ClassifierPairStruct>> matrixDM;
	private ArrayList<ArrayList<Integer>> ranking;
	private ArrayList<Integer> histogram;
	private int firstFilter;
	private AbstractThreshold threshold;

	public Concensus(ArrayList<AbstractDiversityMeasure> dms, int firstFilter, AbstractThreshold threshold){
		super(dms);
		this.firstFilter = firstFilter;
		this.threshold = threshold;
	}

	public Boolean[] select() throws Exception{
		//if(super.check()){
			int i, j;
			ArrayList<Boolean> selected;
			Boolean[] r;
			
			makeMatrix();
			diversityCombination();
			rank();
			makeHistogram();
			selected = applyThreshold();
			r = (Boolean[]) selected.toArray();
			return r;
	}

	private void makeMatrix() throws Exception{
		matrixV = new ArrayList<ArrayList<Boolean>>();   //inicialização da matriz
	    int i, contC = 0, j = 0, k;

	    baseClassifiersSize = classifiers.size()/instancesArray.size();
	    for(i=0;i<classifiers.size();i++)
			matrixV.add(new ArrayList<Boolean>());

	    for(i=0;i<classifiers.size();i++){
			if(i % baseClassifiersSize == 0)
	            j++;

			for(k=0;k<instancesArray.get(j).size();k++){
				if((classifiers.get(i).classifyInstance(instancesArray.get(j).instance(k))) == instancesArray.get(j).instance(k).classValue())
					matrixV.get(i).add(true);
				else{
					matrixV.get(i).add(false);
				}
			}
	    }
	}

	private void diversityCombination(){
		int i, j, d;

		matrixDM = new ArrayList<ArrayList<ClassifierPairStruct>>();
		for(d=0;d<dms.size();d++){
			matrixDM.add(new ArrayList<ClassifierPairStruct>());
			for(i=0;i<matrixV.size();i++){
				for(j=i;j<matrixV.get(0).size();j++){
					matrixDM.get(d).add(new ClassifierPairStruct(dms.get(d), dms.get(d).measure(matrixV.get(i), matrixV.get(j)), i, j));
				}
			}
		}
	}

	private void rank(){
		int d, i;
		ranking = new ArrayList<ArrayList<Integer>>();
		ArrayList<ClassifierPairStruct> array;

		for(d=0;d<dms.size();d++){
			ranking.add(new ArrayList<Integer>());
			for(i=0;i<matrixDM.get(0).size();i++)
				ranking.get(d).add(i);
			array = matrixDM.get(d);
			Collections.sort(ranking.get(d), new IndexSort(array));
		}
	}

	private void makeHistogram(){
		histogram = new ArrayList<Integer>(Collections.nCopies(classifiers.size(), 0));
		int i,j;

		for(i=0;i<dms.size();i++){
			ranking.get(i).subList(firstFilter+1, ranking.get(i).size());
			for(j=0;j<ranking.get(i).size();j++){
				histogram.add(matrixDM.get(i).get(ranking.get(i).get(j)).c1, histogram.get(matrixDM.get(i).get(ranking.get(i).get(j)).c1)+1);
				histogram.add(matrixDM.get(i).get(ranking.get(i).get(j)).c2, histogram.get(matrixDM.get(i).get(ranking.get(i).get(j)).c2)+1);
			}
		}
	}

	private ArrayList<Boolean> applyThreshold(){
		ArrayList<Boolean> selected;

		selected = threshold.check(histogram);
		return selected;
	}
}