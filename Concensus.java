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
	
	private ArrayList<ArrayList<Boolean>> matrixV, ranking;
	private ArrayList<ArrayList<ClassifierPairStruct>> matrixDM;
	private ArrayList<Boolean> vote;
	private ArrayList<Integer> histogram;
	private int firstFilter, secondFilter;
	private Threshold<Integer> threshold;

	public Concensus(ArrayList<AbstractDiversityMeasure> dms, int firstFilter, int secondFilter){
		super(dms);
		this.firstFilter = firstFilter;
		this.secondFilter = secondFilter;
	}

	public ArrayList<Boolean> select() throws Exception{
		int i, j;
		ArrayList<Boolean> selected;
		
		makeMatrix();
		diversityCombination();
		rank();
		makeHistogram();
		selected = applyThreshold();
		return selected;
	}

	private void makeMatrix() throws Exception{
		matrixV = new ArrayList<ArrayList<Boolean>>();   //inicialização da matriz
	    int i, contC = 0, j = 0, k;

	    baseClassifiersSize = classifiers.size()/instancesArray.size();
	    //System.out.println(baseClassifiersSize);
	    for(i=0;i<classifiers.size();i++)
			matrixV.add(new ArrayList<Boolean>());

	    for(i=0;i<classifiers.size();i++){
			for(k=0;k<instancesArray.get(j).size();k++){
				if((classifiers.get(i).classifyInstance(instancesArray.get(j).instance(k))) == instancesArray.get(j).instance(k).classValue())
					matrixV.get(i).add(true);
				else{
					matrixV.get(i).add(false);
				}
			}

			if((i+1) % baseClassifiersSize == 0)
	            j++;
	    }	    
	}

	private void diversityCombination(){
		int i, j, d;

		matrixDM = new ArrayList<ArrayList<ClassifierPairStruct>>();
		for(d=0;d<dms.size();d++){
			matrixDM.add(new ArrayList<ClassifierPairStruct>());
			for(i=0;i<matrixV.size();i++){
				for(j=i;j<matrixV.size();j++){
					matrixDM.get(d).add(new ClassifierPairStruct(dms.get(d), dms.get(d).measure(matrixV.get(i), matrixV.get(j)), i, j));
				}
			}
		}
	}

	private void rank(){
		int d, i;
		ranking = new ArrayList<ArrayList<Boolean>>();
		RankingSelection<ClassifierPairStruct> rs = new RankingSelection<ClassifierPairStruct>(firstFilter);

		for(d=0;d<dms.size();d++){
			rs.setArray(matrixDM.get(d));
			ranking.add(rs.select());
		}
	}

	private void makeHistogram(){
		RankingSelection<Integer> rv = new RankingSelection<Integer>(secondFilter);
		histogram = new ArrayList<Integer>(Collections.nCopies(classifiers.size(), 0));
		int i,j;

		for(i=0;i<dms.size();i++){
			for(j=0;j<ranking.get(i).size();j++){
				if(ranking.get(i).get(j) == true){
					histogram.set(matrixDM.get(i).get(j).c1, histogram.get(matrixDM.get(i).get(j).c1)+1);
					histogram.set(matrixDM.get(i).get(j).c2, histogram.get(matrixDM.get(i).get(j).c2)+1);
				}
			}
		}
		
		rv.setInverted(true);
		rv.setArray(histogram);
		vote = rv.select();
	}

	private ArrayList<Boolean> applyThreshold() throws Exception{
		ArrayList<Boolean> selected;
		ArrayList<Integer> secondHistogram = new ArrayList<Integer>();
		int i;
		for(i=0;i<vote.size();i++){
			if(vote.get(i))
				secondHistogram.add(histogram.get(i));
			else
				secondHistogram.add(-1);
		}

		int mean = 0;
		for(Integer j : histogram){
			mean += j;
		}
		mean = mean/histogram.size();
		threshold = new Threshold<Integer>(mean);
		threshold.setArray(histogram);
		selected = threshold.select();
		return selected;
	}
}