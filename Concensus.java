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
	private ArrayList<Boolean> vote, selected;
	private ArrayList<Integer> histogram;
	private ArrayList<Double> accuracy;
	private int firstFilter, c;
	private double mean;
	private Threshold<Integer> threshold;

	public Concensus(ArrayList<AbstractDiversityMeasure> dms, int firstFilter, int c){
		super(dms);
		this.firstFilter = firstFilter;
		this.c = c;
	}

	public ArrayList<Boolean> select() throws Exception{
		int i, j;
		
		makeMatrix();
		diversityCombination();
		rank();
		makeHistogram();
		return selected;
	}

	private void makeMatrix() throws Exception{
		matrixV = new ArrayList<ArrayList<Boolean>>();   //inicialização da matriz
		accuracy = new ArrayList<Double>(Collections.nCopies(classifiers.size(), 0.0));
	    int i, contC = 0, j = 0, k;
	    double sum = 0.0;

	    baseClassifiersSize = classifiers.size()/instancesArray.size();
	    //System.out.println(baseClassifiersSize);
	    for(i=0;i<classifiers.size();i++)
			matrixV.add(new ArrayList<Boolean>());

	    for(i=0;i<classifiers.size();i++){
			for(k=0;k<instancesArray.get(j).size();k++){
				if((classifiers.get(i).classifyInstance(instancesArray.get(j).instance(k))) == instancesArray.get(j).instance(k).classValue()){
					matrixV.get(i).add(true);
					accuracy.set(i, accuracy.get(i)+1);
					//System.out.println(accuracy.get(i));
				}else{
					matrixV.get(i).add(false);
				}
			}

			if((i+1) % baseClassifiersSize == 0)
	            j++;
	    }

	    //System.out.println(sum);
	    for(i=0;i<accuracy.size();i++){
	    	//System.out.println(accuracy.get(i) + "/" + instancesArray.get(0).size());
	    	accuracy.set(i, (accuracy.get(i)/instancesArray.get(0).size())*100);
	    }

	    for(double a : accuracy){
	    	sum += a;
	    }

	    mean = sum/accuracy.size();
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

		rs.setInverted(true);

		for(d=0;d<dms.size();d++){
			//System.out.println(matrixDM.get(d));
			rs.setArray(matrixDM.get(d));
			ranking.add(rs.select());
			//System.out.println(ranking.get(d));
		}
	}

	private void makeHistogram(){
		RankingSelection<Double> rv = new RankingSelection<Double>(c);
		ArrayList<Double> score = new ArrayList<Double>();
		histogram = new ArrayList<Integer>(Collections.nCopies(classifiers.size(), 0));
		int i,j,htm; //higher than mean

		for(i=0;i<dms.size();i++){
			for(j=0;j<ranking.get(i).size();j++){
				if(ranking.get(i).get(j) == true){
					histogram.set(matrixDM.get(i).get(j).c1, histogram.get(matrixDM.get(i).get(j).c1)+1);
					histogram.set(matrixDM.get(i).get(j).c2, histogram.get(matrixDM.get(i).get(j).c2)+1);
				}
			}
		}

		//System.out.println(accuracy);

		for(i=0;i<classifiers.size();i++){
			if(accuracy.get(i) >= mean){
				htm = 1;
				//System.out.println(">=");
			}else{
				htm = 0;
				//System.out.println("<");
			}
			score.add((double) ((100*histogram.get(i)*htm)+accuracy.get(i)));
		}
		//System.out.println(score);

		rv.setInverted(true);
		rv.setArray(score);

		selected = rv.select();
		//System.out.println(selected);
	}
}