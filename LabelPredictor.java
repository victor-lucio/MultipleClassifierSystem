import java.io.*;
import java.lang.*;
import java.util.*;
import weka.classifiers.*;
import weka.classifiers.lazy.*;
import weka.classifiers.functions.*;
import weka.core.*;
import java.util.*;

public class LabelPredictor{
	
	private MultipleFeatureInstances train, test;
	private ArrayList<Integer> testIndex;
	private ArrayList<ArrayList<Double>> auxtrain, auxtest;
	private MCSClassifier MCS;

	public LabelPredictor() throws Exception{
		train = test = null;

		ArrayList<AbstractClassifier> classifiers = new ArrayList<AbstractClassifier>();
		classifiers.add(new IBk()); // kNN

 		ArrayList<AbstractDiversityMeasure> dm = new ArrayList<AbstractDiversityMeasure>();
		dm.add(new QStatistic());
		dm.add(new DoubleFaultMeasure());
		dm.add(new DisagreementMeasure());
		dm.add(new CorrelationCoefficient());
		dm.add(new InterraterAgreement());

		AverageAccuracyMean metrics = new AverageAccuracyMean();
		Concensus concensus = new Concensus(dm, 100, 6, metrics);

		SMO svm = new SMO(); // SVM
		MCSClassifier MCS = new MCSClassifier(classifiers, concensus, 75, svm);

		testIndex=null;
	}

	public void addFeature(ArrayList<ArrayList<Double>> array){
		int i, j, flag=0;
		
		auxtrain=auxtest=null;
		
		if(testIndex==null){
			testIndex = new ArrayList<Integer>();
			flag=1;
		}

		for(i=0;i<array.size();i++){
			if(array.get(i).get(array.get(i).size()-1) == -1){
				auxtest.add(new ArrayList<Double>(array.get(i)));
				if(flag==1)				
					testIndex.add(i);
			}else{
				auxtrain.add(new ArrayList<Double>(array.get(i)));
			}
		}

       	ArrayList<Attribute> attributes = new ArrayList<Attribute>();
    	double[] instanceValue = new double[auxtrain.get(0).size()];
    	for(i=0;i<auxtrain.get(0).size()-1;i++)
	    	attributes.add(new Attribute("attribute" + i));
	    attributes.add(new Attribute("class"));
	    
	    Instances trainInstances = new Instances("train dataset", attributes, auxtrain.size());
	    Instances testInstances = new Instances("test dataset", attributes, auxtest.size());
	    
	    DenseInstance inst;
   		for(i=0;i<auxtrain.size();i++){
   			inst = new DenseInstance(attributes.size());
    		trainInstances.add(inst);
    		for(j=0;j<auxtrain.get(0).size();j++){
    			trainInstances.get(i).setValue(j, auxtrain.get(i).get(j));
    		}
    	}

    	for(i=0;i<auxtest.size();i++){
   			inst = new DenseInstance(attributes.size());
    		testInstances.add(inst);
    		for(j=0;j<auxtest.get(0).size();j++){
    			testInstances.get(i).setValue(j, auxtest.get(i).get(j));
    		}
    	}

    	if(train == null){
    		train = new MultipleFeatureInstances(new ArrayList<Instances>(Arrays.asList(trainInstances)));
    		test = new MultipleFeatureInstances(new ArrayList<Instances>(Arrays.asList(testInstances)));
    	}else{
    		train.addFeature(trainInstances);
    		test.addFeature(testInstances);
    	}
	}

	public ArrayList<Double> getLabels() throws Exception{
		int i, j=0;
		
		MCS.buildClassifier(train);

		ArrayList<Double> r = new ArrayList<Double>();
		for (i=0;i<train.size()+test.size();i++){
			if(i==testIndex.get(j)){
				r.add(MCS.classifyInstance(test.instance(j)));
				j++;
			}else{
				r.add(train.instance(i).classValue());
			}
		}

		return r;
	}
}
