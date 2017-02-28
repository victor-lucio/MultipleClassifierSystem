/* 
	@author Victor Lúcio
	Federal University of São Paulo - ICT/UNIFESP
	"A Classifiers Fusion System Applied to Fenology"

	Multiple Classifier System
	from: Faria, Fabio "A Framework for Pattern Classifier Selection and Fusion", 2014
*/

import weka.classifiers.*;
import weka.classifiers.lazy.*;
import weka.classifiers.trees.*;
import weka.classifiers.bayes.*;
import weka.core.*;
import java.util.*;
import java.io.*;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.*;

public class Test{

	public static void main(String args[]) throws Exception{
		int i;
		String[] paths = new String[4];
		String[] testPaths = new String[4];
		DataSource source;
		double r;
		ArrayList<Instances> data = new ArrayList<Instances>();
		ArrayList<Instances> testData = new ArrayList<Instances>();
		paths[0] = "data/acc.arff";
		paths[1] = "data/bic.arff";
		paths[2] = "data/ccv.arff";
		paths[3] = "data/gch.arff";
		testPaths[0] = "data/testacc.arff";
		testPaths[1] = "data/testbic.arff";
		testPaths[2] = "data/testccv.arff";
		testPaths[3] = "data/testgch.arff";

		for(i=0;i<paths.length;i++){
			source = new DataSource(paths[i]);
			data.add(source.getDataSet());
			data.get(i).setClassIndex(data.get(i).numAttributes()-1);
			source = null;
		}

		MultipleFeatureInstances instances = new MultipleFeatureInstances(data);

		ArrayList<AbstractClassifier> classifiers = new ArrayList<AbstractClassifier>();
		classifiers.add(new J48());
		classifiers.add(new NaiveBayes());
		classifiers.add(new IBk());

		MCSClassifier MCS = new MCSClassifier(classifiers, (ArrayList<AbstractDiversityMeasure>) null, 60, (AbstractClassifier)null);

		MCS.buildClassifier(instances);

		for(i=0;i<testPaths.length;i++){
			source = new DataSource(testPaths[i]);
			testData.add(source.getDataSet());
			testData.get(i).setClassIndex(testData.get(i).numAttributes()-1);
			source = null;
		}

		MultipleFeatureInstances testInstances = new MultipleFeatureInstances(testData);

		r = MCS.classifyInstance(testInstances.instance(5));

		//for(i=0;i<r.length;i++)
			System.out.println(r + "\n");
	}
}