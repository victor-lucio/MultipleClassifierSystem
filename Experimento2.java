
import weka.classifiers.*;
import weka.classifiers.lazy.*;
import weka.core.converters.*;
import weka.classifiers.functions.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.*;
import java.util.*;
import java.io.*;

public class Experimento2{
	public static void main(String args[]) throws Exception{
		int m;
		DataSource source;

		ArrayList<String> test = new ArrayList<String>();
		ArrayList<String> train = new ArrayList<String>();
		ArrayList<Instances> trainInst = new ArrayList<Instances>();
		ArrayList<Instances> testInst = new ArrayList<Instances>();

		File path1 = new File("data" + File.separator + "teste");
		File path2 = new File("data" + File.separator + "treino");
		File[] folder;


		folder = path1.listFiles();
		Arrays.sort(folder);
		for(File l : folder){
			test.add(l.getPath());
		}

		folder = path2.listFiles();
		Arrays.sort(folder);
		for(File l : folder){
			train.add(l.getPath());
		}

		for(m=0;m<train.size();m++){
			source = new DataSource(train.get(m));
			trainInst.add(source.getDataSet());
			trainInst.get(m).setClassIndex(trainInst.get(m).numAttributes()-1);
			trainInst.get(m).deleteAttributeAt(trainInst.get(m).numAttributes()-2);
		}

		for(m=0;m<test.size();m++){
			source = new DataSource(test.get(m));
			testInst.add(source.getDataSet());
			testInst.get(m).setClassIndex(testInst.get(m).numAttributes()-1);
			testInst.get(m).deleteAttributeAt(testInst.get(m).numAttributes()-2);
		}

		MultipleFeatureInstances treino = new MultipleFeatureInstances(trainInst);
		MultipleFeatureInstances teste = new MultipleFeatureInstances(testInst);

		ObjectInputStream ois = new ObjectInputStream(new FileInputStream("mcs.model"));
 		MCSClassifier cls = (MCSClassifier) ois.readObject();
 		ois.close();

 		Evaluation eval;
		eval = new Evaluation(treino);
		eval.evaluateModel(cls, teste);
		System.out.println(eval.toSummaryString());
	}
}