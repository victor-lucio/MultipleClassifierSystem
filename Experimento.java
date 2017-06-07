/* 
	@author Victor Lúcio
	Federal University of São Paulo - ICT/UNIFESP
	"A Classifiers Fusion System Applied to Fenology"

	Multiple Classifier System
	from: Faria, Fabio "A Framework for Pattern Classifier Selection and Fusion", 2014

	Advisors: Jurandy Gomes de Almeida Junior <http://buscatextual.cnpq.br/buscatextual/visualizacv.do?id=K4736755E0>
			  Fabio Augusto Faria <http://buscatextual.cnpq.br/buscatextual/visualizacv.do?id=K4266712J6>
*/

import weka.classifiers.*;
import weka.classifiers.lazy.*;
import weka.core.converters.*;
import weka.classifiers.functions.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.*;
import java.util.*;
import java.io.*;

public class Experimento{
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
		// Concensus(medidas de diversidade, parametro t, parametro c*, classe de metricas)

		MCSClassifier MCS;
		
		SMO svm = new SMO(); // SVM

		MCS = new MCSClassifier(classifiers, concensus, 75, svm); // MCSClassifier(Classificadores, Seleção, divisão treino/validação, Classificador de Fusão)
		MCS.buildClassifier(treino);

		Evaluation eval;
		eval = new Evaluation(treino);
		eval.evaluateModel(MCS, teste);
		System.out.println(eval.toSummaryString());
	}
}