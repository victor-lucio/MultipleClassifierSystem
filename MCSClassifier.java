/* 
	@author Victor Lúcio
	Federal University of São Paulo - ICT/UNIFESP
	"A Classifiers Fusion System Applied to Fenology"

	Multiple Classifier System
	from: Faria, Fabio "A Framework for Pattern Classifier Selection and Fusion", 2014
*/

import weka.classifiers.*;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.Filter;
import java.util.*;
import weka.classifiers.bayes.*;
import weka.classifiers.lazy.*;
import weka.classifiers.trees.*;
import java.awt.image.BufferedImage;
import java.lang.Iterable;
import java.nio.file.*;
import java.io.*;
import javax.imageio.ImageIO;
import java.util.Enumeration;

public class MCSClassifier extends MultipleClassifiersCombiner{
	
	private int percent; //porcentagem do dataSet para treino
	private int numberClassifiers;
	private AbstractClassifier fusionClassifier;
	private ArrayList<AbstractClassifier> classifiers;
	private ArrayList<ArrayList<Double>> matrixV;
	private ArrayList<AbstractDiversityMeasure> diversityMeasures;
    private ArrayList<Attribute> attributes;

	public MCSClassifier(ArrayList<AbstractClassifier> classifiers, ArrayList<AbstractDiversityMeasure> diversityMeasures,
						 int validationPercentage, AbstractClassifier fusionClassifier) throws Exception{
		this.classifiers = classifiers;
		numberClassifiers = classifiers.size();
		this.diversityMeasures = diversityMeasures;
		this.percent = validationPercentage;
		this.fusionClassifier = fusionClassifier;
	}

	private AbstractClassifier instanceBuilder(AbstractClassifier cls) throws InstantiationException, IllegalAccessException, Exception{
		return (AbstractClassifier) makeCopy(cls);
	}

	@Override
	public void buildClassifier(Instances data) throws Exception{
		int i,j,k;
		MultipleFeatureInstances dataset = (MultipleFeatureInstances) data;
		dataset.selectFeature(0);

		for(i=0;i<dataset.numberOfFeatures()-1;i++){ //constrói o array list com todas combinações de classificadores
            for(j=0;j<numberClassifiers;j++){
                try{
                    classifiers.add(instanceBuilder(classifiers.get(j)));
                }catch(InstantiationException instE){
                    System.err.println("Classifier InstantiationException in " + i);
                }catch(IllegalAccessException illegalE){
                    System.err.println("Classifier IllegalAccessException in " + i);
                }
            }
        }

		//dividir treino e validação
		ArrayList<Instances> train = new ArrayList<Instances>(), validate = new ArrayList<Instances>();
		RemovePercentage removep = new RemovePercentage();

		for(i=0;i<dataset.numberOfFeatures();i++){
			removep.setInputFormat(dataset.getFeature(i));
			removep.setPercentage(percent);
			train.add(Filter.useFilter(dataset.getFeature(i), removep));
		}

		for(i=0;i<dataset.numberOfFeatures();i++){
			removep.setInputFormat(dataset.getFeature(i));
			removep.setPercentage((100-percent));
			removep.setInvertSelection(true);
			validate.add(Filter.useFilter(dataset.getFeature(i), removep));
		}

		//treinar classificadores
		k=0;
		for(i=0;i<dataset.numberOfFeatures();i++){
			for(j=0;j<numberClassifiers;j++,k++){
				classifiers.get(k).buildClassifier(train.get(i));
			}
		}

		//montar matriz
		ArrayList<ArrayList<Double>> matrixV = new ArrayList<ArrayList<Double>>();   //inicialização da matriz
        for(i=0;i<classifiers.size();i++)
			matrixV.add(new ArrayList<Double>());
 		
 		int cont = 0;
        j = 0;
        for(i=0;i<classifiers.size();i++, cont++){
			if(cont == numberClassifiers){
                j++;
                cont = 0;
			}
			for(k=0;k<validate.get(j).size();k++){
				matrixV.get(i).add(classifiers.get(i).classifyInstance(validate.get(j).instance(k)));
			}
        }

        // criando instancias para treinar o segundo classificador
        double[] instanceValue;
        if(fusionClassifier != null){
        	attributes = new ArrayList<Attribute>();
    		instanceValue = new double[classifiers.size() + 1];
    		Instances data2;
    		for(i=0;i<classifiers.size();i++)
	    		attributes.add(new Attribute("classifier "+i));
	    	attributes.add(dataset.get(0).classAttribute());
	    	data2 = new Instances("validation dataset", attributes, 0);

    		for(i=0;i<dataset.size();i++){
	        	for(j=0;j<classifiers.size();j++){ //Are numerical
	    			instanceValue[j] = matrixV.get(j).get(i);
	        	}
	        	instanceValue[j] = dataset.instance(i).classValue(); //classvalue of each instance
	        	data2.add(new DenseInstance(1, instanceValue));
	        }
        	
        	fusionClassifier.buildClassifier(data2);
        }
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception { //no singular, é o padrão da classe abstrata
		
		MultipleFeatureInstance multInst = (MultipleFeatureInstance) instance;
		double classVal = 0; // representação da predição em forma de double
		ArrayList<AbstractInstance> instanceArray;
		double classArray[] = new double[classifiers.size()];
		int i, j, k = 0;

		instanceArray = multInst.toArray();
		for(i=0;i<instanceArray.size();i++){
			for(j=0;j<numberClassifiers;j++,k++){
				classArray[k] = (classifiers.get(k).classifyInstance(instanceArray.get(i)));
				//System.out.println(classArray[k]);
			}
		}
		//System.out.println("---------");

		if(fusionClassifier == null){ 						//majority using hashmap counting
			
			HashMap<Double, Integer> map = new HashMap<Double, Integer>();
			for(i=0;i<classArray.length;i++){
				if(map.containsKey(classArray[i])){
					map.put(classArray[i], map.get(classArray[i])+1);
				}else{
					map.put(classArray[i], 1);
				}
			}
			int aux = 0;
			Double max = 0.0;
			Integer value;
			for(Double key : map.keySet()){
	            value = map.get(key);
	            if(value > aux){
	            	aux = value;
	            	max = key;
	            }
        	}
        	classVal = max;

		}else{ 		
															//fusion classifier
			DenseInstance fusionInstance = new DenseInstance(1, classArray);
			classVal = fusionClassifier.classifyInstance(fusionInstance);
		
		}
		
		return classVal;
	}

	public double[] classifyInstances(Instances instances) throws Exception{ //no plural, classifica um set de instancias
		MultipleFeatureInstances multiInst = (MultipleFeatureInstances) instances;
		double[] r = new double[multiInst.size()];
		int i;
		for(i=0;i<multiInst.size();i++){
			r[i] = classifyInstance(multiInst.instance(i));
		}
		return r;
	}

	@Override
 	public Capabilities getCapabilities() {
 		Capabilities result = super.getCapabilities(); //pegar tudo da classe pai e desativar por segurança
		result.disableAll();

		//atributos que podem ser utilizados
    	result.enable(Capability.NOMINAL_ATTRIBUTES);
    	result.enable(Capability.NUMERIC_ATTRIBUTES);

		//tipo de classes
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NUMERIC_CLASS);

    	return result;
  	}

  	public Classifier getClassifier(int index){
  		return (Classifier) classifiers.get(index);
  	}

  	public Classifier[] getClassifiers(){
  		return (Classifier[]) classifiers.toArray();
  	}
}