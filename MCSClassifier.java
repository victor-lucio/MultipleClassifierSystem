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
	
	private int percent; //porcentagem do dataset para treino
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
		int i,j,k,nof;
		MultipleFeatureInstances dataset;
		if(data instanceof MultipleFeatureInstances){
			//System.out.println("instancia do multiple!!");
			dataset = (MultipleFeatureInstances) data;
			dataset.selectFeature(0);
			nof = dataset.numberOfFeatures();
		}else{
			//System.out.println("nao eh multiple!!");
			ArrayList<Instances> aux = new ArrayList<Instances>();
			aux.add(data);
			dataset = new MultipleFeatureInstances(aux);
			dataset.selectFeature(0);
			aux = null;
			nof = 1;
		}

		for(i=0;i<nof-1;i++){ //constrói o array list com todas combinações de classificadores
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
		ArrayList<Instances> train, validate;

		if(fusionClassifier != null){

			RemovePercentage removeptrain = new RemovePercentage();
			RemovePercentage removepval = new RemovePercentage();

			train = new ArrayList<Instances>();
			validate = new ArrayList<Instances>();

			for(i=0;i<nof;i++){
				removeptrain.setInputFormat(dataset.getFeature(i));
				removeptrain.setPercentage(percent);
				removeptrain.setInvertSelection(true);
				train.add(Filter.useFilter(dataset.getFeature(i), removeptrain));
			}
			//System.out.println(train.get(0) + "\n" + "----------\n" + train.get(0).size() + "\n");

			for(i=0;i<nof;i++){
				removepval.setInputFormat(dataset.getFeature(i));
				removepval.setPercentage(percent);
				validate.add(Filter.useFilter(dataset.getFeature(i), removepval));
			}
			//System.out.println(validate.get(0) + "\n" + "----------\n" + validate.get(0).size() );
		}else{
			train = dataset.featuresToArray();
			validate = null;
		}
		
		//treinar classificadores
		k=0;
		for(i=0;i<nof;i++){
			for(j=0;j<numberClassifiers;j++,k++){
				classifiers.get(k).buildClassifier(train.get(i));
			}
		}

		//montar matriz
		
        if(fusionClassifier != null){
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
        	attributes = new ArrayList<Attribute>();
    		instanceValue = new double[classifiers.size() + 1];
    		Instances data2;
    		for(i=0;i<classifiers.size();i++)
	    		attributes.add(new Attribute("classifier "+i));
	    	attributes.add(validate.get(0).classAttribute());
	    	data2 = new Instances("validation dataset", attributes, matrixV.get(0).size());

	    	DenseInstance inst;
    		for(i=0;i<matrixV.get(0).size();i++){
    			inst = new DenseInstance(classifiers.size() + 1);
	        	for(j=0;j<classifiers.size();j++){ //Are numerical
	    			inst.setValue(j, matrixV.get(j).get(i));
	        	}
	        	inst.setValue(j, validate.get(0).instance(i).classValue()); //classvalue of each instance
	        	data2.add(inst);
	        	inst = null;
	        }
        	data2.setClassIndex(data2.numAttributes()-1);
        	fusionClassifier.buildClassifier(data2);
        	//System.out.println(data2);
        }
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception { //no singular, é o padrão da classe abstrata
		
		MultipleFeatureInstance multiInst;
		double classVal = 0; // representação da predição em forma de double
		ArrayList<AbstractInstance> instanceArray;
		double classArray[] = new double[classifiers.size() + 1];
		int i, j, k = 0;
		if(instance instanceof MultipleFeatureInstance){
			multiInst = (MultipleFeatureInstance) instance;
			instanceArray = multiInst.toArray();
		}else{
			instanceArray = new ArrayList<AbstractInstance>();
			instanceArray.add((AbstractInstance) instance);
		}
		for(i=0;i<instanceArray.size();i++){
			for(j=0;j<numberClassifiers;j++,k++){
				classArray[k] = (classifiers.get(k).classifyInstance(instanceArray.get(i)));
				//System.out.print(classArray[k] + " ");
			}
		}
		//System.out.println("\n---------");

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
			Instances fusionInstances;												//fusion classifier
			DenseInstance fusionInstance = new DenseInstance(1, classArray);
			fusionInstances = new Instances("fusion instance", attributes, 0);
			fusionInstances.add(fusionInstance);
			fusionInstances.setClassIndex(fusionInstance.numAttributes()-1);
			classVal = fusionClassifier.classifyInstance(fusionInstances.instance(0));
			fusionInstances = null;
			fusionInstance = null;
		}
		
		return classVal;
	}

	public double[] classifyInstances(Instances instances) throws Exception{ //no plural, classifica um set de instancias
		double[] r = new double[instances.size()];
		int i;
		for(i=0;i<instances.size();i++){
			r[i] = classifyInstance(instances.instance(i));
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