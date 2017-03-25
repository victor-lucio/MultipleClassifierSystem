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
import weka.filters.unsupervised.instance.Randomize;
import java.util.*;
import java.awt.image.BufferedImage;
import java.lang.Iterable;
import java.nio.file.*;
import java.io.*;
import javax.imageio.ImageIO;
import java.util.Enumeration;

public class MCSClassifier extends MultipleClassifiersCombiner{
	
	private int percent; //porcentagem do dataset para treino
	private int numberClassifiers;
	private int selectedNumber;
	private AbstractClassifier fusionClassifier;
	private ArrayList<AbstractClassifier> classifiers;
	private ArrayList<ArrayList<Double>> matrixV;
	private AbstractClassifierSelection selectionMethod;
	private AbstractInstanceBasedSelection selectionMethodCasted;
    private ArrayList<Attribute> attributes;
    private Boolean[] selected;
    private ArrayList<AbstractClassifier> selectedClassifiers;

	public MCSClassifier(ArrayList<AbstractClassifier> classifiers, AbstractClassifierSelection selectionMethod,
						 int validationPercentage, AbstractClassifier fusionClassifier) throws Exception{
		this.classifiers = classifiers;
		numberClassifiers = classifiers.size();
		this.selectionMethod = selectionMethod;
		this.percent = validationPercentage;
		this.fusionClassifier = fusionClassifier;
		if(selectionMethod instanceof AbstractInstanceBasedSelection)
			selectionMethodCasted = (AbstractInstanceBasedSelection) selectionMethod;
	}

	private AbstractClassifier instanceBuilder(AbstractClassifier cls) throws InstantiationException, IllegalAccessException, Exception{
		return (AbstractClassifier) makeCopy(cls);
	}

	@Override
	public void buildClassifier(Instances data) throws Exception{
		int i,j,k,nof;
		MultipleFeatureInstances dataset;

		if(data instanceof MultipleFeatureInstances){
			dataset = (MultipleFeatureInstances) data;
			dataset.selectFeature(0);
			nof = dataset.numberOfFeatures();
		}else{
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

        //randomize instances
        for(i=0;i<dataset.numberOfFeatures();i++){
        	dataset.selectFeature(i);
        	dataset.randomize(new Random(1));
        }
        dataset.selectFeature(0);

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

		//selecionar classificadores
		if(selectionMethod instanceof AbstractInstanceBasedSelection){
			selectionMethodCasted.setClassifiers(classifiers);
			selectionMethodCasted.setInstances(validate);
			selected = selectionMethodCasted.select();
		}else if(selectionMethod == null){
			selected = new Boolean[classifiers.size()];
			Arrays.fill(selected, true);
		}else{
			selectionMethod.setClassifiers(classifiers);
			selected = selectionMethod.select();
		}

		selectedNumber = 0;
		for(i=0;i<classifiers.size();i++){
			if(selected[i]){
				selectedNumber++;		
			}
		}

		//montar matriz	
    	Instances validationInstances;	//validation
        if(fusionClassifier != null){

			ArrayList<ArrayList<Double>> matrixV = new ArrayList<ArrayList<Double>>();   //inicialização da matriz
	        for(i=0;i<selectedNumber;i++)
				matrixV.add(new ArrayList<Double>());
	 		
	 		int contC = 0;
	        j = 0;
	        for(i=0;i<classifiers.size();i++){
				if(i % numberClassifiers == 0)
	                j++;

				if(selected[i]){
					for(k=0;k<validate.get(j).size();k++){
						matrixV.get(contC).add(classifiers.get(i).classifyInstance(validate.get(j).instance(k)));
					}
					contC++;
				}
	        }

	        // criando instancias para treinar o segundo classificador
	        double[] instanceValue;
        	attributes = new ArrayList<Attribute>();
    		instanceValue = new double[selectedNumber + 1];
    		for(i=0;i<classifiers.size();i++)
    			if(selected[i])
	    			attributes.add(new Attribute("classifier "+i));
	    	attributes.add(validate.get(0).classAttribute());
	    	validationInstances = new Instances("validation dataset", attributes, matrixV.get(0).size());

	    	DenseInstance inst;
    		for(i=0;i<matrixV.get(0).size();i++){
    			inst = new DenseInstance(selectedNumber + 1);
	        	for(j=0;j<selectedNumber;j++){ //Are numerical
	    			inst.setValue(j, matrixV.get(j).get(i));
	        	}
	        	inst.setValue(j, validate.get(0).instance(i).classValue()); //classvalue of each instance
	        	validationInstances.add(inst);
	        	inst = null;
	        }
        	validationInstances.setClassIndex(validationInstances.numAttributes()-1);
        	fusionClassifier.buildClassifier(validationInstances);
        	//System.out.println(validationInstances);
        }
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception { //no singular, é o padrão da classe abstrata
		
		MultipleFeatureInstance multiInst;
		double classVal = 0; // representação da predição em forma de double
		ArrayList<AbstractInstance> instanceArray;
		double classArray[] = new double[selectedNumber + 1];
		int i, j, k = 0, contC;
		Instances fusionInstances;

		if(instance instanceof MultipleFeatureInstance){
			multiInst = (MultipleFeatureInstance) instance;
			instanceArray = multiInst.toArray();
		}else{
			instanceArray = new ArrayList<AbstractInstance>();
			instanceArray.add((AbstractInstance) instance);
		}

		j = 0;
		contC = 0;
		for(i=0;i<classifiers.size();i++){
			if(i % numberClassifiers == 0)
	            j++;

			if(selected[i]){
				classArray[contC] = classifiers.get(i).classifyInstance(instanceArray.get(j));
				contC++;
			}
	    }

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