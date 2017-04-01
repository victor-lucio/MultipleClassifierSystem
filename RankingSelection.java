/* 
	@author Victor Lúcio
	Federal University of São Paulo - ICT/UNIFESP
	"A Classifiers Fusion System Applied to Fenology"

	Multiple Classifier System
	from: Faria, Fabio "A Framework for Pattern Classifier Selection and Fusion", 2014
*/

import java.util.*;

public class RankingSelection<T extends Comparable> extends AbstractSelectionMethod<T>{

	private int t;
	private ArrayList<Boolean> out;
	
	public RankingSelection(int t){
		this.t = t;
	}

	public void setArray(ArrayList<T> array){
		ArrayList<Integer> ranking = new ArrayList<Integer>();
		out = new ArrayList<Boolean>(Collections.nCopies(array.size(), false));
		int i, size;
		for(i=0;i<array.size();i++)
			ranking.add(i);
		Collections.sort(ranking, new IndexSort<T>(array));
		if(t > ranking.size())
			size = ranking.size();
		for(i=0;i<t;i++){
			out.set(ranking.get(i), true);
		}
	}


	public ArrayList<Boolean> select(){
		return out;
	}
}