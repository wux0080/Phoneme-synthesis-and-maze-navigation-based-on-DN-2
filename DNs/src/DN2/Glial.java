package DN2;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;


public class Glial implements Serializable{
	//the glial cell's 3D location
	private float[] mLocation;
	//the number of neighbor neurons each glial cell pulls
	private int mTopK;
	//the index of glial cell
	private int mIndex;
	//the vector recording neighbor neurons' index which need to be pulled
	private int[] mPullIndex;
	//the vector which contains pulling vector for each neighbor neuron
	private float[][] mPullvectors;
	
	public Glial(int topk, int index){
		//set the number of neighbor neurons each glial cell pulls
		mTopK = topk;
		//set the index
		mIndex = index;
		//construct the location vector
		mLocation = new float[3];
		//construct the vector which recording neighbor neurons' index
		mPullIndex = new int[mTopK];
		//construct the vector which contains pulling vector for each neighbor neuron
		mPullvectors = new float[mTopK][3];
	}
	
	//calculate the distance between the two 3-D locations
	public float computeDistance(float[] vector){
		//make sure the two location vector have same length
		assert vector.length == mLocation.length;
		//calculate the distance
		float delta_h = (mLocation[0]-vector[0])*(mLocation[0]-vector[0]);
		float delta_v = (mLocation[1]-vector[1])*(mLocation[1]-vector[1]);
		float delta_d = (mLocation[2]-vector[2])*(mLocation[2]-vector[2]);
		float dis = (float)Math.sqrt((double)(delta_h+delta_v+delta_d));
		return dis;
		
	}
	
	//set the number of neighbor neurons each glial cell pulls
	public void settopk(int topk){
		this.mTopK=topk;
	}
	
	//get the number of neighbor neurons each glial cell pulls
	public int gettopk(){
		return mTopK;
	}
	
	//set the index
	public void setindex(int index){
		this.mIndex=index;
	}
	
	//get the index
	public int getindex(){
		return mIndex;
	}

	//set one neighbor neurons' index in corresponding location
	public void setpullindex(int index, int input){
		this.mPullIndex[index] = input;
	}
	
	//get the vector which recording neighbor neurons' index
	public int getpullindex(int index){
		return mPullIndex[index];
	}
	
	 //set one pulling vector for the neighbor neuron
	public void setpullvector(float[] vector, int index){
		if(index < mTopK){
			this.mPullvectors[index] = vector;	
		}
	}
	
	//get one pulling vector for the neighbor neuron
	public float[] getpullvector(int index){
		return mPullvectors[index];
	}

	//set glial cell's location
	public void setlocation(float[] location){
		this.mLocation = location;	
    }

	//set glial cell's location
    public float[] getlocation(){
	   return mLocation;
    }

    //set the vector which contains pulling vector for each neighbor neuron
	public void setlpullvectors(float[][] vectors){
		this.mPullvectors = vectors;	
    }

	//get the vector which contains pulling vector for each neighbor neuron
    public float[][] getpullvectors(){
	    return mPullvectors;
    }
    
    
}