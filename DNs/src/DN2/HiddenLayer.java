package DN2;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import javax.swing.Spring;

import DN2.HiddenLayer.Pair;
import MazeInterface.Agent;
import MazeInterface.Commons;
import MazeInterface.DNCaller;

import java.io.*;

public class HiddenLayer implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
    public int addNum;
	//the number of winner neurons
	private int topK;
    //the number of neighbor neurons being pulled
    private int glialtopK;
    //the number of Y (hidden) neurons
	private int numNeurons;
    //the percentage of lateral preResponse takes
	private float lateralPercent;
    //the current bottom-up input vector
	private float[] currentBottomUpInput;
    //the current top-down input vector
	private float[] currentTopDownInput;
    //the preResponse vector
    private float[] preResponse;
    //the input size vector
	private int[] inputSize;
    //the number of bottom-up connections
	private int numBottomUpWeights;
    //the number of top-down connections
	private int numTopDownWeights;
    //the receiptive field size
	private int rfSize;
    //the stride of receiptive field
	private int rfStride;
    //the matrix recording receiptive field location
	private int[][] rf_id_loc;
//	private int usedHiddenNeurons;
    //the vector recording the number of used neuron for each type
	private int[] typeindex;
    //number of neuron types
	private int numTypes;
    //the matrix recording each type neurons' index
	private float[][] areaMask;
    // The is the number of Y neurons
	public int usedNeurons;
	//private int[] winnerIndexs;
    //the vector recording growth rate
    private float[][] mGrowthRate;
    //the vector recording the coefficient of mean value for dynamic top-k
    private float[][] mMeanValue;
    //machine zero value
	private final float MACHINE_FLOAT_ZERO = 0.0001f;
    //the perfect match value
	private final float ALMOST_PERFECT_MATCH_RESPONSE = 1.0f - 10 * MACHINE_FLOAT_ZERO;
    //whether use dynamic top-k competition
	private boolean dynamicInhibition;
    //the percentage of prescreening keep
	private float prescreenPercent;
	private boolean type3;
	private boolean type4learn;
	private boolean type5learn;
	private boolean type5g;
	private boolean type3learn;
	private boolean type3g;
	private boolean type7learn;
	private boolean type7g;
    //the y neuron array
	public Neuron[] hiddenNeurons;
    //the inhibitory neuron array
	public InhibitoryNeuron[] inhibitoryNeurons;
    //the size of glial cells
	private int glialsize;
    //the number of glial cells
	public int numGlial;
    //the glial cell array
	public Glial[] glialcells;
    public boolean isPriLateral;
    public int PriLateralsize;
    public float[] priLateralvector;
    
	private boolean yAttend;
	
	public int winner;
	//the construction of hidden layer
	public HiddenLayer(int initialNumNeurons, int topK, int sensorSize, int priHidden, int motorSize, int rfSize, int rfStride,
			int[][] rf_id_loc, int[][] inputSize, float prescreenPercent, int[] typeNum, float[][] growthTable, float[][] meanTable, 
			boolean dynamicinhibition) {
		this.setTopK(topK);
		//winnerIndexs = new int [topK];
//		this.usedHiddenNeurons = topK + 1; // bound of used neurons.
		//get used neuron types
		typeindex = new int[typeNum.length];
		for(int i=0; i<typeNum.length; i++){
			typeindex[i] = typeNum[i];
		}
		for(int i = 0; i < typeindex.length; i++){
		    numTypes += typeindex[i]/2;
		}
		
		isPriLateral = true;
		PriLateralsize = priHidden;
		priLateralvector = new float[PriLateralsize];
		this.addNum = 1;
		//initialize the number of used neurons
        this.usedNeurons = 2*numTypes;
        //initialize the number of total neurons
		this.numNeurons = initialNumNeurons + 2*numTypes;
		//whether use dynamic inhibition or not
		this.dynamicInhibition = dynamicinhibition;
		//size of local receptive field
		this.rfSize = rfSize;
		//moving step of local receptive field
		this.rfStride = rfStride;
		//array of generated local receptive fields
		this.rf_id_loc = rf_id_loc;
        //initialize the input size
		this.inputSize = inputSize[0];
        //get the grow rate table values
		this.mGrowthRate = new float[growthTable.length][];
		for(int i = 0; i < growthTable.length; i++){
			mGrowthRate[i] = new float[growthTable[i].length];
			System.arraycopy(growthTable[i], 0, mGrowthRate[i], 0, growthTable[i].length);
		}
		//get the mean value table values
		this.mMeanValue = new float[meanTable.length][];
		for(int i = 0; i < meanTable.length; i++){
			mMeanValue[i] = new float[meanTable[i].length];
			System.arraycopy(meanTable[i], 0, mMeanValue[i], 0, meanTable[i].length);
		}

        //initialize the neurons		
		int number = 0;
		hiddenNeurons = new Neuron[numNeurons];
		//initialize each initial neuron 
		for(int i=0; i<typeindex.length; i++){
			if(typeindex[i]!=0){
		        for(int j=number*2;j<(number+1)*2;j++){
		        	if(isPriLateral){
		        		hiddenNeurons[j] = new Neuron(sensorSize,motorSize,numNeurons+PriLateralsize, true,(i+1), j,this.inputSize);
			        }
		        	else{
		        		hiddenNeurons[j] = new Neuron(sensorSize,motorSize,numNeurons, true,(i+1), j,this.inputSize);
		        	}
		        }
		        number++;
			}
		}
		//initialize each not-used neuron 
		for(int i=2*numTypes;i<numNeurons;i++){
			if(isPriLateral){
				hiddenNeurons[i] = new Neuron(sensorSize,motorSize,numNeurons+PriLateralsize, true,0, i,this.inputSize);
			}
			else{
				hiddenNeurons[i] = new Neuron(sensorSize,motorSize,numNeurons, true,0, i,this.inputSize);
			}
		}
        //initialize the initial inhibitory neurons		
		inhibitoryNeurons = new InhibitoryNeuron[numNeurons];
		number = 0;
		for(int i=0; i<typeindex.length; i++){
			if(typeindex[i]!=0){
		        for(int j=number*2;j<(number+1)*2;j++){
			        inhibitoryNeurons[j] = new InhibitoryNeuron(sensorSize,motorSize,numNeurons, true,(i+1), j,this.inputSize);
		        }
		        number++;
			}
		}
		//initialize the not-used inhibitory neurons
		for(int i=2*numTypes;i<numNeurons;i++){
			inhibitoryNeurons[i] = new InhibitoryNeuron(sensorSize,motorSize,numNeurons, true,0, i,this.inputSize);
		}
        //set initial neurons' locations		
		for(int i=0;i<numTypes;i++){
			float[] temp = {4.5f+(float)Math.pow(-1, i)*0.1f*(i/2), 4.5f, 4.5f};
			hiddenNeurons[2*i].setlocation(temp);
			temp[1]+=0.1f;
			hiddenNeurons[2*i+1].setlocation(temp);

		}
        //set the initial inhibitory neurons' locations		
		for(int i=0;i<numTypes;i++){
			float[] temp = {4.5f+(float)Math.pow(-1, i)*0.1f*(i/2), 4.5f, 4.6f};
			inhibitoryNeurons[2*i].setlocation(temp);
			temp[1]+=0.1f;
			inhibitoryNeurons[2*i+1].setlocation(temp);

		}
		// Each type neuron initial two neurons have global receptive fields.
		for (int i = 0; i < numTypes; i++) {
			for (int j = 0; j < sensorSize; j++) {
				hiddenNeurons[2*i].setBottomUpMask(1, j);
				hiddenNeurons[2*i+1].setBottomUpMask(1, j);
				inhibitoryNeurons[2*i].setBottomUpMask(1, j);
				inhibitoryNeurons[2*i+1].setBottomUpMask(1, j);
			}
		}
		
		// topDownMask are ones at the beginning
		for (int i = 0; i < numNeurons; i++) {
			for (int j = 0; j < motorSize; j++) {
				hiddenNeurons[i].setTopDownMask(1, j);
				inhibitoryNeurons[i].setTopDownMask(1, j);
			}
		}	
		
		//set neurons' lateral mask
/*		for (int i = 0; i < numNeurons; i++) {
			for (int j = 0; j < numNeurons; j++) {
				hiddenNeurons[i].setLateralMask(1, j);
			}
		}	*/
		
		//inhibitory neurons' lateral mask only contain the same type neurons.
		for (int i = 0; i <numTypes; i++) {
			inhibitoryNeurons[2*i].setLateralWeight(1, 2*i);
			inhibitoryNeurons[2*i].setLateralWeight(1, 2*i+1);
			inhibitoryNeurons[2*i+1].setLateralWeight(1, 2*i);
			inhibitoryNeurons[2*i+1].setLateralWeight(1, 2*i+1);
			inhibitoryNeurons[2*i].setLateralMask(1, 2*i);
			inhibitoryNeurons[2*i].setLateralMask(1, 2*i+1);
			inhibitoryNeurons[2*i+1].setLateralMask(1, 2*i);
			inhibitoryNeurons[2*i+1].setLateralMask(1, 2*i+1);
		}
		//set the percentage of lateral excitation responses
		this.lateralPercent = 1.0f;
		
		this.prescreenPercent = prescreenPercent;
		//set the pre-response values
	    this.preResponse = new float[numNeurons];
	    for(int i=0; i<numNeurons; i++){
	    	preResponse[i] = 0;
	    }
		//get the bottom-up weights size
		numBottomUpWeights = sensorSize;
		currentBottomUpInput = new float[numBottomUpWeights];
		//get the top-down weights size
		numTopDownWeights = motorSize;
		currentTopDownInput = new float[motorSize];
		//construct the area mask array
		areaMask = new float[7][numNeurons];
        //initialize the area mask
		int tempindex = 0;
		for(int i=0; i<7; i++){
			for(int j=0; j<numNeurons; j++){
				 areaMask[i][j] = 0;
			}
			if(typeindex[i] !=0){
				areaMask[i][2*tempindex]=1.0f;
				areaMask[i][2*tempindex+1]=1.0f;
				tempindex += 1;
			}
		}
		yAttend = false;
		type3 = true;
		type4learn = true;
		type5learn = true;
		type3g =true;
		type3learn = true;
		type5g =true;
		type7learn = true;
		type7g =true;
		
		//initialize the glial cells
        //set the number of neighbor neurons to be pulled
		glialtopK = 3;
        //set the size of glial cells
		glialsize = 10;
		numGlial = glialsize*glialsize*glialsize;
		glialcells = new Glial[numGlial];
		for(int i=0; i<glialsize;i++){
			for(int j=0; j<glialsize;j++){
				for(int k=0; k<glialsize;k++){
					float[] temp = {(float)k,(float)j,(float)i};
					glialcells[glialsize*glialsize*i+glialsize*j+k] = new Glial(glialtopK,glialsize*glialsize*i+glialsize*j+k);
					glialcells[glialsize*glialsize*i+glialsize*j+k].setlocation(temp);
				}
			}
		}
		
	}

    //set the growth rate
	public void setGrowthRate(float[][] growth_table){
        //construct the growth rate array
		this.mGrowthRate = new float[growth_table.length][];
		for(int i = 0; i < growth_table.length; i++){
			mGrowthRate[i] = new float[growth_table[i].length];
			System.arraycopy(growth_table[i], 0, mGrowthRate[i], 0, growth_table[i].length);
		}
	}
    
	public void saveAgeToFile(String hidden_ind, String index) {
		try {
			PrintWriter wr_age = new PrintWriter(new File(hidden_ind + "firing age "+index+".txt"));
			for (int i = 0; i < numNeurons; i++) {				
					wr_age.print(Integer.toString(hiddenNeurons[i].getfiringage()));
					wr_age.println();
				}			
			wr_age.close();

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	//save weights to a txt file
	public void saveWeightToFile(String hidden_ind) {
		try {
			PrintWriter wr_weight = new PrintWriter(new File(hidden_ind + "bottom_up_weight.txt"));
			//PrintWriter wr_mask = new PrintWriter(new File(hidden_ind + "bottom_up_mask.txt"));
			PrintWriter wr_topdown = new PrintWriter(new File(hidden_ind + "top_down_weight.txt"));
			//PrintWriter wr_topdownMask = new PrintWriter(new File(hidden_ind + "top_down_mask.txt"));
			PrintWriter wr_lateral = new PrintWriter(new File(hidden_ind + "lateral_weight.txt"));
			//PrintWriter wr_lateralMask = new PrintWriter(new File(hidden_ind + "lateral_mask.txt"));
			for (int i = 0; i < numNeurons; i++) {
				if(hiddenNeurons[i].getType() == 5){
					for (int j = 0; j < numBottomUpWeights; j++) {
						wr_weight.print(String.format("% .2f", hiddenNeurons[i].getBottomUpWeights()[j]) + ',');
//						wr_mask.print(Float.toString(hiddenNeurons[i].getBottomUpMask()[j]) + ',');
					}
					wr_weight.println();
//					wr_mask.println();
				}

				wr_topdown.print(Integer.toString(hiddenNeurons[i].getType())+ ',');
				for (int j = 0; j < hiddenNeurons[i].getTopDownWeights().length; j++) {
					wr_topdown.print(String.format("% .2f", hiddenNeurons[i].getTopDownWeights()[j]) + ',');
				}
				wr_topdown.println();

//				for (int j = 0; j < hiddenNeurons[i].getTopDownMask().length; j++) {
//					wr_topdownMask.print(String.format("% .2f", hiddenNeurons[i].getTopDownMask()[j]) + ',');
//				}
//				wr_topdownMask.println();
				if(hiddenNeurons[i].getType() == 3){
					for (int j = 0; j <  hiddenNeurons[i].getLateralWeights().length; j++) {
						wr_lateral.print(String.format("% .3f", hiddenNeurons[i].getLateralWeights()[j]) + ',');
					}
					wr_lateral.println();
				}

//				for (int j = 0; j <  hiddenNeurons[i].getLateralMask().length; j++) {
//					wr_lateralMask.print(String.format("% .2f", hiddenNeurons[i].getLateralMask()[j]) + ',');
//				}
//				wr_lateralMask.println();
			}
			wr_weight.close();
//			wr_mask.close();
			wr_topdown.close();
//			wr_topdownMask.close();
			wr_lateral.close();
//			wr_lateralMask.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}


	// Initialize receptive field for the ith neuron, according to the whereID.
	// The center of the receptive field is located at rf_id_loc[i].
	// size of the receptive field is rf_size.
	public void initializeRfMask(int i, int whereID, DN2.MODE mode) {
        //initialize the receipt field for toy problem
		if (mode == DN2.MODE.GROUP) {
			if (whereID >= 0) {
				int half_rf_size = (rfSize - 1) / 2;
				int rf_begin_row = rf_id_loc[whereID][1] - half_rf_size;
				int rf_end_row = rf_id_loc[whereID][1] + half_rf_size;
				int rf_begin_col = rf_id_loc[whereID][0] - half_rf_size;
				int rf_end_col = rf_id_loc[whereID][0] + half_rf_size;
				// assert inputSize[0] * inputSize[1] ==
				// hiddenNeurons[i].getBottomUpMask().length;
				for (int row = rf_begin_row; row <= rf_end_row; row++) {
                    //initialize the bottom-up mask vector
					for (int col = rf_begin_col; col <= rf_end_col; col++) {
						int current_idx = col * inputSize[1] + row;
						hiddenNeurons[i].setBottomUpMask(1.0f, current_idx);
					}
				}
				for (int pixel_ind = inputSize[0] * inputSize[1]; pixel_ind < hiddenNeurons[i]
						.getBottomUpMask().length; pixel_ind++) {
					hiddenNeurons[i].setBottomUpMask(1.0f, pixel_ind);
				}
			} else {
				for (int pixel_ind = 0; pixel_ind < hiddenNeurons[i].getBottomUpMask().length; pixel_ind++) {
					hiddenNeurons[i].setBottomUpMask(1.0f, pixel_ind);
				}
			}
			System.out.println("Rf initialized: " + whereID);
        //initialize the receipt field for maze problem
		}
		else if(mode == DN2.MODE.Speech){
			for (int pixel_ind = 0; pixel_ind < hiddenNeurons[i].getBottomUpMask().length; pixel_ind++) {
				hiddenNeurons[i].setBottomUpMask(1.0f, pixel_ind);
			}
		}
		else if (mode == DN2.MODE.MAZE){
			if (Commons.vision_2D_flag == false){
				int rf_loc = DNCaller.curr_loc;
				int rf_size = DNCaller.curr_scale;
				float[] bottom_up_mask = new float[currentBottomUpInput.length];
	            //initialize the bottom-up mask vector
				if (rf_loc < 0){
					for (int j = 0; j < 3 * Agent.vision_num; j++){
						bottom_up_mask[j] = 1;
					}
				} else {
					for (int j = rf_loc * 3; j < (rf_loc + rf_size) * 3; j++){
						bottom_up_mask[j] = 1;
					}
				}
				for (int j = 3 * Agent.vision_num; j < currentBottomUpInput.length; j++){
					bottom_up_mask[j] = 1;
				}
				
				hiddenNeurons[i].setBottomUpMasks(bottom_up_mask);
			} else {
				int rf_loc = DNCaller.curr_loc;
				int rf_size = DNCaller.curr_scale;
				int rf_type = DNCaller.curr_type;
				float[] bottom_up_mask = new float[currentBottomUpInput.length];
				if (rf_loc < 0){
					for (int j = 0; j < currentBottomUpInput.length; j++){
						bottom_up_mask[j] = 1;
					}
				} else {
					int vision_height = Agent.vision_num * 3/4;
					int count = 0;
					for (int j = 0; j < vision_height; j++){
						for (int k = 0; k < Agent.vision_num; k++){
							if (k >= rf_loc - 1 && k <= (rf_loc + rf_size)){
								bottom_up_mask[count * 3] = 1;
								bottom_up_mask[count * 3 + 1] = 1;
								bottom_up_mask[count * 3 + 2] = 1;
							}
							count ++ ;
						}
					}
				}
				hiddenNeurons[i].setBottomUpMasks(bottom_up_mask);
			}
		}
	}

	public void addFiringAges() {
		for(int j=0; j<usedNeurons; j++){
			hiddenNeurons[j].addFiringAge();
			inhibitoryNeurons[j].addFiringAge();
		}
	}
	
    //hebbian learning for y neurons
	public void hebbianLearnHidden(float[] sensorInput, float[] motorInput) {
		boolean learning = true;
		float[] tempResponse1 = new float[numNeurons+PriLateralsize];
		float[] tempResponse2 = new float[numNeurons];
        //get the response from last frame
		for(int i=0;i<numNeurons;i++){
			tempResponse1[i] = hiddenNeurons[i].getoldresponse();
			tempResponse2[i] = preResponse[i];
//			winnerIndex[i] = hiddenNeurons[i].getwinnerflag();
//			System.out.println("neuron "+i+" winnerflag "+hiddenNeurons[i].getwinnerflag());
		}
		for(int i=0; i<PriLateralsize; i++){
			tempResponse1[i+numNeurons] = priLateralvector[i];
		}
        int tempIndex = (int)(((float)usedNeurons/numNeurons)/0.05f);
        if(usedNeurons == numNeurons){
        	tempIndex = tempIndex-1;
        }
        for(int j=0; j<usedNeurons; j++){
        	if(hiddenNeurons[j].getType() == 5){
        		learning  = type5learn;
            }
        	else if(hiddenNeurons[j].getType() == 7){
        		learning  = type7learn;
        	}
        	else if(hiddenNeurons[j].getType() == 3){
        		learning  = type3learn;
        	}
        	else if(hiddenNeurons[j].getType() == 4){
        		learning  = type4learn;
        	}
        	else{
        		learning = true;
        	}
        	
        	if(learning){
            //update weights for each y neuron
        	hiddenNeurons[j].hebbianLearnHidden(sensorInput, motorInput, tempResponse1);
        	
        	if(hiddenNeurons[j].getIsGrowLateral()) {
        		int[] temp = hiddenNeurons[j].getLateralGrowList();
        		growLateralConnection(j, temp);        		
        		hiddenNeurons[j].setIsGrowLateral(false);
        		hiddenNeurons[j].clearLateralGrowList();
        		hiddenNeurons[j].clearLateralDeleteList();
        	}
            //update weights for each inhibitory neuron
        	inhibitoryNeurons[j].hebbianLearnHidden(sensorInput, motorInput, tempResponse2,areaMask);
       			    //update the inhibition area for each neuron
      	        float[] tempWeights = inhibitoryNeurons[j].getLateralWeights();
        			float meanWeight = mean(tempWeights,areaMask[inhibitoryNeurons[j].getType()-1 ])*mMeanValue[tempIndex][1];  			
        			     for(int k = 0; k < usedNeurons; k++){
        			    	 if((tempWeights[k]>=meanWeight) && (areaMask[inhibitoryNeurons[j].getType()-1 ][k]!=0)){
        			    		 inhibitoryNeurons[j].setLateralMask(1, k);			    		 
        			    	 }
        			    	 else{
        			    		 inhibitoryNeurons[j].setLateralMask(0, k);
        			    	 }
        			     }	
          }
        }

	}

	   //hebbian learning for y neurons
		public void hebbianLearnHiddenParallel(float[][] sInputs, float[] sensorInput, float[] motorInput) {
			boolean learning = true;
			float[] tempResponse1 = new float[numNeurons+PriLateralsize];
			float[] tempResponse2 = new float[numNeurons];
	        //get the response from last frame
			for(int i=0;i<numNeurons;i++){
				tempResponse1[i] = hiddenNeurons[i].getoldresponse();
				tempResponse2[i] = preResponse[i];
//				winnerIndex[i] = hiddenNeurons[i].getwinnerflag();
//				System.out.println("neuron "+i+" winnerflag "+hiddenNeurons[i].getwinnerflag());
			}
			for(int i=0; i<PriLateralsize; i++){
				tempResponse1[i+numNeurons] = priLateralvector[i];
			}
	        int tempIndex = (int)(((float)usedNeurons/numNeurons)/0.05f);
	        if(usedNeurons == numNeurons){
	        	tempIndex = tempIndex-1;
	        }
	        for(int j=0; j<usedNeurons; j++){
	        	if(hiddenNeurons[j].getType() == 5){
	        		learning  = type5learn;
	            }
	        	else if(hiddenNeurons[j].getType() == 7){
	        		learning  = type7learn;
	        	}
	        	else if(hiddenNeurons[j].getType() == 3){
	        		learning  = type3learn;
	        	}
	        	else if(hiddenNeurons[j].getType() == 4){
	        		learning  = type4learn;
	        	}
	        	else{
	        		learning = true;
	        	}
	        	
	        	if(learning){
	            //update weights for each y neuron
	        	hiddenNeurons[j].hebbianLearnHiddenParallel(sInputs, motorInput, tempResponse1);
	        	
	        	if(hiddenNeurons[j].getIsGrowLateral()) {
	        		int[] temp = hiddenNeurons[j].getLateralGrowList();
	        		growLateralConnection(j, temp);
	        		hiddenNeurons[j].setIsGrowLateral(false);
	        		hiddenNeurons[j].clearLateralGrowList();
	        		hiddenNeurons[j].clearLateralDeleteList();
	        	}
	        	
	            //update weights for each inhibitory neuron
	        	inhibitoryNeurons[j].hebbianLearnHidden(sensorInput, motorInput, tempResponse2,areaMask);
	       			    //update the inhibition area for each neuron
	      	        float[] tempWeights = inhibitoryNeurons[j].getLateralWeights();
	        			float meanWeight = mean(tempWeights,areaMask[inhibitoryNeurons[j].getType()-1 ])*mMeanValue[tempIndex][1];  			
	        			     for(int k = 0; k < usedNeurons; k++){
	        			    	 if((tempWeights[k]>=meanWeight) && (areaMask[inhibitoryNeurons[j].getType()-1 ][k]!=0)){
	        			    		 inhibitoryNeurons[j].setLateralMask(1, k);			    		 
	        			    	 }
	        			    	 else{
	        			    		 inhibitoryNeurons[j].setLateralMask(0, k);
	        			    	 }
	        			     }	
	          }
	        }

		}
		
	// convert into 1d Array
	public float[] getResponse1D() {
        //construct the new array
		float[] inputArray = new float[numNeurons];
        //copy values
	    for (int j = 0; j < usedNeurons; j++) {
			        inputArray[j] = hiddenNeurons[j].getoldresponse();
		}
		
		return inputArray;
	}

    //compute the bottom-up preResponse
	public void computeBottomUpResponse(float[][] sensorInput, int[] sensorSize) {
		// Keep track of the sensor Input
		int beginIndex = 0;
		for (int j = 0; j < sensorSize.length; j++) {
			System.arraycopy(sensorInput[j], 0, currentBottomUpInput, beginIndex, sensorSize[j]);
			beginIndex += sensorSize[j];
		}
        //copy the current bottom-up input
	    float[] currentInput = new float[numBottomUpWeights];

	    System.arraycopy(currentBottomUpInput, 0, currentInput,0, numBottomUpWeights);

        for(int i=0; i<usedNeurons; i++){
			    hiddenNeurons[i].computeBottomUpResponse(currentInput);
		}

	}
	
    //compute the bottom-up preResponse
	public void computeBottomUpResponseInParallel(float[][] sensorInput, int[] sensorSize) {
		float[][] currentInput = new float[sensorSize.length][];
		// Keep track of the sensor Input
		int beginIndex = 0;
		for (int j = 0; j < sensorSize.length; j++) {
			System.arraycopy(sensorInput[j], 0, currentBottomUpInput, beginIndex, sensorSize[j]);
			beginIndex += sensorSize[j];
		
			//copy the current bottom-up input
			currentInput[j] = new float[sensorSize[j]];
			System.arraycopy(sensorInput[j], 0, currentInput[j], 0, sensorSize[j]);
		}
		for(int i=0; i<usedNeurons; i++){
			    hiddenNeurons[i].computeBottomUpResponseInParallel(currentInput);
		}
	}

    //compute the top-down preResponse
	public void computeTopDownResponse(float[][] motorInput, int[] motorSize) {
        // Keep track of the top-down Input
		int beginIndex = 0;
		for (int j = 0; j < motorSize.length; j++) {
			System.arraycopy(motorInput[j], 0, currentTopDownInput, beginIndex, motorSize[j]);
			beginIndex += motorSize[j];
		}
        //copy the top-down input
	   float[] currentInput = new float[numTopDownWeights];

	   System.arraycopy(currentTopDownInput, 0, currentInput, 0, numTopDownWeights);


			// If using SM.
			for(int i=0; i<usedNeurons; i++){
			    hiddenNeurons[i].computeTopDownResponse(currentInput);
		}
	}

    //compute the lateral preResponse
	public void computeLateralResponse(){
		if(isPriLateral){
			float[] temResponse = new float[PriLateralsize+numNeurons];
			System.arraycopy(priLateralvector, 0, temResponse, numNeurons, PriLateralsize);
			float[] tempResponse2 = this.getResponse1D();
			System.arraycopy(tempResponse2, 0, temResponse, 0, numNeurons);
			
			for(int i = 0; i < usedNeurons; i++){
				hiddenNeurons[i].computeLateralResponse(temResponse, lateralPercent);
			}
		}
		else{
			float[] tempResponse = new float[numNeurons];
			tempResponse = this.getResponse1D();
//			tempResponse = this.preResponse;

			for(int i=0;i<usedNeurons; i++){
				hiddenNeurons[i].computeLateralResponse(tempResponse,lateralPercent);
			
			}
		}
	}

    //compute the final response
	public void computeResponse(int whereId, boolean learn_flag, DN2.MODE mode) {
		prescreenResponse();
		
		for(int i=0;i<usedNeurons; i++){
			 hiddenNeurons[i].computeResponse();
		}
	
        // do the topKcompetition
		float temp = (float)usedNeurons/numNeurons;
		int index = 0;
		if (mode == DN2.MODE.GROUP || mode == DN2.MODE.Speech) {
			index = (int)(temp/0.02f);
		}
		if(mode == DN2.MODE.MAZE){
		    index = (int)(temp/0.05f);
		}
		if(usedNeurons == numNeurons){
			index = index-1;
		}
		/*
		int index2 = 0;
		if(type5g){
			index2 = 4;
		}*/
		
		//dynamic top-k competition for each type neurons
        if(dynamicInhibition){
        	if(typeindex[4]!=0 && type5g == true){
		        dynamictopKCompetition(whereId, learn_flag,5,mGrowthRate[index][5], mode);
		    }
        	if(typeindex[0]!=0){
		        dynamictopKCompetition(whereId, learn_flag,1,mGrowthRate[index][1], mode);
		    }
        	if(typeindex[3]!=0){
		        dynamictopKCompetition(whereId, learn_flag,4,mGrowthRate[index][4], mode);
		    }
        	if(typeindex[2]!=0  && type3g == true){
		        dynamictopKCompetition(whereId, learn_flag,3,mGrowthRate[index][3], mode);
		    }
        	if(typeindex[5]!=0){
		        dynamictopKCompetition(whereId, learn_flag,6,mGrowthRate[index][6], mode);
		    }
        	if(typeindex[6]!=0  && type7g == true){
		        dynamictopKCompetition(whereId, learn_flag,7,mGrowthRate[index][7], mode);
		    }
        	if(typeindex[1]!=0){
		        dynamictopKCompetition(whereId, learn_flag,2,mGrowthRate[index][2], mode);
		    }
		}
        //global top-k competition for each type neurons
        else{
        	if(typeindex[4]!=0){
        		topKCompetition(whereId, learn_flag,5,mGrowthRate[index][5], mode);
        	}
        	if(typeindex[0]!=0){
        	    topKCompetition(whereId, learn_flag,1,mGrowthRate[index][1], mode);
        	}
        	if(typeindex[3]!=0){
        	    topKCompetition(whereId, learn_flag,4,mGrowthRate[index][4], mode);
        	}        	
        	if(typeindex[1]!=0){
        	    topKCompetition(whereId, learn_flag,2,mGrowthRate[index][2], mode);
        	}
        	if(typeindex[2]!=0){
        	    topKCompetition(whereId, learn_flag,3,mGrowthRate[index][3], mode);
        	}
        	if(typeindex[5]!=0){
        	    topKCompetition(whereId, learn_flag,6,mGrowthRate[index][6], mode);
        	}
        	if(typeindex[6]!=0){
        	    topKCompetition(whereId, learn_flag,7,mGrowthRate[index][7], mode);
        	}
        }
	}

	public void computeResponse(int whereId, boolean learn_flag, DN2.MODE mode, int type) {
		prescreenResponse();
		
		for(int i=0;i<usedNeurons; i++){
			if(hiddenNeurons[i].getType() == type){
				hiddenNeurons[i].computeResponse();
			 }
			else{
				hiddenNeurons[i].setnewresponse(0);
			}
		}
	
        // do the topKcompetition
		float temp = (float)usedNeurons/numNeurons;
		int index = 0;
		if (mode == DN2.MODE.GROUP || mode == DN2.MODE.Speech) {
			index = (int)(temp/0.02f);
		}
		if(mode == DN2.MODE.MAZE){
		    index = (int)(temp/0.05f);
		}
		if(usedNeurons == numNeurons){
			index = index-1;
		}
		//dynamic top-k competition for each type neurons
		dynamictopKCompetition(whereId, learn_flag, type, mGrowthRate[index][type], mode);		
	}

    //do prescreening
    private void prescreenResponse() {
        // Prescreen bottomUpResponse
        float[] tempArray = new float[numNeurons];
        for(int i = 0; i < numNeurons; i++){
        	tempArray[i] = hiddenNeurons[i].getbottomUpresponse();
        }
        //sort the bottom-up preResponses
        Arrays.sort(tempArray);
        int cutOffPos = (int) Math.ceil((double)tempArray.length * (double)prescreenPercent);
        if(cutOffPos >= numNeurons-1) cutOffPos = numNeurons-1;
        float cutOffValue = tempArray[cutOffPos];
        for (int i = 0; i < numNeurons; i++){
          if(hiddenNeurons[i].getbottomUpresponse() < cutOffValue){
        	  hiddenNeurons[i].setbottomUpresponse(0);
          }
        }

        // Prescreen topDownResponse
        tempArray = new float[numNeurons];
        for (int i = 0; i < numNeurons; i++){
        	tempArray[i] = hiddenNeurons[i].gettopDownresponse();
        }
        //sort the top-down preResponses
        Arrays.sort(tempArray);
        cutOffPos = (int) Math.ceil((double)tempArray.length * (double)prescreenPercent);
        if(cutOffPos >= numNeurons-1) cutOffPos = numNeurons-1;
        cutOffValue = tempArray[cutOffPos];
        for (int i = 0; i < numNeurons; i++){
          if(hiddenNeurons[i].gettopDownresponse() < cutOffValue){
        	  hiddenNeurons[i].settopDownresponse(0);
          }
        }
        
     
     // Prescreen lateralExcitationResponse
        tempArray = new float[numNeurons];
        for (int i = 0; i < numNeurons; i++){
        	tempArray[i] = hiddenNeurons[i].getlateralExcitationresponse();
        }
        //sort the lateral preResponses
        Arrays.sort(tempArray);
        cutOffPos = (int) Math.ceil((double)tempArray.length * (double)prescreenPercent);
        if(cutOffPos >= numNeurons-1) cutOffPos = numNeurons-1;
        cutOffValue = tempArray[cutOffPos];
        for (int i = 0; i < numNeurons; i++){
          if(hiddenNeurons[i].getlateralExcitationresponse() < cutOffValue){
        	  hiddenNeurons[i].setlateralExcitationresponse(0);
          }
        }

      }
   


	// Sort the topK elements to the beginning of the sort array where the index
	// of the top
	// elements are still in the pair.
	private static void topKSort(Pair[] sortArray, int topK) {

		for (int i = 0; i < topK; i++) {
			Pair maxPair = sortArray[i];
			int maxIndex = i;

			for (int j = i + 1; j < sortArray.length; j++) {
            // select temporary max value
                if (sortArray[j].value > maxPair.value) {
															
					maxPair = sortArray[j];
					maxIndex = j;

				}
			}
            // store the value of pivot (top i) element
			if (maxPair.index != i) {
				Pair temp = sortArray[i];
				// replace with the maxPair object
				sortArray[i] = maxPair;
                //replace maxPair index elements with the pivot
				sortArray[maxIndex] = temp;
			}
		}
	}

	private void topKCompetition(int attentionId, boolean learn_flag, int type, float perfectmatch, DN2.MODE mode) {

		// initializing the indexes
		// winnerIndex is only for the winner neurons among the active neurons.
		int winnerIndex = 0;

		float[] copyArray = new float[usedNeurons];

		// Pair is an object that contains the (index,response_value) of each
		// hidden neurons.
		Pair[] sortArray = new Pair[usedNeurons];
        
//		System.out.println("get neuron "+i+" response: "+hiddenNeurons[i].getnewresponse());
		for(int j=0; j<usedNeurons; j++){
			        if(areaMask[type-1][j]!=0){
			            sortArray[j] = new Pair(j, hiddenNeurons[j].getnewresponse());
			            copyArray[j] = hiddenNeurons[j].getnewresponse();
			            preResponse[j] = hiddenNeurons[j].getnewresponse();
			
			            hiddenNeurons[j].setnewresponse(0.0f);
			            hiddenNeurons[j].setwinnerflag(false);
			            inhibitoryNeurons[j].setwinnerflag(true);
			        }
			        else{
			        	sortArray[j] = new Pair(j, 0);
			            copyArray[j] = 0;
			        }
	     }
		

		// Sort the array of Pair objects by its response_value in
		// non-increasing order.
		// The index is in the Pair, goes with the response ranked.
		topKSort(sortArray, topK);

//		System.out.println("High top1 value of hidden: " + sortArray[0].value);

		// check if the top winner has almost perfect match.
//		System.out.println(sortArray[0].value < (ALMOST_PERFECT_MATCH_RESPONSE+1.0f*lateralPercent) && usedHiddenNeurons < numNeurons);

		if (learn_flag) {
			if (sortArray[0].value < (ALMOST_PERFECT_MATCH_RESPONSE)*perfectmatch && usedNeurons < numNeurons && perfectmatch>0.1f) { // add
				System.out.println("Top 1 response: "+sortArray[0].value+", it index: "+sortArray[0].index);

					hiddenNeurons[usedNeurons].setwinnerflag(true);
					inhibitoryNeurons[usedNeurons].setwinnerflag(false);
					hiddenNeurons[usedNeurons].setnewresponse(1.0f);// set to perfect match.
					hiddenNeurons[usedNeurons].setType(type);
				    if(type==2||type==3||type==6||type==7){
						for (int j = 0; j < numNeurons; j++) {
							hiddenNeurons[usedNeurons].setLateralMask(1, j);
				        }
					}
					inhibitoryNeurons[usedNeurons].setType(type);
					initializeRfMask(usedNeurons, attentionId, mode); 
					preResponse[usedNeurons] = 1-MACHINE_FLOAT_ZERO;																		
				//	setwinnerIndex(0,hiddenNeurons[usedHiddenNeurons].getindex());
					float[] temp1 = hiddenNeurons[sortArray[0].get_index()].getlocation();
					float[] temp2 = new float [3];
					temp2 = setnormvector(temp2);
					float[] temp = {temp1[0]+temp2[0]*10*MACHINE_FLOAT_ZERO, temp1[1]+temp2[1]*10*MACHINE_FLOAT_ZERO, temp1[2]+temp2[2]*10*MACHINE_FLOAT_ZERO};
					hiddenNeurons[usedNeurons].setlocation(temp);					
					areaMask[type-1][usedNeurons]=1.0f;
					typeindex[type-1] += 1;					
					inhibitoryNeurons[usedNeurons].setLateralWeights(areaMask[type-1]);
					
					for(int k=0; k<usedNeurons; k++){
						if(inhibitoryNeurons[k].getType() == type){
							inhibitoryNeurons[k].setLateralWeight(1,usedNeurons);
						}
					}
					usedNeurons++;
					winnerIndex++;					
			}
		}

		// identify the ranks of topk winners
		float value_top1 = sortArray[0].value;
		float value_topkplus1 = sortArray[topK].value; // this neurons is the
														// largerst response
														// neurons to be set to
														// zero.

		// Find the topK winners and their indexes.
		while (winnerIndex < topK && perfectmatch>0.1f) {
			// get the index of the top element.
			int topIndex = sortArray[winnerIndex].get_index();
		//	setwinnerIndex(winnerIndex,topIndex);
			float tempnew= (copyArray[topIndex] - value_topkplus1)
					/ (value_top1 - value_topkplus1 + MACHINE_FLOAT_ZERO);
			hiddenNeurons[topIndex].setnewresponse(tempnew);
			winnerIndex++;

			hiddenNeurons[topIndex].setwinnerflag(true);
			inhibitoryNeurons[topIndex].setwinnerflag(false);
			System.out.println("winner neuron: "+ topIndex);
		}
		if(!learn_flag){
		    System.out.println("********************************************");
		}
        System.out.println("used number of type "+type+" neurons: "+typeindex[type-1]);
	}

	private void dynamictopKCompetition(int attentionId, boolean learn_flag, int type, float percent, DN2.MODE mode) {

		// initializing the indexes
		// winnerIndex is only for the winner neurons among the active neurons.
		int numwinner = topK;
        boolean concept = true;
		float[] copyArray = new float[numNeurons];

		// Pair is an object that contains the (index,response_value) of each
		// hidden neurons.
		Pair[] sortArray = new Pair[usedNeurons];

		for(int j=0; j<usedNeurons; j++){
			        if(areaMask[type-1][j]!=0){
	//		        	System.out.print("neuron"+j+" type "+type+" reached: "+hiddenNeurons[j].getnewresponse()+" ");
			            sortArray[j] = new Pair(j, hiddenNeurons[j].getnewresponse());
			            copyArray[j] = hiddenNeurons[j].getnewresponse();
			            preResponse[j] = hiddenNeurons[j].getnewresponse();
			
			            hiddenNeurons[j].setnewresponse(0.0f);
			            hiddenNeurons[j].setwinnerflag(false);
			            inhibitoryNeurons[j].setwinnerflag(true);
			        }
			        else{
			            sortArray[j] = new Pair(j, 0);
			            copyArray[j] = 0;
			        }
	     }
		for(int i=usedNeurons; i<numNeurons;i++){
			 copyArray[i] = 0;
		}
		// Sort the array of Pair objects by its response_value in
		// non-increasing order.
		// The index is in the Pair, goes with the response ranked.
		topKSort(sortArray, topK);

//		System.out.println("High top1 value of hidden: " + sortArray[0].value);

		// check if the top winner has almost perfect match.
//		System.out.println(sortArray[0].value < (ALMOST_PERFECT_MATCH_RESPONSE+1.0f*lateralPercent) && usedHiddenNeurons < numNeurons);
		 if(type == 3){
				concept = type3;				
		 }
		 else{
				concept = true;
		 }
		 if(type == 4){
				learn_flag = type4learn;				
		 }

		if (learn_flag) {
			if (sortArray[0].value < (ALMOST_PERFECT_MATCH_RESPONSE)*percent && usedNeurons < numNeurons && percent>0.001f && concept) { // add
				   System.out.println("Top 1 response: "+sortArray[0].value+", it index: "+sortArray[0].index);

				    hiddenNeurons[usedNeurons].setwinnerflag(true);
				    inhibitoryNeurons[usedNeurons].setwinnerflag(false);
				    hiddenNeurons[usedNeurons].setnewresponse(1.0f);// set to perfect match.
				    hiddenNeurons[usedNeurons].setType(type);
				    if(type==2||type==3||type==6||type==7){
						for (int j = 0; j < numNeurons+PriLateralsize; j++) {
							hiddenNeurons[usedNeurons].setLateralMask(1, j);
						}
				    }
				    inhibitoryNeurons[usedNeurons].setType(type);
				    initializeRfMask(usedNeurons, attentionId, mode); 
				    preResponse[usedNeurons] = 1-MACHINE_FLOAT_ZERO;
				//	setwinnerIndex(0,hiddenNeurons[usedHiddenNeurons].getindex());
					float[] temp1 = hiddenNeurons[sortArray[0].get_index()].getlocation();
					float[] temp2 = new float [3];
					temp2 = setnormvector(temp2);
					float[] temp = {temp1[0]+temp2[0]*10*MACHINE_FLOAT_ZERO, temp1[1]+temp2[1]*10*MACHINE_FLOAT_ZERO, temp1[2]+temp2[2]*10*MACHINE_FLOAT_ZERO};
					hiddenNeurons[usedNeurons].setlocation(temp);
					
					areaMask[type-1][usedNeurons]=1.0f;
					typeindex[type-1] += 1;					
					inhibitoryNeurons[usedNeurons].setLateralWeights(areaMask[type-1]);
					inhibitoryNeurons[usedNeurons].setLateralMasks(areaMask[type-1]);
					
					for(int k=0; k<usedNeurons; k++){
						if(inhibitoryNeurons[k].getType() == type){
							inhibitoryNeurons[k].setLateralWeight(1,usedNeurons);
							inhibitoryNeurons[k].setLateralMask(1,usedNeurons);
						}
					}
//track winner index
					if(type == 7){
						setWinner(usedNeurons);
					}
					usedNeurons++;
					numwinner--;					
				
			}
		}
				  

		
		if(numwinner!=0 && percent>0.001f && concept){

            for(int j=0; j< usedNeurons; j++){
                if(areaMask[type-1][j] != 0){
				        float[] tempResponse = elementWiseProduct(copyArray, inhibitoryNeurons[j].getLateralMask());
//				        System.out.println("mask: "+inhibitoryNeurons[j].getLateralMask()[0]+" "+inhibitoryNeurons[j].getLateralMask()[1]+" "+inhibitoryNeurons[j].getLateralMask()[2]);
//				        System.out.println("product: "+tempResponse[0]+" "+tempResponse[1]+" "+tempResponse[2]);
//				        System.out.println("res: "+copyArray[0]+" "+copyArray[1]+" "+copyArray[2]);
				        Arrays.sort(tempResponse);
				        if(copyArray[j] >= tempResponse[numNeurons-numwinner]){
				    	    float tempnew= (copyArray[j] - tempResponse[numNeurons-numwinner-1])
									/ (tempResponse[numNeurons-1] - tempResponse[numNeurons-numwinner-1] + MACHINE_FLOAT_ZERO);
							hiddenNeurons[j].setnewresponse(tempnew);

							hiddenNeurons[j].setwinnerflag(true);
//							if(!learn_flag){
								System.out.println("winner: neuron "+j+" the top: "+tempResponse[numNeurons-numwinner]);
//track winner index		
							if(type == 7){	
								setWinner(j);
							}
//							}
				            inhibitoryNeurons[j].setwinnerflag(false);
				       }
				}
            }
	    }
        System.out.println("Used number of type "+type+" neurons: "+typeindex[type-1]);

	}

    //convert the new responses to old responses
	public void replaceHiddenLayerResponse() {
		for(int j=0;j<numNeurons; j++){
			 hiddenNeurons[j].replaceResponse();
		}
	}

    //set y attend information
	public void setYattend(boolean y){
		yAttend = y;
	}

    //get the number of winner y neurons
	public int getTopK() {
		return topK;
	}

    //set the number of winner y neurons
	public void setTopK(int topK) {
		this.topK = topK;
	}

	public int getNumBottomUpWeights() {
		return numBottomUpWeights;
	}

	public void setNumBottomUpWeights(int numBottomUpWeights) {
		this.numBottomUpWeights = numBottomUpWeights;
	}

	public int getNumTopDownWeights() {
		return numTopDownWeights;
	}

	public void setNumTopDownWeights(int numTopDownWeights) {
		this.numTopDownWeights = numTopDownWeights;
	}

	public int getNumNeurons() {
		return numNeurons;
	}
	
	public int gettypeNum(){
		return numTypes;
	}
	
	public void setNeuronGrowthrate(int type, boolean a){
		if(type == 5){
			type5g = a;
		}
		if(type == 3){
			type3g = a;
		}
		if(type == 7){
			type7g = a;
		}
	}
	public void setConcept(boolean c3){
		type3 = c3;
	}

	public void setNeuronLearning(int type, boolean l5){
		if(type == 4){
			type4learn = l5;
		}
		if(type == 5){
			type5learn = l5;
		}
		if(type == 3){
			type3learn = l5;
		}
		if(type == 7){
			type7learn = l5;
		}
	}

	public void setPriLateralvector(float[] vector){
		for(int i = 0; i < vector.length; i++){
			priLateralvector[i] = vector[i];
		}
	}
	
	public void setBottomupMask(float[] mask, int index){
		hiddenNeurons[index].setBottomUpMasks(mask);
	}
	
	public void setTopdownMask(float[] mask, int index){
		hiddenNeurons[index].setTopDownMasks(mask);
	}
	
    //get the local receptive field
	public float[][] getRfMask() {
		float[][] bottomUpMask = new float [numNeurons][numBottomUpWeights];
		for(int i = 0; i < numNeurons; i++){
			System.arraycopy(hiddenNeurons[i].getBottomUpMask(),0,bottomUpMask[i],0,numBottomUpWeights);
		}
		return bottomUpMask;
	}

    //get the number of y neurons
	public int getUsedHiddenNeurons() {
		return usedNeurons;
	}
    //re-set responses for each neuron
	public void resetResponses() {

		for(int i = 0; i < numNeurons; i++){
			hiddenNeurons[i].resetResponses();
		}

	}
	
	public int getWinner(){
		return winner;
	}

	public void setWinner(int a){
		winner = a;
	}
	
    //multiply elements between 2 vectors
    public float[] elementWiseProduct(float[] vec1, float[] vec2) {
			assert vec1.length == vec2.length;
			int size = vec1.length;
			float[] result = new float[size];
			for (int i = 0; i < size; i++) {
				result[i] = vec1[i] * vec2[i];
			}
			return result;
    }
	  
	public class Pair implements Comparable<Pair> {
		public final int index;
		public final float value;

		public Pair(int index, float value) {
			this.index = index;
			this.value = value;
		}

		public int compareTo(Pair other) {
			return -1 * Float.valueOf(this.value).compareTo(other.value);
		}

		public int get_index() {
			return index;
		}
	}
   //calculate the mean value for the non-zero elements in a vector
	public static float mean(float[] m, float[] mask) {
		float sum = 0;
		float count = 0;
		for (int i = 0; i < m.length; i++) {
			if (mask[i] != 0) {
				sum += m[i];
				count++;
			}
		}
		return sum / count;
	}

    //calculate the mean value for a vector
	public static float mean(float[] m) {
		float sum = 0;
		float count = 0;
		for (int i = 0; i < m.length; i++) {
			if (m[i] > 0) {
				sum += m[i];
				count++;
			}
		}
		return sum / count;
	}
	/*
	 * This is the protocol for toy data visualization. All data are transfered
	 * as float to save effort in translation. There are these things to send
	 * over socket: 1. number of hidden neurons 2. length of bottomUp input 3.
	 * bottom up weight 4. bottom up mask 5. length of topDown input 6. topDown
	 * Age 7. topDown Mask 8. topDown Variance 9. topdown Weight 10.bottom up
	 * input 11.top down input
	 */
	
	
	public void sendNetworkOverSocket(PrintWriter string_out, DataOutputStream data_out, int display_y_zone, int display_y2_zone,
			int display_num, int display_start_id) throws IOException, InterruptedException {
		int start_id = display_start_id - 1;
		if (start_id < 0)
			start_id = 0;
		if (start_id >= numNeurons)
			start_id = 0;
		int end_id = start_id + display_num;
		if (end_id > numNeurons)
			end_id = numNeurons;
		if (end_id < 0)
			end_id = numNeurons;

		// number of hidden neurons
		data_out.writeInt(end_id - start_id);

		// length of bottom up input
		data_out.writeInt(numBottomUpWeights);

		// length of topDown input
		data_out.writeInt(numTopDownWeights);

		data_out.writeInt(numGlial);
		
		data_out.writeInt(numNeurons);
        int usedHiddenNeurons=0;
		for(int i=0; i<typeindex.length; i++){
			usedHiddenNeurons += typeindex[i];
		}
		data_out.writeInt(usedHiddenNeurons);
				
		// bottom up weight
		if (display_y_zone == 1) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numBottomUpWeights; j++) {
					data_out.writeFloat((float) hiddenNeurons[i].getBottomUpWeights()[j] * hiddenNeurons[i].getBottomUpMask()[j]);
				}
			}
		}

		// bottom up age
		else if (display_y_zone == 2) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numBottomUpWeights; j++) {
					data_out.writeInt(hiddenNeurons[i].getbottomupage()[j]);
				}
			}
		}

		// bottom up mask
		else if (display_y_zone == 3) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numBottomUpWeights; j++) {
					data_out.writeFloat((float)hiddenNeurons[i].getBottomUpMask()[j]);
				}
			}
		}

		// bottom up age
		else if (display_y_zone == 4) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numBottomUpWeights; j++) {
					data_out.writeFloat((float) hiddenNeurons[i].getBottomUpVariances()[j]);
				}
			}
		}

		// topDown weight
		else if (display_y_zone == 5) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numTopDownWeights; j++) {
					data_out.writeFloat((float) hiddenNeurons[i].getTopDownWeights()[j]);
				}
			}
		}

		// topDown age
		else if (display_y_zone == 6) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numTopDownWeights; j++) {
					data_out.writeInt(hiddenNeurons[i].gettopdownage()[j]);
				}
			}
		}

		// topDown mask
		else if (display_y_zone == 7) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numTopDownWeights; j++) {
					data_out.writeFloat((float)hiddenNeurons[i].getTopDownMask()[j]);
				}
			}
		}

		// topDown variance
		else if (display_y_zone == 8) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numTopDownWeights; j++) {
					data_out.writeFloat((float) hiddenNeurons[i].getTopDownVariances()[j]);
				}
			}
		}
		 //lateral excitation weights
		if (display_y2_zone == 1) {
			for (int i = start_id; i < end_id; i++) {
			    for(int j = 0; j < numNeurons; j++){
				    data_out.writeFloat((float) hiddenNeurons[i].getLateralWeights()[j]);
			    }
		    }
		} 
		 //lateral excitation ages	
		if (display_y2_zone == 2) {
			for (int i = start_id; i < end_id; i++) {
			    for(int j = 0; j < numNeurons; j++){
				    data_out.writeInt(hiddenNeurons[i].getlateralage()[j]);
			    }
		    }
		} 
		 //lateral excitation masks
		if (display_y2_zone == 3) {
		    for (int i = start_id; i < end_id; i++) {
			    for(int j = 0; j < numNeurons; j++){
				    data_out.writeFloat((float) hiddenNeurons[i].getLateralMask()[j]);
			    }
		    }
		} 
		 //lateral excitation variances
		if (display_y2_zone == 4) {
			for (int i = start_id; i < end_id; i++) {
			    for(int j = 0; j < numNeurons; j++){
				    data_out.writeFloat((float) hiddenNeurons[i].getLateralVariances()[j]);
			    }
		    }
		} 
		 //inhibition weights
		if (display_y2_zone == 5) {
			for (int i = start_id; i < end_id; i++) {
			    for(int j = 0; j < numNeurons; j++){
				    data_out.writeFloat((float) inhibitoryNeurons[i].getLateralWeights()[j]);
			    }
		    }
		} 
		// bottom up input
		for (int i = 0; i < currentBottomUpInput.length; i++) {
			data_out.writeFloat((float) currentBottomUpInput[i]);
		}

		// top down input
		for (int i = 0; i < currentTopDownInput.length; i++) {
			data_out.writeFloat((float) currentTopDownInput[i]);
		}

		// bottom up response
		for (int i = start_id; i < end_id; i++) {
			data_out.writeFloat((float) hiddenNeurons[i].getbottomUpresponse());
		}

		// top down response
		for (int i = start_id; i < end_id; i++) {
			data_out.writeFloat((float) hiddenNeurons[i].gettopDownresponse());
		}

		// final response
		for (int i = start_id; i < end_id; i++) {
			data_out.writeFloat((float) hiddenNeurons[i].getnewresponse());
		}
				
		for(int i = 0; i < numGlial; i++){
			for(int j = 0; j < 3; j++){
				data_out.writeFloat((float) glialcells[i].getlocation()[j]);
			}
		}
		
		for (int i = 0; i < usedHiddenNeurons; i++){
			for (int j = 0; j < 3; j++){
				data_out.writeFloat((float)hiddenNeurons[i].getlocation()[j]);
			}
		}
		
	    for (int i = 0; i < usedHiddenNeurons; i++){
	    	 
	        data_out.writeFloat((float)hiddenNeurons[i].getType());
	   
	    }
	}
	
	public float[] setnormvector(float[] vector){
		Random ran = new Random();
		for(int i = 0; i < vector.length; i++){
			vector[i] = ran.nextFloat();
		}
		norm(vector);
		return vector;
	}

	public void setNumNeurons(int numNeurons) {
		this.numNeurons = numNeurons;
	}
	/*
	public int[] getwinnerIndexs() {
		return winnerIndexs;
	}

	public void setwinnerIndex(int topk, int index) {
		this.winnerIndexs[topk] = index;
	}*/
	
	//calculate the mean value
	public float mean(float[] m, int[] length) {
		float sum = 0;
		float count = 0;
		int temp = 0;
		for (int i = 0; i < length.length; i++) {
			if(length[i]!=0){
				for(int j=temp*numNeurons; j<temp*numNeurons+length[i]; j++){
			        if (m[j] > 0) {
				        sum += m[j];
				        count++;}
				}
			temp++;
			}
		}
		return sum / count;
	}
	
	//L-2 normalization
	public float[] norm(float[] weight){
		float norm = 0;
		int size = weight.length;
		for (int i = 0; i < size; i++){
			norm += weight[i]* weight[i];
		}
		norm = (float) Math.sqrt(norm);
		if (norm > 0){
			for (int i = 0; i < size; i++){
				weight[i] = weight[i]/norm;
				}
		}	
		return weight;
	}
	//find the top-k neighbors for each glial cell 
	public void topKNeighborSort(Pair[] sortArray, int topK){
		
		for (int i = 0; i < topK; i++) {
			Pair minPair = sortArray[i]; 
			int minIndex = i;
					
			for (int j = i+1; j < sortArray.length; j++) {
				
				if(sortArray[j].value < minPair.value){ // select temporary max
					minPair = sortArray[j];
					minIndex = j;
					
				}
			}
			
			if(minPair.index != i){
				Pair temp = sortArray[i]; // store the value of pivot (top i) element
				sortArray[i] = minPair; // replace with the maxPair object.
				sortArray[minIndex] = temp; // replace maxPair index elements with the pivot. 
			}
		}
	}
	
	//pull the y neurons locations away
	public void pullneurons(float pullrate){
		int minvalue;
		if(usedNeurons < glialcells[0].gettopk()){
			minvalue = usedNeurons;
		}
		else{
			minvalue = glialcells[0].gettopk();
		}
		for(int i = 0;i < numGlial; i++){
			Pair[] distances =new Pair [usedNeurons];

		    for(int k=0; k<usedNeurons; k++){
				  distances[k] = new Pair(k,glialcells[i].computeDistance(hiddenNeurons[k].getlocation()));
		    }
			
		
			topKNeighborSort(distances,minvalue);
			for(int k = 0;k < minvalue; k++){
				glialcells[i].setpullindex(k, distances[k].index);
				float[] dis =new float[3];
				for(int m = 0; m < 3; m++){
					dis[m]=glialcells[i].getlocation()[m] - hiddenNeurons[distances[k].index].getlocation()[m];
				}
				glialcells[i].setpullvector(dis,k);
			}
		}
	    for(int k =0; k<usedNeurons; k++){	
			    float[] pullvector = new float [3];
			    for(int l = 0;l < numGlial; l++){
				    for(int h = 0;h < minvalue; h++){
					    if(glialcells[l].getpullindex(h) == hiddenNeurons[k].getindex()){
						    for(int m = 0; m < 3; m++){
							    pullvector[m] += glialcells[l].getpullvector(h)[m];
						    }
					    }
				    }
			    }
			    float[] temp = new float[3];
			    pullvector = norm(pullvector);
			    for(int m = 0; m < 3; m++){
				    temp[m] = hiddenNeurons[k].getlocation()[m]+pullrate*pullvector[m];
			    }
			    hiddenNeurons[k].setlocation(temp);
			  }
	}
	
	//calculate the distances between other neurons and record the indexs of neighbors
	public void calcuateNeiDistance() {
		 Pair[][] dist = new Pair[usedNeurons][usedNeurons-1];
		 for(int i =0; i < usedNeurons; i++){	
			 for(int j =i+1; j < usedNeurons; j++){	
				 float dis = hiddenNeurons[i].computeDistance(hiddenNeurons[j].getlocation());
				 dist[i][j-1] = new Pair(j,dis);
				 dist[j][i] = new Pair(i,dis);
			 }
			 topKNeighborSort(dist[i],addNum);
			 int[] NeiIndex = new int[addNum];
			 for(int k = 0; k < addNum; k++) {
				 NeiIndex[k] = dist[i][k].get_index();
			 }
			 hiddenNeurons[i].setneiIndex(NeiIndex);
		 }
	}
	
	//add new lateral connections for each neuron
	public void growLateralConnection(int j, int[] list) {
		for(int i = 0; i < list.length; i++) {
			for(int k = 0; k < addNum; k++) {
				if(i<numNeurons){
					int index = hiddenNeurons[i].getneiIndex()[k];
					boolean g = false;
					int t = 0;
				//add the neuron's neighbor lateral connection's neighbor (j-1)-th in the connections
					if ((hiddenNeurons[j].InlateralMask(index) == false)&&(hiddenNeurons[j].InlateralDeletelist(index) == false)) {
						g = true;
						hiddenNeurons[j].addLateralMasks(index);

					}
				}
			
/*				while(g) {
					if(t <1) {
						System.out.println("excuted!");
						System.out.println("neuron "+j+" add neuron "+i+"'s neighbor: neuron "+ index);
						t += 1;
						}
					if(getKey()) {
						g = false;
					}
				}*/
			}
		}
	}
	
    //send the y neurons' weights
	public float[][] send_y(int display_num, int display_start_id, int type){
		//get the specific y neurons 
		float[][] weights;
		int start_id = display_start_id - 1;
		if (start_id < 0)
			start_id = 0;
		if (start_id >= numNeurons)
			start_id = 0;
		int end_id = start_id + display_num;
		if (end_id > numNeurons)
			end_id = numNeurons;
		if (end_id < 0)
			end_id = numNeurons;
				
		// bottom up weight
		if (type == 1) {
			weights = new float[display_num][numBottomUpWeights];
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numBottomUpWeights; j++) {
					weights[i][j] = hiddenNeurons[i].getBottomUpWeights()[j];
				}
			}
			return weights;	
		}

		// bottom up age
		else if (type == 2) {
			weights = new float[display_num][numBottomUpWeights];
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numBottomUpWeights; j++) {
					weights[i][j] = hiddenNeurons[i].getbottomupage()[j];
				}
			}
			return weights;	
		}

		// bottom up mask
		else if (type == 3) {
			weights = new float[display_num][numBottomUpWeights];
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numBottomUpWeights; j++) {
					weights[i][j] = hiddenNeurons[i].getBottomUpMask()[j];
				}
			}
			return weights;	
		}

		// bottom up variance
		else if (type == 4) {
			weights = new float[display_num][numBottomUpWeights];
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numBottomUpWeights; j++) {
					weights[i][j] = hiddenNeurons[i].getBottomUpVariances()[j];
				}
			}
			return weights;	
		}

		// topDown weight
		else if (type == 5) {
			weights = new float[display_num][numTopDownWeights];
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numTopDownWeights; j++) {
					weights[i][j] = hiddenNeurons[i].getTopDownWeights()[j];
				}
			}
			return weights;	
		}

		// topDown age
		else if (type == 6) {
			weights = new float[display_num][numTopDownWeights];
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numTopDownWeights; j++) {
					weights[i][j] = hiddenNeurons[i].gettopdownage()[j];
				}
			}
			return weights;	
		}

		// topDown mask
		else if (type == 7) {
			weights = new float[display_num][numTopDownWeights];
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numTopDownWeights; j++) {
					weights[i][j] = hiddenNeurons[i].getTopDownMask()[j];
				}
			}
			return weights;	
		}

		// topDown variance
		else if (type == 8) {
			weights = new float[display_num][numTopDownWeights];
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numTopDownWeights; j++) {
					weights[i][j] = hiddenNeurons[i].getTopDownVariances()[j];
				}
			}
			return weights;	
		}
        //lateral excitation weights
		if (type == 9) {
			weights = new float[display_num][numNeurons];
			for (int i = start_id; i < end_id; i++) {
			    for(int j = 0; j < numNeurons; j++){
			    	weights[i][j] = hiddenNeurons[i].getLateralWeights()[j];
			    }
		    }
			return weights;	
		} 
		//lateral excitation ages
		if (type == 10) {
			weights = new float[display_num][numNeurons];
			for (int i = start_id; i < end_id; i++) {
			    for(int j = 0; j < numNeurons; j++){
			    	weights[i][j] = hiddenNeurons[i].getlateralage()[j];
			    }
		    }
			return weights;	
		} 
		//lateral excitation masks
		if (type == 11) {
			weights = new float[display_num][numNeurons];
		    for (int i = start_id; i < end_id; i++) {
			    for(int j = 0; j < numNeurons; j++){
			    	weights[i][j] = hiddenNeurons[i].getLateralMask()[j];
			    }
		    }
		    return weights;	
		} 
		//lateral excitation variances
		if (type == 12) {
			weights = new float[display_num][numNeurons];
			for (int i = start_id; i < end_id; i++) {
			    for(int j = 0; j < numNeurons; j++){
			    	weights[i][j] = hiddenNeurons[i].getLateralVariances()[j];
			    }
		    }
			return weights;	
		} 
		//inhibition weights
		if (type == 13) {
			weights = new float[display_num][numNeurons];
			for (int i = start_id; i < end_id; i++) {
			    for(int j = 0; j < numNeurons; j++){
			    	weights[i][j] = inhibitoryNeurons[i].getLateralWeights()[j];
			    }			    
		    }
			return weights;	
		}
		//inhibition areas
		if (type == 14) {
			weights = new float[display_num][numNeurons];
			for (int i = start_id; i < end_id; i++) {
			    for(int j = 0; j < numNeurons; j++){
			    	weights[i][j] = inhibitoryNeurons[i].getLateralMask()[j];
			    }			    
		    }
			return weights;	
		}
		return null;
	}
	
	public int sumConnections() {
		int result = 0;
		for(int i=0; i<usedNeurons; i++) {
			result += sumArray(hiddenNeurons[i].getBottomUpMask());
			result += sumArray(hiddenNeurons[i].getLateralMask());
			result += sumArray(hiddenNeurons[i].getTopDownMask());
		}
		return result;
	}
	
	public int sumArray(float[] a) {
		int result = 0;
		for(int i=0; i<a.length; i++) {
			if(a[i] != 0) {
				result += 1;
			}
		}
		return result;
	}
	
}
