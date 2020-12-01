package DN2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import javax.swing.Spring;

import DN2.DN2;
import DN2.InhibitoryNeuron;
import DN2.Neuron;
import DN2.DN2.MODE;
import DN2.HiddenLayer.Pair;

import java.io.*;

public class PrimaryHiddenLayer implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
    private DN2.MODE mode;
    private float mPerfectPercent;
	//the type of neuron
	private int mType;
	//the number of winner neurons
	private int topK;
	private int width;
	private int height;
    //the number of Y (hidden) neurons in different loc
	public int numNeurons;
	//the number of neuron in each loc
	public int mDepth;
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
	private int[] rfSize;
    //the stride of receiptive field
	private int[] rfStride;
    //the matrix recording receiptive field location
	private int[][] rf_id_loc;
    // The is the number of Y neurons
	private int usedNeurons;
	//private int[] winnerIndexs;
    //the vector recording the coefficient of mean value for dynamic top-k
    private float[][] mMeanValue;
    //machine zero value
	private final float MACHINE_FLOAT_ZERO = 0.0001f;
    //the perfect match value
	private final float ALMOST_PERFECT_MATCH_RESPONSE = 1.0f - 10 * MACHINE_FLOAT_ZERO;
    //the y neuron array
	public Neuron[][] hiddenNeurons;
    //the inhibitory neuron array
	public InhibitoryNeuron[][] inhibitoryNeurons;
    //the size of glial cells
	private int glialsize;
    //the number of glial cells
	public int numGlial;
    //the glial cell array
	public Glial[] glialcells;
    //the number of neighbor neurons being pulled
    private int glialtopK;
    private int[] startloc;

	//the construction of hidden layer
	public PrimaryHiddenLayer(int type, int topK, int sensorSize, int motorSize, int[][] inputSize, int depth, float[][] meanTable) {
		this.setTopK(topK);
		mPerfectPercent = 0.6f;      //0.6
		//winnerIndexs = new int [topK];
//		this.usedHiddenNeurons = topK + 1; // bound of used neurons.
        mode = DN2.MODE.GROUP;
		this.mType = type;
		//initialize the number of used neurons
        this.usedNeurons = 0;
		this.mDepth = depth;
		//size of local receptive field
		this.rfSize = new int[2];
		this.rfSize[0] = 5;   //5
		this.rfSize[1] = 4;   //4
		//moving step of local receptive field
		this.rfStride = new int[2];
		this.rfStride[0] = 2;
		this.rfStride[1] = 2;
		startloc = new int[4];   //4
		for(int i = 0; i < 4; i++){
			startloc[i] = 0;
		}
        //initialize the input size
		this.inputSize = inputSize[0];
		//array of generated local receptive fields
		this.rf_id_loc = configure_where_count(rfSize, rfStride,this.inputSize);
		//get the mean value table values
		this.mMeanValue = new float[meanTable.length][];
		for(int i = 0; i < meanTable.length; i++){
			mMeanValue[i] = new float[meanTable[i].length];
			System.arraycopy(meanTable[i], 0, mMeanValue[i], 0, meanTable[i].length);
		}
		height = (this.inputSize[0]-rfSize[0])/rfStride[0]+1;
		width = (this.inputSize[1]-rfSize[1])/rfStride[1]+1;
        //initialize the number of total neurons
		this.numNeurons = height*width;
        //initialize the neurons		
		hiddenNeurons = new Neuron[numNeurons][mDepth];
		//initialize each initial neuron 
		for(int i=0; i<numNeurons; i++){
			for(int j=0;j<mDepth;j++){
			        hiddenNeurons[i][j] = new Neuron(sensorSize, motorSize, 0, true, mType, i*mDepth+j, this.inputSize);		       
			}
		}
		
        //initialize the initial inhibitory neurons		
		inhibitoryNeurons = new InhibitoryNeuron[numNeurons][mDepth];
		for(int i=0; i<numNeurons; i++){
		    for(int j=0;j<mDepth;j++){
			        inhibitoryNeurons[i][j] = new InhibitoryNeuron(sensorSize,motorSize,numNeurons*mDepth, true,mType, i*mDepth+j,this.inputSize);
			}
		}
/*		
        //set initial neurons' locations		
		for(int i=0;i<height;i++){
			for(int j=0;j<width;j++){
			    for(int k=0;k<mDepth;k++){
			    	float[] temp = {1.0f*i, 1.0f*j, 1.0f*k};			    	
			    	hiddenNeurons[i*width+j][k].setlocation(temp);
			    	temp[2] = temp[2]+0.1f;
			    	inhibitoryNeurons[i*width+j][k].setlocation(temp);			    }
			}
		}
*/
		// Each type neuron initial two neurons have global receptive fields.
		for (int i = 0; i < numNeurons; i++) {
			for(int k=0;k<mDepth;k++){
				initializeRfMask(i, k, i, mode); 
				inhibitoryNeurons[i][k].setBottomUpMasks(hiddenNeurons[i][k].getBottomUpMask());
			    
			}
		}
		
		// topDownMask are ones at the beginning
		for (int i = 0; i < numNeurons; i++) {
			for(int k=0;k<mDepth;k++){
			    for (int j = 0; j < motorSize; j++) {
			    	hiddenNeurons[i][k].setTopDownMask(1, j);
			    	inhibitoryNeurons[i][k].setTopDownMask(1, j);
			    }
			}
		}	

		//inhibitory neurons' lateral mask only contain the same type neurons.
		for (int i = 0; i <height; i++) {
			int minHeight = 0;
			int maxHeight = 0;
			if(i-2<0){
				minHeight = 0;
			}
			else{
				minHeight = i-2;
			}
			if(i+2>=height){
				maxHeight = height-1;
			}
			else{
				maxHeight = i+2;
			}
			for (int j = 0; j <width; j++) {
				int minWidth = 0;
				int maxWidth = 0;

				if(j-2<0){
					minWidth = 0;
				}
				else{
					minWidth = j-2;
				}
				if(j+2>=width){
					maxWidth = width-1;
				}
				else{
					maxWidth = j+2;
				}
				for(int k=0;k<mDepth;k++){
					for(int ii = minHeight; ii<=maxHeight; ii++){
						for(int jj = minWidth; jj<=maxWidth; jj++){
							for(int m=0; m<mDepth; m++){
								inhibitoryNeurons[i*width+j][k].setLateralWeight(1, (ii*width+jj)*mDepth+m );
					        }
					    }
					}
				}
			}
		}
		//set the percentage of lateral excitation responses
		this.lateralPercent = 1.0f;
		//set the pre-response values
	    this.preResponse = new float[numNeurons*mDepth];
	    for(int i=0; i<numNeurons*mDepth; i++){
	    	preResponse[i] = 0;
	    }
		//get the bottom-up weights size
		numBottomUpWeights = sensorSize;
		currentBottomUpInput = new float[numBottomUpWeights];
		//get the top-down weights size
		numTopDownWeights = motorSize;
		currentTopDownInput = new float[motorSize];	
		
		//initialize the glial cells
        //set the number of neighbor neurons to be pulled
		glialtopK = 2;
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
    
	//save weights to a txt file
	public void saveWeightToFile(String hidden_ind) {
		try {
			PrintWriter wr_weight = new PrintWriter(new File(hidden_ind + "bottom_up_weight.txt"));
			PrintWriter wr_age = new PrintWriter(new File(hidden_ind + "age.txt"));
			PrintWriter wr_response = new PrintWriter(new File(hidden_ind + "response.txt"));
			PrintWriter wr_flag = new PrintWriter(new File(hidden_ind + "flag.txt"));
//			PrintWriter wr_inhibit = new PrintWriter(new File(hidden_ind + "inhibit.txt"));
//			PrintWriter wr_mask = new PrintWriter(new File(hidden_ind + "bottom_up_mask.txt"));
			for (int i = 0; i < numNeurons; i++) {
				for(int k = 0; k < mDepth; k++){
//					wr_weight.print(Integer.toString(hiddenNeurons[i][k].getindex()+1) + ' ');
//					wr_age.print(Integer.toString(hiddenNeurons[i][k].getindex()+1) + ' ');
//					wr_response.print(Integer.toString(hiddenNeurons[i][k].getindex()+1) + ' ');
					wr_age.print(Integer.toString(hiddenNeurons[i][k].getfiringage()) + ' ');
					wr_response.print(String.format("% .3f",hiddenNeurons[i][k].getnewresponse()) + ' ');
					if(hiddenNeurons[i][k].getState()){
						wr_flag.print(Integer.toString(1) + ' ');
					}
					else{
						wr_flag.print(Integer.toString(0) + ' ');
					}
				    for (int j = 0; j < numBottomUpWeights; j++) {
				    	if(hiddenNeurons[i][k].getBottomUpMask()[j] != 0){
				    		wr_weight.print(String.format("% .3f", hiddenNeurons[i][k].getBottomUpWeights()[j]) + ' ');				    		
//				    		wr_mask.print(Float.toString(hiddenNeurons[i][k].getBottomUpMask()[j]) + ',');
					    }
				    }
				
				    wr_weight.println();
				    wr_age.println();
				    wr_response.println();
				    wr_flag.println();
//				    wr_mask.println();
				}				
			}			
/*			for (int i = 0; i < numNeurons; i++) {	
				for(int k = 0; k < mDepth; k++){
					wr_inhibit.print(Float.toString(inhibitoryNeurons[71][0].getLateralWeights()[i*mDepth+k]) + ',');
				}
				wr_inhibit.println();
			}*/
			wr_weight.close();
			wr_age.close();
			wr_response.close();
			wr_flag.close();
//			wr_inhibit.close();
//			wr_mask.close();			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}


	// Initialize receptive field for the ith neuron, according to the whereID.
	// The center of the receptive field is located at rf_id_loc[i].
	// size of the receptive field is rf_size.
	public void initializeRfMask(int i, int depth, int whereID, DN2.MODE mode) {
        //initialize the receipt field for toy problem
		if (mode == DN2.MODE.GROUP) {
			if (whereID >= 0) {
	/*			int half_rf_size1 = (rfSize[0] - 1) / 2;
				int half_rf_size2 = (rfSize[1] - 1) / 2;
				int rf_begin_row = rf_id_loc[whereID][1] - half_rf_size1;
				int rf_end_row = rf_id_loc[whereID][1] + half_rf_size1;
				int rf_begin_col = rf_id_loc[whereID][0] - half_rf_size2;
				int rf_end_col = rf_id_loc[whereID][0] + half_rf_size2;    */
				// assert inputSize[0] * inputSize[1] ==
				// hiddenNeurons[i].getBottomUpMask().length;
				//initialize the bottom-up mask vector
				int rf_begin_row = rf_id_loc[whereID][0];
				int rf_end_row = rf_id_loc[whereID][0] + rfSize[0];
				int rf_begin_col = rf_id_loc[whereID][1];
				int rf_end_col = rf_id_loc[whereID][1] + rfSize[1];
				for (int col = rf_begin_col; col < rf_end_col; col++) {
					for (int row = rf_begin_row; row < rf_end_row; row++) {                   
						int current_idx = row * inputSize[1] + col;
						hiddenNeurons[i][depth].setBottomUpMask(1.0f, current_idx);
					}
				}
/*				for (int pixel_ind = inputSize[0] * inputSize[1]; pixel_ind < hiddenNeurons[i][depth].getBottomUpMask().length; pixel_ind++) {
					hiddenNeurons[i][depth].setBottomUpMask(1.0f, pixel_ind);
				}*/
			} else {
				for (int pixel_ind = 0; pixel_ind < hiddenNeurons[i][depth].getBottomUpMask().length; pixel_ind++) {
					hiddenNeurons[i][depth].setBottomUpMask(1.0f, pixel_ind);
				}
			}
			System.out.println("Rf initialized: " + whereID);
        //initialize the receipt field for maze problem
		} 
		/*else if (mode == DN2.MODE.MAZE){
			int rf_loc = DN2.getLocFromWhereID(whereID);
			int rf_size = DN2.getScaleFromWhereID(whereID);
			System.out.println("Decoding as: " + Integer.toString(rf_loc) + ", " + Integer.toString(rf_size));
			float[] bottom_up_mask = new float[currentBottomUpInput.length];
            //initialize the bottom-up mask vector
			if (rf_loc < 0){
				for (int j = 0; j < 3 * 2; j++){
					bottom_up_mask[j] = 1;
				}
			} else {
				for (int j = rf_loc * 3; j < (rf_loc + rf_size) * 3; j++){
					bottom_up_mask[j] = 1;
				}
			}
			for (int j = 3 * 2; j < currentBottomUpInput.length; j++){
				bottom_up_mask[j] = 1;
			}
			
			hiddenNeurons[i].setBottomUpMasks(bottom_up_mask);
		}*/
	}

    //hebbian learning for y neurons
	public void hebbianLearnHidden(float[] input) {
		boolean learning = true;
		float[] tempResponse1 = new float[numNeurons*mDepth];
		float[] tempResponse2 = new float[numNeurons*mDepth];
        //get the response from last frame
		for(int i=0;i<numNeurons;i++){
			for(int k=0;k<mDepth;k++){
				tempResponse1[i*mDepth+k] = hiddenNeurons[i][k].getoldresponse();
				tempResponse2[i*mDepth+k] = preResponse[i*mDepth+k];
//			winnerIndex[i] = hiddenNeurons[i].getwinnerflag();
//			System.out.println("neuron "+i+" winnerflag "+hiddenNeurons[i].getwinnerflag());
			}
		}
        for(int j=0; j<numNeurons; j++){
        	for(int k=0;k<mDepth;k++){
        		//update weights for each y neuron
        		hiddenNeurons[j][k].PrimaryhebbianLearnHidden(input);
        		//update weights for each inhibitory neuron
        		//inhibitoryNeurons[j][k].hebbianLearnHidden(input);
        	}
        }

	}

	// convert into 1d Array
	public float[] getResponse1D(float a) {
        //construct the new array
		float[] inputArray = new float[numNeurons*mDepth];
        //copy values
	    for (int i = 0; i < numNeurons; i++) {
	    	for(int j=0;j<mDepth;j++){
	    		inputArray[i*mDepth+j] = hiddenNeurons[i][j].getoldresponse()*a;
			}
		}
		
		return inputArray;
	}
	
	// convert into 1d Array
	public float[] getNewResponse1D() {
        //construct the new array
		float[] inputArray = new float[numNeurons*mDepth];
        //copy values
	    for (int i = 0; i < numNeurons; i++) {
	    	for(int j=0;j<mDepth;j++){
	    		inputArray[i*mDepth+j] = hiddenNeurons[i][j].getnewresponse();
			}
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

        for(int i=0; i<numNeurons; i++){
        	for(int j=0;j<mDepth;j++){
			    hiddenNeurons[i][j].computeBottomUpResponse(currentInput);
			}
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
	   for(int i=0; i<numNeurons; i++){
		   for(int j=0;j<mDepth;j++){
			    hiddenNeurons[i][j].computeTopDownResponse(currentInput);
		   }
		}
	}

/*
    //compute the lateral preResponse
	public void computeLateralResponse(){
		float[] tempResponse = new float[numNeurons*mDepth];
		tempResponse = this.getResponse1D();
//		tempResponse = this.preResponse;

		for(int i=0;i<usedNeurons; i++){
			for(int j=0;j<mDepth;j++){
				hiddenNeurons[i][j].computeLateralResponse(tempResponse,lateralPercent);
			 }
			
		}
	}
*/
	
    //compute the final response
	public void computeResponse(boolean learn_flag, DN2.MODE mode) {		
		for(int i=0;i<numNeurons; i++){
			for(int j=0;j<mDepth;j++){
				hiddenNeurons[i][j].computeResponse();
			}
		}
	
        // do the topKcompetition
/*		float temp = (float)usedNeurons/numNeurons;
		int index = 0;
		if (mode == DN2.MODE.GROUP) {
			index = (int)(temp/0.02f);
		}
		if(mode == DN2.MODE.MAZE){
		    index = (int)(temp/0.05f);
		}
		if(usedNeurons == numNeurons){
			index = index-1;
		}  */
		//dynamic top-k competition for each type neurons
		
//        topKCompetition(learn_flag, mType, mPerfectPercent, mode);
        
        dynamictopKCompetition(learn_flag, mType, mPerfectPercent, mode);
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

	private static void topKSort2(Pair2[] sortArray, int topK) {

		for (int i = 0; i < topK; i++) {
			Pair2 maxPair = sortArray[i];
			int maxIndex = i;
			
			for (int j = i + 1; j < sortArray.length; j++) {
            // select temporary max value
                if (sortArray[j].value > maxPair.value) {
															
					maxPair = sortArray[j];
					maxIndex = j;

				}
			}
            // store the value of pivot (top i) element
			if (maxPair.index1 != i) {
				Pair2 temp = sortArray[i];
				// replace with the maxPair object
				sortArray[i] = maxPair;
                //replace maxPair index elements with the pivot
				sortArray[maxIndex] = temp;
			}
		}
	}
	
	private void topKCompetition(boolean learn_flag, int type, float perfectmatch, DN2.MODE mode) {
		// initializing the indexes
		float[] copyArray = new float[numNeurons*mDepth];
		// Pair is an object that contains the (index,response_value) of each
		// hidden neurons.
		Pair[] sortArray = new Pair[numNeurons*mDepth];
        
//		System.out.println("get neuron "+i+" response: "+hiddenNeurons[i].getnewresponse());
		for(int i=0; i<numNeurons; i++){
			for(int j=0; j<mDepth; j++){	
				sortArray[i*mDepth+j] = new Pair(i*mDepth+j, hiddenNeurons[i][j].getnewresponse());
				copyArray[i*mDepth+j] = hiddenNeurons[i][j].getnewresponse();
				preResponse[i*mDepth+j] = hiddenNeurons[i][j].getnewresponse();			
				hiddenNeurons[i][j].setnewresponse(0.0f);
				hiddenNeurons[i][j].setwinnerflag(false);
				inhibitoryNeurons[i][j].setwinnerflag(true);
			}
	     }
		// check if the top winner has almost perfect match.
//		System.out.println(sortArray[0].value < (ALMOST_PERFECT_MATCH_RESPONSE+1.0f*lateralPercent) && usedHiddenNeurons < numNeurons);
		if (learn_flag) {
			for(int i = 0; i < numNeurons; i++){
				for(int j = mDepth-1; j >=0 ; j--){
					if(hiddenNeurons[i][j].getState()==false){
						if(copyArray[i*mDepth+j] > (ALMOST_PERFECT_MATCH_RESPONSE)*perfectmatch){
							if(j>0){
								if(hiddenNeurons[i][j-1].getState()==true){
									hiddenNeurons[i][j].setState(true);
									hiddenNeurons[i][j].setfiringage(0);
									hiddenNeurons[i][j].setwinnerflag(true);
									inhibitoryNeurons[i][j].setwinnerflag(false);
									hiddenNeurons[i][j].setnewresponse(copyArray[i*mDepth+j]);
								}
							}
							else{
								hiddenNeurons[i][j].setState(true);
								hiddenNeurons[i][j].setfiringage(0);
								hiddenNeurons[i][j].setwinnerflag(true);
								inhibitoryNeurons[i][j].setwinnerflag(false);
								hiddenNeurons[i][j].setnewresponse(copyArray[i*mDepth+j]);
							}
						}

					}
					else{
						hiddenNeurons[i][j].setwinnerflag(true);
					}
				}
			}			
		}

		// Find the topK winners and their indexes.
		for(int i = 0; i < numNeurons; i++){
			for(int j = 0; j < mDepth; j++){
				if(hiddenNeurons[i][j].getState()==true && hiddenNeurons[i][j].getfiringage()>0){
					float curResponse = copyArray[i*mDepth+j];					
					float[] tempResponse = elementWiseProduct(copyArray, inhibitoryNeurons[i][j].getLateralWeights());
					float www = tempResponse[i*mDepth+j];
			        Arrays.sort(tempResponse);
			        if(curResponse >= tempResponse[numNeurons*mDepth-topK]){
			    	    float tempnew= (curResponse - tempResponse[numNeurons*mDepth-topK-1])
								/ (tempResponse[numNeurons*mDepth-1] - tempResponse[numNeurons*mDepth-topK-1] + MACHINE_FLOAT_ZERO);
						hiddenNeurons[i][j].setnewresponse(tempnew);
						hiddenNeurons[i][j].setwinnerflag(true);
			            inhibitoryNeurons[i][j].setwinnerflag(false);
			       }
				}
			}
		}
		if(!learn_flag){
		    System.out.println("********************************************");
		}
	}

	private void Competition(boolean learn_flag, int type, float perfectmatch, DN2.MODE mode, int starth, int endh, int startw, int endw, int startindex) {
		int winner = 0;
		int num = (endh - starth)*(endw - startw);
		Pair2[] sortArray1 = new Pair2[num*mDepth];
		Pair2[] sortArray2 = new Pair2[num*mDepth];
		int i = 0;
		for(int m = starth; m < endh; m++){
			for(int n = startw; n < endw; n++){
				for(int j=0; j<mDepth; j++){					
					preResponse[(m*width+n)*mDepth+j] = hiddenNeurons[m*width+n][j].getnewresponse();	
//					System.out.println("neuron "+((m*width+n)*mDepth+j)+"'s pre-response: "+preResponse[(m*width+n)*mDepth+j]);
					if(hiddenNeurons[m*width+n][j].getState()){
						sortArray1[i*mDepth+j] = new Pair2((m*width+n),j, hiddenNeurons[m*width+n][j].getnewresponse());
						sortArray2[i*mDepth+j] = new Pair2((m*width+n), j, -1);
					}
					else{
						sortArray2[i*mDepth+j] = new Pair2((m*width+n), j, hiddenNeurons[m*width+n][j].getnewresponse());
						sortArray1[i*mDepth+j] = new Pair2((m*width+n), j, -1);
					}
					hiddenNeurons[m*width+n][j].setnewresponse(0.0f);
					hiddenNeurons[m*width+n][j].setwinnerflag(false);
					inhibitoryNeurons[m*width+n][j].setwinnerflag(true);
				}
				i++;
			}
		}
		
		topKSort2(sortArray1, topK);
		topKSort2(sortArray2, 1);
		
		if (learn_flag) {
			if (sortArray1[0].value < (ALMOST_PERFECT_MATCH_RESPONSE)*perfectmatch){
				if(sortArray2[0].value > -1){
						if(hiddenNeurons[sortArray2[0].get_index1()][sortArray2[0].get_index2()].getState() == false){
							int ind = sortArray2[0].get_index1();
							int jnd = sortArray2[0].get_index2();
							hiddenNeurons[ind][jnd].setState(true);
							hiddenNeurons[ind][jnd].setfiringage(0);
							hiddenNeurons[ind][jnd].setwinnerflag(true);
							inhibitoryNeurons[ind][jnd].setwinnerflag(false);
							hiddenNeurons[ind][jnd].setnewresponse(1.0f);
							if(startloc[startindex] == 0){
								float[] temp = {4.5f+(float)Math.pow(-1, ind)*0.1f, 4.5f-(float)Math.pow(-1, ind)*0.1f, 4.5f+(float)Math.pow(-1, ind)*0.1f};
								hiddenNeurons[ind][jnd].setlocation(temp);
								startloc[startindex] += 1;
							}
  						    else{
								float[] temp1 = hiddenNeurons[sortArray1[0].get_index1()][sortArray1[0].get_index2()].getlocation();
								float[] temp2 = new float [3];
								temp2 = setnormvector(temp2);
								float[] temp = {temp1[0]+temp2[0]*10*MACHINE_FLOAT_ZERO, temp1[1]+temp2[1]*10*MACHINE_FLOAT_ZERO, temp1[2]+temp2[2]*10*MACHINE_FLOAT_ZERO};
								hiddenNeurons[ind][jnd].setlocation(temp);
							}
							
							winner++;
						
					}
				}
			}
		}
		
		float value_top1 = sortArray1[0].value;
		float value_topkplus1 = sortArray1[topK].value; 
		
		while(winner < topK){
			float tempresponse = 1.0f;
			if(value_top1 > value_topkplus1) {
				tempresponse = (sortArray1[winner].value - value_topkplus1)
					/ (value_top1 - value_topkplus1 + MACHINE_FLOAT_ZERO);
			}

			int ind = sortArray1[winner].get_index1();
			int jnd = sortArray1[winner].get_index2();
			hiddenNeurons[ind][jnd].setnewresponse(tempresponse);
			hiddenNeurons[ind][jnd].setwinnerflag(true);
			inhibitoryNeurons[ind][jnd].setwinnerflag(false);
			winner++;			
		}
	}
	
	private void dynamictopKCompetition(boolean learn_flag, int type, float perfectmatch, DN2.MODE mode) {

		Competition(learn_flag, type, perfectmatch, mode, 0, height/2, 0 , width/2, 0);
		Competition(learn_flag, type, perfectmatch, mode, 0, height/2, width/2, width, 1);
		Competition(learn_flag, type, perfectmatch, mode, height/2, height, 0 , width/2, 2);
		Competition(learn_flag, type, perfectmatch, mode, height/2, height , width/2, width, 3);
/*
		Competition(learn_flag, type, perfectmatch, mode, 0, height/3, 0 , width/3, 0);
		Competition(learn_flag, type, perfectmatch, mode, 0, height/3, width/3, 2*width/3, 1);
		Competition(learn_flag, type, perfectmatch, mode, 0, height/3, 2*width/3, width, 2);
		Competition(learn_flag, type, perfectmatch, mode, height/3, 2*height/3, 0 , width/3, 3);
		Competition(learn_flag, type, perfectmatch, mode, height/3, 2*height/3, width/3, 2*width/3, 4);
		Competition(learn_flag, type, perfectmatch, mode, height/3, 2*height/3, 2*width/3, width, 5);
		Competition(learn_flag, type, perfectmatch, mode, 2*height/3, height , 0 , width/3, 6);
		Competition(learn_flag, type, perfectmatch, mode, 2*height/3, height , width/3, 2*width/3, 7);
		Competition(learn_flag, type, perfectmatch, mode, 2*height/3, height , 2*width/3, width, 8);
*/		
	}

    //convert the new responses to old responses
	public void replaceHiddenLayerResponse() {
		for(int i=0;i<numNeurons; i++){
			for(int j=0;j<mDepth;j++){
				hiddenNeurons[i][j].replaceResponse();
			}
		}
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
		return numNeurons*mDepth;
	}
	
    //get the local receptive field
	public float[][][] getRfMask() {
		float[][][] bottomUpMask = new float [numNeurons][mDepth][numBottomUpWeights];
		for(int i = 0; i < numNeurons; i++){
			for(int j=0;j<mDepth;j++){
				System.arraycopy(hiddenNeurons[i][j].getBottomUpMask(),0,bottomUpMask[i],0,numBottomUpWeights);
			}
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
			for(int j=0;j<mDepth;j++){
				hiddenNeurons[i][j].resetResponses();
			}
		}

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
	
	public class Pair2 implements Comparable<Pair> {
		public final int index1;
		public final int index2;
		public final float value;

		public Pair2(int index1,int index2, float value) {
			this.index1 = index1;
			this.index2 = index2;
			this.value = value;
		}

		public int compareTo(Pair other) {
			return -1 * Float.valueOf(this.value).compareTo(other.value);
		}

		public int get_index1() {
			return index1;
		}
		
		public int get_index2() {
			return index2;
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
		
		data_out.writeInt(numNeurons);
				
		// bottom up weight
		if (display_y_zone == 1) {
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
				    for (int j = 0; j < numBottomUpWeights; j++) {
				    	data_out.writeFloat((float) hiddenNeurons[i][k].getBottomUpWeights()[j] * hiddenNeurons[i][k].getBottomUpMask()[j]);
					}
				}
			}
		}

		// bottom up age
		else if (display_y_zone == 2) {
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for (int j = 0; j < numBottomUpWeights; j++) {
						data_out.writeInt(hiddenNeurons[i][k].getbottomupage()[j]);
					}
				}
			}
		}

		// bottom up mask
		else if (display_y_zone == 3) {
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for (int j = 0; j < numBottomUpWeights; j++) {
						data_out.writeFloat((float)hiddenNeurons[i][k].getBottomUpMask()[j]);
					}
				}
			}
		}

		// bottom up age
		else if (display_y_zone == 4) {
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for (int j = 0; j < numBottomUpWeights; j++) {
						data_out.writeFloat((float) hiddenNeurons[i][k].getBottomUpVariances()[j]);
					}
				}
			}
		}

		// topDown weight
		else if (display_y_zone == 5) {
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for (int j = 0; j < numTopDownWeights; j++) {
						data_out.writeFloat((float) hiddenNeurons[i][k].getTopDownWeights()[j]);
					}
				}
			}
		}

		// topDown age
		else if (display_y_zone == 6) {
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for (int j = 0; j < numTopDownWeights; j++) {
						data_out.writeInt(hiddenNeurons[i][k].gettopdownage()[j]);
					}
				}
			}
		}

		// topDown mask
		else if (display_y_zone == 7) {
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for (int j = 0; j < numTopDownWeights; j++) {
						data_out.writeFloat((float)hiddenNeurons[i][k].getTopDownMask()[j]);
					}
				}
			}
		}

		// topDown variance
		else if (display_y_zone == 8) {
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for (int j = 0; j < numTopDownWeights; j++) {
						data_out.writeFloat((float) hiddenNeurons[i][k].getTopDownVariances()[j]);
					}
				}
			}
		}
		 //lateral excitation weights
		if (display_y2_zone == 1) {
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for(int j = 0; j < numNeurons*mDepth; j++){
						data_out.writeFloat((float) hiddenNeurons[i][k].getLateralWeights()[j]);
					}
			    }
		    }
		} 
		 //lateral excitation ages	
		if (display_y2_zone == 2) {
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for(int j = 0; j < numNeurons*mDepth; j++){
						data_out.writeInt(hiddenNeurons[i][k].getlateralage()[j]);
				    }
			    }
		    }
		} 
		 //lateral excitation masks
		if (display_y2_zone == 3) {
		    for (int i = start_id; i < end_id; i++) {
		    	for(int k = 0; k < mDepth; k++){
		    		for(int j = 0; j < numNeurons*mDepth; j++){
		    			data_out.writeFloat((float) hiddenNeurons[i][k].getLateralMask()[j]);
				    }
			    }
		    }
		} 
		 //lateral excitation variances
		if (display_y2_zone == 4) {
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for(int j = 0; j < numNeurons*mDepth; j++){
						data_out.writeFloat((float) hiddenNeurons[i][k].getLateralVariances()[j]);
					}
			    }
		    }
		} 
		 //inhibition weights
		if (display_y2_zone == 5) {
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for(int j = 0; j < numNeurons*mDepth; j++){
						data_out.writeFloat((float) inhibitoryNeurons[i][k].getLateralWeights()[j]);
					}
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
			for(int k = 0; k < mDepth; k++){
				data_out.writeFloat((float) hiddenNeurons[i][k].getbottomUpresponse());
			}
		}

		// top down response
		for (int i = start_id; i < end_id; i++) {
			for(int k = 0; k < mDepth; k++){
				data_out.writeFloat((float) hiddenNeurons[i][k].gettopDownresponse());
			}
		}

		// final response
		for (int i = start_id; i < end_id; i++) {
			for(int k = 0; k < mDepth; k++){
				data_out.writeFloat((float) hiddenNeurons[i][k].getnewresponse());
			}
		}
		
		for (int i = 0; i < numNeurons; i++){
			for(int k = 0; k < mDepth; k++){
				for (int j = 0; j < 3; j++){
					data_out.writeFloat((float)hiddenNeurons[i][k].getlocation()[j]);
				}
			}
		}
		
	    for (int i = 0; i < numNeurons; i++){
	    	for(int k = 0; k < mDepth; k++){ 
	    		data_out.writeFloat((float)hiddenNeurons[i][k].getType());
	    	}
	   
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
				for(int k = 0; k < mDepth; k++){
					for (int j = 0; j < numBottomUpWeights; j++) {
						weights[i*mDepth+k][j] = hiddenNeurons[i][k].getBottomUpWeights()[j];
					}
				}
			}
			return weights;	
		}

		// bottom up age
		else if (type == 2) {
			weights = new float[display_num][numBottomUpWeights];
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for (int j = 0; j < numBottomUpWeights; j++) {
						weights[i*mDepth+k][j] = hiddenNeurons[i][k].getbottomupage()[j];
					}
				}
			}
			return weights;	
		}

		// bottom up mask
		else if (type == 3) {
			weights = new float[display_num][numBottomUpWeights];
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for (int j = 0; j < numBottomUpWeights; j++) {
						weights[i*mDepth+k][j] = hiddenNeurons[i][k].getBottomUpMask()[j];
					}
				}
			}
			return weights;	
		}

		// bottom up variance
		else if (type == 4) {
			weights = new float[display_num][numBottomUpWeights];
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for (int j = 0; j < numBottomUpWeights; j++) {
						weights[i*mDepth+k][j] = hiddenNeurons[i][k].getBottomUpVariances()[j];
					}
				}
			}
			return weights;	
		}

		// topDown weight
		else if (type == 5) {
			weights = new float[display_num][numTopDownWeights];
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for (int j = 0; j < numTopDownWeights; j++) {
						weights[i*mDepth+k][j] = hiddenNeurons[i][k].getTopDownWeights()[j];
					}
				}
			}
			return weights;	
		}

		// topDown age
		else if (type == 6) {
			weights = new float[display_num][numTopDownWeights];
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for (int j = 0; j < numTopDownWeights; j++) {
						weights[i*mDepth+k][j] = hiddenNeurons[i][k].gettopdownage()[j];
					}
				}
			}
			return weights;	
		}

		// topDown mask
		else if (type == 7) {
			weights = new float[display_num][numTopDownWeights];
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numTopDownWeights; j++) {
					for(int k = 0; k < mDepth; k++){
						weights[i*mDepth+k][j] = hiddenNeurons[i][k].getTopDownMask()[j];
					}
				}
			}
			return weights;	
		}

		// topDown variance
		else if (type == 8) {
			weights = new float[display_num][numTopDownWeights];
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for (int j = 0; j < numTopDownWeights; j++) {
						weights[i*mDepth+k][j] = hiddenNeurons[i][k].getTopDownVariances()[j];
					}
				}
			}
			return weights;	
		}
        //lateral excitation weights
		if (type == 9) {
			weights = new float[display_num][numNeurons*mDepth];
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for(int j = 0; j < numNeurons*mDepth; j++){
						weights[i*mDepth+k][j] = hiddenNeurons[i][k].getLateralWeights()[j];
					}
			    }
		    }
			return weights;	
		} 
		//lateral excitation ages
		if (type == 10) {
			weights = new float[display_num][numNeurons*mDepth];
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for(int j = 0; j < numNeurons*mDepth; j++){
						weights[i*mDepth+k][j] = hiddenNeurons[i][k].getlateralage()[j];
					}
			    }
		    }
			return weights;	
		} 
		//lateral excitation masks
		if (type == 11) {
			weights = new float[display_num][numNeurons*mDepth];
		    for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for(int j = 0; j < numNeurons*mDepth; j++){
						weights[i][j] = hiddenNeurons[i][k].getLateralMask()[j];
			    	}
			    }
		    }
		    return weights;	
		} 
		//lateral excitation variances
		if (type == 12) {
			weights = new float[display_num][numNeurons*mDepth];
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for(int j = 0; j < numNeurons*mDepth; j++){
						weights[i][j] = hiddenNeurons[i][k].getLateralVariances()[j];
					}
			    }
		    }
			return weights;	
		} 
		//inhibition weights
		if (type == 13) {
			weights = new float[display_num][numNeurons*mDepth];
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for(int j = 0; j < numNeurons*mDepth; j++){
						weights[i][j] = inhibitoryNeurons[i][k].getLateralWeights()[j];
					}
			    }			    
		    }
			return weights;	
		}
		//inhibition areas
		if (type == 14) {
			weights = new float[display_num][numNeurons];
			for (int i = start_id; i < end_id; i++) {
				for(int k = 0; k < mDepth; k++){
					for(int j = 0; j < numNeurons; j++){
						weights[i][j] = inhibitoryNeurons[i][k].getLateralMask()[j];
					}
			    }			    
		    }
			return weights;	
		}
		return null;
	}
	
	public int[][] configure_where_count(int[] rfSize, int[] rfStride,
			int[] inputSize) {
		// in matlab we use the rf_id in col major order.
		// but rf_id_loc is [id][height][width]
		int where_count = 0;
		int[][] rfIdLoc;
//		if(rfSize!=0){
/*		  int half_rf_size1 = (rfSize[0] - 1) / 2;
		  int half_rf_size2 = (rfSize[1] - 1) / 2;
		  for (int height = half_rf_size1; height < inputSize[0] - half_rf_size1; height += rfStride[0]) {
			  for (int width = half_rf_size2; width < inputSize[1] - half_rf_size2; width += rfStride[1]) {			
				where_count++;
			}
	      
		  }*/

		  for (int height = 0; height < inputSize[0] - rfSize[0]+1; height += rfStride[0]) {
			  for (int width = 0; width < inputSize[1] - rfSize[1]+1; width += rfStride[1]) {			
				where_count++;
			}
	      
		  }
		  rfIdLoc = new int[where_count][2];
		  int id = 0;
		  for (int height = 0; height < inputSize[0]- rfSize[0]+1; height += rfStride[0]) {
			  for (int width = 0; width < inputSize[1]- rfSize[1]+1; width += rfStride[1]) {			
				rfIdLoc[id][0] = height;
				rfIdLoc[id][1] = width;
				id++;
			}
		  }
		
/*		else{ 
		  rfIdLoc = new int[1][2];
		  rfIdLoc[0][0] = 0;
		  rfIdLoc[0][1] = 0;
		  
		}*/
		return rfIdLoc;
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

		for(int i = 0;i < numGlial; i++){
			Pair[] distances =new Pair[numNeurons*mDepth];

		    for(int k=0; k<numNeurons; k++){
		    	for(int l=0; l<mDepth; l++){
		    		if(hiddenNeurons[k][l].getState()){
		    			distances[k*mDepth+l] = new Pair(k*mDepth+l, glialcells[i].computeDistance(hiddenNeurons[k][l].getlocation()));
		    		}
		    		else{
		    			distances[k*mDepth+l] = new Pair(k*mDepth+l, 50);
		    		}
				}
		    }
			
		
			topKNeighborSort(distances,glialtopK);
			for(int k = 0;k < glialtopK; k++){
				glialcells[i].setpullindex(k, distances[k].index);
				float[] dis =new float[3];
				for(int m = 0; m < 3; m++){
					int tempdepth = (distances[k].index)%mDepth;
					int tempindex = (distances[k].index)/mDepth;
					dis[m]=glialcells[i].getlocation()[m] - hiddenNeurons[tempindex][tempdepth].getlocation()[m];
				}
				glialcells[i].setpullvector(dis,k);
			}
		}
	    for(int k =0; k<numNeurons; k++){	
	    	for(int g =0; g<mDepth; g++){	
			    	float[] pullvector = new float [3];
			    	for(int l = 0;l < numGlial; l++){
				    	for(int h = 0;h < glialtopK; h++){
					    	if(glialcells[l].getpullindex(h) == k*mDepth+g){
						    	for(int m = 0; m < 3; m++){
							    	pullvector[m] += glialcells[l].getpullvector(h)[m];
								}
						    }
					    }
				    }
			    float[] temp = new float[3];
			    pullvector = norm(pullvector);
			    for(int m = 0; m < 3; m++){
				    temp[m] = hiddenNeurons[k][g].getlocation()[m]+pullrate*pullvector[m];
			    }
			    hiddenNeurons[k][g].setlocation(temp);
			  }
		    }
	}

	
}

