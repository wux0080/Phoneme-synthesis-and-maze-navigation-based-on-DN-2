package GUI;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

import DN2.DN2;

public class PrimaryComputation {
    private DN2 mNetwork;
    
	public PrimaryComputation(DN2 network) {
		mNetwork = network;
	}
	
	public void training(String filename, int length, int num){
		InputDataReader input = new InputDataReader(filename, length);
		int t = 1;
		do {
			System.out.println("primary hidden computation: "+t);
			float[][][] oldInputPattern = new float[num][][];
			oldInputPattern = input.getStreamInput();
			mNetwork.computePrimaryHiddenResponse(oldInputPattern, true);
			t++;
		} while (input.hasInput());
		
		mNetwork.savePriBottomWeights();

	}
	
	public DN2 getDN2() {
		return mNetwork;
	}
	
	public static void saveInputToFile(String hidden_ind, float[][][] input) {
		try {
			PrintWriter wr_input = new PrintWriter(new File(hidden_ind + "input.txt"));
			for (int i = 0; i < input[0].length; i++) {
				for(int k = 0; k <  input[0][i].length; k++){
					wr_input.print(Float.toString(input[0][i][k]) + ' ');
				}
				 wr_input.println();			
			}
			wr_input.close();
//			wr_inhibit.close();
//			wr_mask.close();			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
		
	}
	
	
    public static float[][] getTablevector(String filename) throws IOException{
    	return new TableReader(filename).getTable();
    }
}
