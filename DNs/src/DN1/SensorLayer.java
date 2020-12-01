package DN1;
import java.io.Serializable;


public class SensorLayer implements Serializable{
	
	private int height;
	private int width;
	
	private float[][] input;
	
	public SensorLayer(int height, int width){
		this.setWidth(width);
		this.setHeight(height);
		
		input = new float[height][width];
	}
	
	public SensorLayer(int height, int width, float[][] input){
		this.setWidth(width);
		this.setHeight(height);
		
		this.setInput(input);
	}

	public int getHeight() {
		return height;
	}

	public void setHeight(int height) {
		this.height = height;
	}

	public int getWidth() {
		return width;
	}

	public void setWidth(int width) {
		this.width = width;
	}

	// convert into 1d Array
	public float[] getInput1D() {
		float[] inputArray = new float[height * width];
		
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int index = i*width + j;
				inputArray[index] = input[i][j];
			}
			
		}
		
		return inputArray;
	}
	
	public float[][] getInput() {
		return input;
	}

	public void setInput(float[][] input) {
		this.input = input;
	}
	
	

}

