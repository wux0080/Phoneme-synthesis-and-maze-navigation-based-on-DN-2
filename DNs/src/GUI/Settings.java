package GUI;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class Settings {

	private int numSegments;

	private int numPractices;

	private int numPracticesPerTest;

	private int testSegmentStart;

	private int[] lenSequence;

	private int numInput;

	private int numMotor;

	private int numHidden;

	private int rfsize;

	private int rfStride;

	private int[][] inputSize;

	// TODO: motor size need to get where and scale information.
	private int[][] motorSize;

	private int[] hiddenSize;

	private int[] topKMotor;

	private int[] topKHidden;

	private char modality;

	private Boolean useVisualFlag;

	private float prescreeningPercent;
	
	private int[] neuronType;
	
	private String growthTable;
	
	private String meanTable;
	
	private String[] traininginputfile;
	private String[] trainingmotorfile;
	private String[] trainingperformancefile;
	
	private String[] testinputfile;
	private String[] testmotorfile;
	private String[] testperformancefile;

	public void setPrescreeningPercent(float prescreeningPercent) {

		this.prescreeningPercent = prescreeningPercent;

	}

	public float getPrescreeningPercent() {

		return prescreeningPercent;

	}

	public Settings(String filename) {
		rfsize = 0;
		rfStride = 0;
		numPractices = 1;
		numPracticesPerTest = 1;
		prescreeningPercent = 0.5f;
		neuronType = new int[7];
		// Set all settings from a text file
		Scanner reader;
		try {
			reader = new Scanner(new File(filename));

			// read each line from 1 to n
			// count the index and set the object.
			int parameterCount = 0;
			int loopIndex = 0;

			while (reader.hasNext()) {

				String temp = reader.next();

				if (temp.charAt(0) != '\\') {

					switch (parameterCount) {
					case 0:
						numSegments = Integer.parseInt(temp);
						lenSequence = new int[numSegments];
						parameterCount++;
						break;

					case 1:
						testSegmentStart = Integer.parseInt(temp);
						parameterCount++;
						traininginputfile = new String[testSegmentStart-1];
						trainingmotorfile = new String[testSegmentStart-1];
						trainingperformancefile = new String[testSegmentStart-1];
						testinputfile = new String[numSegments-testSegmentStart+1];
						testmotorfile = new String[numSegments-testSegmentStart+1];
						testperformancefile = new String[numSegments-testSegmentStart+1];
						break;

					case 2:
						lenSequence[loopIndex] = Integer.parseInt(temp);
						loopIndex++;

						if (loopIndex >= numSegments) {
							parameterCount++;
							loopIndex = 0;
						}
						break;

					case 3:
						numInput = Integer.parseInt(temp);
						inputSize = new int[numInput][2];
						parameterCount++;
						break;

					case 4:
						inputSize[loopIndex][0] = Integer.parseInt(temp);
						inputSize[loopIndex][1] = Integer.parseInt(reader.next());
						// System.out.println(inputSize[loopIndex][1]);
						loopIndex++;

						if (loopIndex >= numInput) {
							parameterCount++;
							loopIndex = 0;
						}
						break;

					case 5:
						numMotor = Integer.parseInt(temp);
						motorSize = new int[numMotor][2];
						topKMotor = new int[numMotor];
						parameterCount++;
						break;

					case 6:
						motorSize[loopIndex][0] = Integer.parseInt(temp);
						motorSize[loopIndex][1] = Integer.parseInt(reader.next());
						// System.out.println(motorSize[loopIndex][1]);
						loopIndex++;

						if (loopIndex >= numMotor) {
							parameterCount++;
							loopIndex = 0;
						}
						break;

					case 7:
						topKMotor[0] = Integer.parseInt(temp);
						for (int i = 1; i < numMotor; i++) {
							topKMotor[i] = Integer.parseInt(reader.next());
						}
						parameterCount++;
						break;

					case 8:
						numHidden = Integer.parseInt(temp);
						hiddenSize = new int[numHidden];
						topKHidden = new int[numHidden];
						parameterCount++;
						break;

					case 9:
						hiddenSize[loopIndex] = Integer.parseInt(temp);
						loopIndex++;

						if (loopIndex >= numHidden) {
							parameterCount++;
							loopIndex = 0;
						}
						break;

					case 10:
						topKHidden[0] = Integer.parseInt(temp);
						for (int i = 1; i < numHidden; i++) {
							topKHidden[i] = Integer.parseInt(reader.next());
							// System.out.println(topKMotor[i]);
						}
						parameterCount++;
						break;

					case 11:
						modality = temp.charAt(0);
						parameterCount++;
						break;

					case 12:
						setPrescreeningPercent(Float.parseFloat(temp));
						parameterCount++;
						break;

					case 13:
						setUseVisualFlag(Boolean.parseBoolean(temp));
						parameterCount++;
						break;
						
					case 14:
						numPractices = Integer.parseInt(temp);
						parameterCount++;
						break;

					case 15:
						numPracticesPerTest = Integer.parseInt(temp);
						parameterCount++;
						break;
						
					case 16:
						rfsize = Integer.parseInt(temp);
						rfStride = Integer.parseInt(reader.next());
						parameterCount++;
						break;
						
					case 17:
						neuronType[loopIndex] = Integer.parseInt(temp);
						loopIndex++;

						if (loopIndex >= 7) {
							parameterCount++;
							loopIndex = 0;
						}
						break;
						
					case 18:
						growthTable = temp;
						parameterCount++;
						break;
						
					case 19:
						meanTable = temp;
						parameterCount++;
						break;
						
					case 20:
//						System.out.println(temp);
						traininginputfile[loopIndex] = temp;
						reader.nextLine();
						String temp2 = reader.next();
						trainingmotorfile[loopIndex] = temp2;
						reader.nextLine();
						String temp3 = reader.next();
						trainingperformancefile[loopIndex] = temp3;
						loopIndex++;

						if (loopIndex >= testSegmentStart-1) {
							parameterCount++;
							loopIndex = 0;
						}
						break;
						
					case 21:
						testinputfile[loopIndex] = temp;
						reader.nextLine();
						String temp4 = reader.next();
						testmotorfile[loopIndex] = temp4;
						reader.nextLine();
						String temp5 = reader.next();
						testperformancefile[loopIndex] = temp5;
						loopIndex++;

						if (loopIndex >= numSegments-(testSegmentStart-1)) {
							parameterCount++;
							loopIndex = 0;
						}
						break;	
						
					default:
						break;
					}
				}

				else {

					reader.nextLine();

				}
			}

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public Settings() {

	}

	public int getRfSize() {
		return rfsize;
	}

	public int getRfStride() {

		return rfStride;		
	}

	public int getNumSegments() {
		return numSegments;
	}

	public int getTestSegmentStart() {
		return testSegmentStart;
	}

	public int[] getLenSequence() {
		return lenSequence;
	}
	
	public int[] getneurontypes() {
		return neuronType;
	}
	
	public int getNumInput() {
		return numInput;
	}

	public int getNumMotor() {
		return numMotor;
	}

	public int getNumHidden() {
		return numHidden;
	}

	public int[][] getInputSize() {
		return inputSize;
	}

	public int[][] getMotorSize() {
		return motorSize;
	}

	public int[] getHiddenSize() {
		return hiddenSize;
	}

	public int[] getTopKMotor() {
		return topKMotor;
	}

	public int[] getTopKHidden() {
		return topKHidden;
	}

	public int getNumPractices() {
		return numPractices;
	}

	public char getModality() {
		System.out.println("run into it");
		return modality;
	}

	public int getNumPracticesPerTest() {
		return numPracticesPerTest;
	}
	
	public String getgrowthTable(){
		return growthTable;
	}
	
	public String getmeanTable(){
		return meanTable;
	}
	
	public String[] gettrianinginputfile() {
		return traininginputfile;
	}
	
	public String[] gettrianingmotorfile() {
		return trainingmotorfile;
	}
	public String[] gettrianingperformancefile() {
		return trainingperformancefile;
	}
	
	public String[] gettestinputfile() {
		return testinputfile;
	}
	
	public String[] gettestmotorfile() {
		return testmotorfile;
	}
	public String[] gettestperformancefile() {
		return testperformancefile;
	}
	
	public Boolean getUseVisualFlag() {
		return useVisualFlag;
	}
	
	public void setUseVisualFlag(Boolean useGuiFlag) {
		this.useVisualFlag = useGuiFlag;
	}
}