package MazeInterface;
import java.awt.Color;

public interface Commons {
	// DN settings.
	public static final int DNVERSION = 2;
    public static final boolean use_socket_gui_flag = false;
    public static final boolean vision_2D_flag = true;
    public static final boolean where_what_flag = false;

	public static enum ComputingMode{CPU, GPU};
	public ComputingMode computing_mode = ComputingMode.CPU;
	
	// General environment settings.
    public static int WIDTH = 620;
    public static int HEIGTH = 580;
    public static final int BRICK_SIZE = 50;
    public static final int AGENT_SIZE = 20;
    public static final int DELAY = 500;
    public static final int PERIOD = 10;
	public static final int NULLVALUE = -1000;
	public static final float step_value = 0.02f;
    
    // Reward values
    public static int LOW_PAIN = -10;
    public static int HIGH_PAIN = -100;
    
    // Traffic light update interval
    public static int traffic_light_interval = 75;
    
    // COVERT OVERT
    public static int COVERT = 1;
    public static int OVERT = 0;
    
    // Noises
    public static float noise = 0.0f; // percent. Added to vision. 
    
    // Environment
    // URLT: upper right traffic light
    // LLLT: lower left traffic light
    // LRLT: lower right traffic light
    public static enum env {OPEN, WALL, DEST, OBST, RWRD, RAND, LIGHT_PASS, LIGHT_STOP, URLT, LLLT, LRLT};
    public static final env OPEN = env.OPEN;
    public static final env WALL = env.WALL;
    public static final env DEST = env.DEST;
    public static final env OBST = env.OBST;
    public static final env RWRD = env.RWRD;
    public static final env RAND = env.RAND;
    public static final env LIGHT_STOP = env.LIGHT_STOP;
    public static final env LIGHT_PASS = env.LIGHT_PASS;
    public static final env URLT = env.URLT;
    public static final env LLLT = env.LLLT;
    public static final env LRLT = env.LRLT;
    public static env parseEnvString(String str){
    	if(str.equals("WALL")){
    		return WALL;
    	} else if (str.equals("OPEN")){
    		return OPEN;
    	} else if (str.equals("DEST")){
    		return DEST;
    	} else if (str.equals("OBST")){
    		return OBST;
    	} else if (str.equals("RAND")){
    		return RAND;
    	} else if (str.startsWith("RWRD")){
    		return RWRD;
    	} else if (str.equals("URLT")){
    		return URLT;
    	} else if (str.equals("LLLT")){
    		return LLLT;
    	} else if (str.equals("LRLT")){
    		return LRLT;
    	} else {
    		return null;
    	}
    }
    
    // GPS
    public static enum gps {UP, LEFT, RIGHT, DOWN, ARRIVE, NONE};
    public static int[] gps_angles = {-90, 180, 0, 90, 0, NULLVALUE}; // Corresponding to GPS above.
    public static final gps UP     = gps.UP;
    public static final gps LEFT   = gps.LEFT;
    public static final gps RIGHT  = gps.RIGHT;
    public static final gps DOWN   = gps.DOWN;
    public static final gps ARRIVE = gps.ARRIVE;
    public static final gps NONE   = gps.NONE;
    public static gps parseGpsString(String str){
    	if(str.equals("UP")){
    		return UP;
    	} else if (str.equals("LEFT")){
    		return LEFT;
    	} else if (str.equals("RIGHT")){
    		return RIGHT;
    	} else if (str.equals("DOWN")){
    		return DOWN;
    	} else if (str.equals("ARRIVE")){
    		return ARRIVE;
    	} else if (str.equals("NONE")){
    		return NONE;
    	} else {
    		return null;
    	}
    }
    
    // VISION
	public static final Color[] VISIONCOLORS = {Color.lightGray, new Color(178,84,12), new Color(131,185,37), new Color(123,170,255), Color.yellow, 
			                                    Color.BLACK, Color.green, new Color(255, 0, 255)};
	public static final String VISIONTITLE = "Vision";
	public static final int VISIONTITLEX = WIDTH - 140;
	public static final int VISIONTITLEY = 25;
	public static final int VISIONLINESPACE = 20;
	
	// This class facilitates testing return result.
	public class ActionConceptPair{
		public Agent.Action action;
		public int skill;
		public int destination;
		
		public ActionConceptPair(Agent.Action input_action, int input_skill, int input_dest){
			action = input_action;
			skill = input_skill;
			destination = input_dest;
		}
	}
	
	// Learning modes.
	public static enum mode {SUPERVISED, TEST, PLAN};
	public static final mode SUPERVISED = mode.SUPERVISED;
	public static final mode TEST       = mode.TEST;
	public static final mode PLAN    = mode.PLAN;
	public static mode parseModeString(String str){
		if (str.equals("SUPERVISED")){
			return SUPERVISED;
		} else if (str.equals(TEST)){
			return TEST;
		} else if (str.equals(PLAN)){
			return PLAN;
		} else {
			return null;
		}
	}
	
	// Lessons types.
	public static enum lessonType {SKILL, DESTINATION, PLANNING};
	public static final lessonType SKILL = lessonType.SKILL;
	public static final lessonType DESTINATION = lessonType.DESTINATION;
	public static final lessonType PLANNING = lessonType.PLANNING;
	
	// Planning return type.
	public class PlanResult {
		public float[] values;
		public ActionConceptPair result;
		public PlanResult(int values_num, int[] input_values, ActionConceptPair input_result){
			values = new float[values_num];
			for (int i = 0; i < values_num; i++){
				values[i] = input_values[i];
			}
			result = input_result;
		}
	}
	
	public class Point{
		public int x;
		public int y;
		public boolean mode; // 0 for old path, 1 for new path
		public Point(int input_x, int input_y, boolean input_mode){
			x = input_x;
			y = input_y;
			mode = input_mode;
		}
	}
}