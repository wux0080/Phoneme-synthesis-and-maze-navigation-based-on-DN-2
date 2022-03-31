/*
 * Agent is the living robot inside the maze environment.
 * During supervised learning its internal AI system can generate supervised correct movements according
 * to the current environment. It then updates the internal DN with the vision input, the gps input and the
 * supervised gps response.
 * During testing it updates its internal DN with the current vision input and gps input to get the emergent
 * action.
 * Agent interacts with the DN by the DNCaller instance. DNCaller is a singleton class.
 */

package MazeInterface;
import java.awt.Graphics2D;
import java.awt.event.KeyEvent;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;

public class Agent extends Sprite implements Commons{
	// xdir and ydir is the current moving direction of the agent. 
	// xdir = cos(angle); y_dir = sin(angle);
	private float xdir;
    private float ydir;
    
    // init_x and init_y are the initial location of the agent.
    private int init_x;
    private int init_y;
    
    public int angle; // current heading
    private int gps_diff; // gps_diff = abs(angle - currentGPS)
    
    public static final int vision_num = 43;
	public static final int[] visino_angle = { -105, -100, -95, -90, -85, -80, 
			                           -75, -70, -65, -60, -55, -50, -45, -40,
			                           -35, -30, -25, -20, -15, -10, -5, 0, 5, 
			                            10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 
			                            60, 65, 70, 75, 80, 85, 90, 95, 100, 105 }; 
    public static int FORWARD = 21; // 0 degrees
    public static int SSRFORWARD = 22; // 5 degrees
    public static int SRFORWARD = 23; // 10 degrees
    public static int RFORWARD = 28; // 35 degrees
    public static int FRIGHT = 35; // 70 degrees
    public static int BRIGHT = 42; // 105 degrees
    public static int LFORWARD = 14; // -35 degrees
    public static int SSLFORWARD = 20; // 5 degrees
    public static int SLFORWARD = 19; // 10 degrees
    public static int FLEFT = 7; // -70 degrees
    public static int BLEFT = 0; // -105 degrees
    
    // Upper limit of visible object distances.
    public static final int vision_range = 70;
    public static final float vision_angle_range = 210.0f; 
    
    public int max_landmark_size = 0;
    public int min_landmark_size = 100;
    
    // This is for plotting the vision information on the vision panel. This guarantees that the vision
    // pannel would display the information in the specified order.
    public int[] vision_order = {BLEFT, FLEFT, LFORWARD, FORWARD, RFORWARD, FRIGHT, BRIGHT};
    
    // AI would make turns when seeing obstacle less than this distance.
    public final float safe_distance = 30;
    
    // Speed of movement.
    private float speed;
    
    // Actions of the agent.
    public static final int action_num = 4;
    public static enum Action {LEFT, RIGHT, FORWARD, STOP};
    private int AFORWARD = Action.FORWARD.ordinal();
    private int ARIGHT = Action.RIGHT.ordinal();
    private int ALEFT = Action.LEFT.ordinal();
    private int ASTOP = Action.STOP.ordinal();
    
    // last_action is used by the AI system to make decision about the next movement.
    // i.e. if last_action is making turns, the agent would keep turning until it has turned 90 degrees.
    private Action last_action; 
    
    // This is actually the black arrow gps on the screen.
    public GPS heading;

    // Construction method of agent. numLessons is needed for DN to set its concept motor.
    public Agent(int num_skills, int num_destinations) {
    	speed = 0; // not moving at first frame.
        xdir = 1.0f; // facing right as default. Each map has its own facing directions.
        ydir = 0f;
        angle = 0;
        gps_diff = 0;
        heading = new GPS(x, y);
        
        // Load the icon for this agent.
        ImageIcon ii = new ImageIcon(this.getClass().getResource("Images/Agent.png"));
        image = ii.getImage();
        
        // Width and height for this agent. Used during collision detection.
        i_width = image.getWidth(null);
        i_heigth = image.getHeight(null);
        
        // Create the internal DN upon initialization.
        File file = new File("maps/network2.ser");
    	DNCaller.getInstance().createDN(num_skills, num_destinations);
        if(file.exists()) {
        	try {
				DNCaller.getInstance().loadDN("maps/network2");
				System.out.println("load network");
			} catch (ClassNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
        }
		
        resetState();
    }
    
    // Each map has a different starting location and initial facing direction.
    // Thus during map reseting, we need to set the initial conditions of the agent.
    public void setInitAgentLoc(int[] loc, int gps_angle){
    	// The initial location from the map is the center location of the starting block.
    	// However init_x and init_y are the upper right corner of the agent.
    	// Thus we need to do a little shifting.
    	init_x = (int) (loc[0] - 0.5 * getWidth());
    	init_y = (int) (loc[1] - 0.5 * getHeight());
    	x = init_x;
    	y = init_y;
    	angle = gps_angle;
    	// Set the moving direction as sin and cos values of the angle.
    	xdir = (float) Math.cos(Math.toRadians(angle));
    	ydir = (float) Math.sin(Math.toRadians(angle));
    	// Set the initialized flag in DN to be false as we need to reset the responses.
    	DNCaller.getInstance().setInitialized(false);
    }
    
    // Update the gps_diff information according to the input gps.
    // Need to make sure that gps_diff is in the range of -180 to 180.
    public void updateGPS(int gps){
    	if (gps != NULLVALUE){
    	    gps_diff = (gps - angle);
    	    if (gps_diff > 180){
    	    	gps_diff = gps_diff - 360;
    	    }
    	    if (gps_diff < -180){
    		    gps_diff = gps_diff + 360;
    	    }
    	}
    }
    
    // obvious, right?
    public int getGpsDiff(){
    	return gps_diff;
    }
    
    public PlanResult plan(VisionLine[] visions, BufferedImage vision_image, gps curr_gps, boolean block_change_flag) {
    	PlanResult plan;
    	plan = DNCaller.getInstance().planOneStep(visions, vision_image, gps_diff, block_change_flag);
    	performAction(plan.result);
    	return plan;
    }
    
    public void learnMoreLess(){
    	//3;
    	DNCaller.getInstance().learnMoreLess3();
    }
    
    // Get the emergent action from DN.
    // Returns the concept number to be displayed on the board.
    public int getEmergentAction(VisionLine[] visions, BufferedImage vision_image, gps curr_gps, int curr_lesson, int total_skills, int previous_skill,
    		                     float current_value, boolean block_change_flag){
    	ActionConceptPair result;
    	if (curr_lesson < total_skills){
    		result = DNCaller.getInstance().test(visions, vision_image, gps_diff, curr_lesson, NULLVALUE, previous_skill, block_change_flag);
    	} else {
    		if (curr_gps != NONE){
    		    result = DNCaller.getInstance().test(visions, vision_image, gps_diff, NULLVALUE, curr_lesson - total_skills, previous_skill, block_change_flag);
    		} else {
    			result = DNCaller.getInstance().test(visions, vision_image, NULLVALUE, NULLVALUE, curr_lesson - total_skills, previous_skill, block_change_flag);
    		}
    	}
    	performAction(result);
    	return result.skill;
    }
    
    // Perform action according to motor response
    private void performAction(ActionConceptPair motor){
    	Action emergent_action = motor.action;
    	if (emergent_action == Action.FORWARD){
    		actionForward();
    	} else if (emergent_action == Action.LEFT){
    		actionLeft();
    	} else if (emergent_action == Action.RIGHT){
    		actionRight();
    	} else if (emergent_action == Action.STOP){
    		actionStop();
    	}
    }
    
    // During supervised training, call the underlying AI system to get the supervised action.
    // Also trains the network using the generated supervision.
    // AI logic: when arriving, stop.
    //           not  arriving, then 
    //                if heading equals to gps pointed direction
    //                   1) if no obstacle, move forward.
    //                   2) if obstacle, then make turns based on closest wall distance.
    //                else can only be two possible situations
    //                   1) turning away from obstacle, then need to keep turning until turned 90 degrees.
    //                   2) correcting facing direction, then need to keep turning until diff is 0 degrees.
    public int getSupervision(VisionLine[] visions, BufferedImage vision_image, gps curr_gps, int curr_lesson, int total_skills, int previous_skill,
    		                  boolean train_flag, float current_value, boolean block_change_flag){
    	// first find if there is land mark inside the current vision range.
    	env landmark_type = null;
    	int rf_size = 19;
    	int half_rf_size = 9;
    	int landmark_size = NULLVALUE;
    	int landmark_loc = NULLVALUE;
    	int landmark_motor_size = NULLVALUE;
    	int landmark_motor_loc = NULLVALUE;
    	for (int i = 0; i < visions.length; i++){
    		if (visions[i].type == LIGHT_PASS){
    			landmark_size = 0;
    			for (int j = i; j < visions.length; j++){
    				if (visions[j].type == visions[i].type){
    					landmark_size ++;
    				} else {
    					landmark_loc = i;
    		    		landmark_type = LIGHT_PASS;
    					break;
    				}
    			}
    			if (i + landmark_size == visions.length){
    				landmark_loc = i;
    				landmark_type = LIGHT_PASS;
    			}
    			break;
    		}
    	}
    	for (int i = 0; i < visions.length; i++){
    		if (visions[i].type == OBST){
    			landmark_size = 0;
    			for (int j = i; j < visions.length; j++){
    				if (visions[j].type == visions[i].type){
    					landmark_size ++;
    				} else {
    					landmark_loc = i;
    		    		landmark_type = OBST;
    					break;
    				}
    			}
    			if (i + landmark_size == visions.length){
    				landmark_loc = i;
    				landmark_type = OBST;
    			}
    			break;
    		}
    	}
    	for (int i = 0; i < visions.length; i++){
    		if (visions[i].type == LIGHT_STOP){
    			landmark_size = 0;
    			for (int j = i; j < visions.length; j++){
    				if (visions[j].type == visions[i].type){
    					landmark_size ++;
    				} else {
    					landmark_loc = i;
    		    		landmark_type = LIGHT_STOP;
    					break;
    				}
    			}
    			if (i + landmark_size == visions.length){
    				landmark_loc = i;
    				landmark_type = LIGHT_STOP;
    			}
    			break;
    		}
    	}
		if (landmark_type != null) {
			int landmark_center_left = landmark_loc + (landmark_size - 1) / 2;
			int landmark_center_right = landmark_loc + (landmark_size) / 2;
			int left_end = landmark_center_left - half_rf_size;
			int right_end = landmark_center_right + half_rf_size;
			if (left_end < 0) {
				left_end = 0;
				right_end = left_end + rf_size - 1;
			}
			if (right_end > vision_num - 1) {
				right_end = vision_num - 1;
				left_end = right_end - rf_size + 1;
			}
			landmark_loc = left_end;
			//landmark_size = right_end - left_end + 1;
		}

    	//System.out.println("loc: " + landmark_loc + ", size: " + landmark_size + ", type: " + landmark_type);
    	
    	if (landmark_size != NULLVALUE){
    		if (landmark_size < min_landmark_size) {
    			min_landmark_size = landmark_size;
    		}
    		if (landmark_size > max_landmark_size) {
    			max_landmark_size = landmark_size;
    		}
    	}
    	
    	if(landmark_size >=0){
    		if(landmark_size <=5){
        		landmark_motor_size = 1;
        	}
        	else if(landmark_size >5 && landmark_size <=10){
        		landmark_motor_size = 2;
        	}
        	else if(landmark_size >10 && landmark_size <=15){
        		landmark_motor_size = 3;
        	}
        	else if(landmark_size >15 && landmark_size <=20){
        		landmark_motor_size = 4;
        	}
        	else{
        		landmark_motor_size = 5;
        	}
    	}	
    	
    	if(landmark_loc >= 0){
    		if(landmark_loc < 8){
        		landmark_motor_loc = 1;
        	}
        	else if(landmark_loc > 34){
        		landmark_motor_loc = 5;
        	}
        	else if(landmark_loc >= 8 && landmark_loc < 16){
        		landmark_motor_loc = 2;
        	}
        	else if(landmark_loc > 26 && landmark_loc <= 34){
        		landmark_motor_loc = 4;
        	}
        	else{
        		landmark_motor_loc = 3;
        	}
    	}
  	
    	// then the supervision AI system figures out the correct movement.
    	if (visions[LFORWARD].type == LIGHT_STOP || visions[RFORWARD].type == LIGHT_STOP){
    		actionStop();
    	}
    	else if (curr_gps != gps.ARRIVE){
    		if (last_action == Action.RIGHT || last_action == Action.LEFT){
    			if ((Math.abs(gps_diff) != 90) && (Math.abs(gps_diff) != 0)){
    				if (last_action == Action.RIGHT){
    					actionRight();
    				}
    				if (last_action == Action.LEFT){
    					actionLeft();
    				}
    			} else {
    				actionForward();
    			}
    		} else {
    			 if ((visions[FORWARD].type == env.OBST && 
      				   visions[FORWARD].getLength() < safe_distance)){
      			if(visions[FLEFT].getLength() < visions[FRIGHT].getLength()){
      				actionRight();
      			} else {
      				actionLeft();
      			}
      			}
    		    else if (gps_diff > 0){
    				if ((visions[FRIGHT].type == env.OBST || 
    					 visions[BRIGHT].type == env.OBST  ||
    					 visions[FRIGHT].type == env.WALL  || 
    					 visions[BRIGHT].type == env.WALL) &&
    					 (gps_diff < 180)){
    					actionForward();
    				} else {
    					actionRight();
    				}
        		} else if (gps_diff < 0){
        			if ((visions[FLEFT].type == env.OBST || 
        				 visions[BLEFT].type == env.OBST ||
        				 visions[FLEFT].type == env.WALL ||
        				 visions[BLEFT].type == env.WALL) &&
        				 (gps_diff > -180)){
        				actionForward();
        			} 
        			//else if((getX() < 100) && (getY() < 100) && (gps_diff >= -5)){
        			//	actionForward();
        			//}
        		else {
        				actionLeft();
        			} 
        		} else {
        			actionForward();
        		}
    		}
    	} else {
    		actionStop();
    	}
    	if (curr_lesson < total_skills){
    		if (train_flag == true){
    			// When training individual skills we don't train cost or destination.
    	        return DNCaller.getInstance().train(visions, vision_image, gps_diff, last_action, 
    	    		                            curr_lesson, NULLVALUE, previous_skill, NULLVALUE, block_change_flag, 
    	    		                            landmark_motor_loc, landmark_type, landmark_motor_size);
//    	    		                            NULLVALUE, null, NULLVALUE);
    		} else {
    			// Going back.
    			DNCaller.getInstance().computeIdle(visions, vision_image, gps_diff, NULLVALUE, curr_lesson, NULLVALUE, block_change_flag);
    			return 0;
    		}
    	} else {
    		if (train_flag == true){
    			// When training destination/means we are training cost.
    		    return DNCaller.getInstance().train(visions, vision_image, gps_diff, last_action, 
    				                            NULLVALUE, curr_lesson - total_skills, previous_skill, current_value, block_change_flag, NULLVALUE, null, NULLVALUE);
    		} else {
    			// Going back.
    			DNCaller.getInstance().computeIdle(visions, vision_image, gps_diff, current_value,NULLVALUE, curr_lesson - total_skills, block_change_flag);
    			return 0;
    		}
    	}
    }
    
    // After learning the destination, the agent needs to learn how to plan for this destination.
    public void learnPlanning(int[] skill_sequence, int curr_destination, int reward_value, float state_value) {
    	DNCaller.getInstance().learnPlanning(skill_sequence, curr_destination, reward_value, state_value);
    }
    
    public void learnMeansLoop(int means_num){
    	DNCaller.getInstance().learnMeansLoop(means_num);
    }
    
    // Update the location of the agent according to its facing direction and speed.
    public boolean move() {
    	x += xdir * speed;
    	y += ydir * speed;
    	if (speed == 0)
    		return false;
    	else
    		return true;
    }
    
    // Update the speed and action when the command is forward.
    public void actionForward(){
    	speed = 1;
    	last_action = Action.FORWARD;
    }

    // Update the speed and action when the command is stop.
    public void actionStop(){
    	speed = 0;
    	last_action = Action.STOP;
    }

    // Update the speed and action when the command is left.
    public void actionLeft(){
    	angle -= 15;
    	if (angle == -180){
    		angle = 180;
    	}
    	speed = 0;
    	xdir = (float) Math.cos(Math.toRadians(angle));
    	ydir = (float) Math.sin(Math.toRadians(angle));
    	last_action = Action.LEFT;
    }

    // Update the speed and action when the command is right.
    public void actionRight(){
    	angle += 15;
    	if (angle == 195){   
    		angle = -165;  
    	}
    	speed = 0f;
    	xdir = (float) Math.cos(Math.toRadians(angle));
    	ydir = (float) Math.sin(Math.toRadians(angle));
    	last_action = Action.RIGHT;
    }
    
    // Key events to control the agent. This function was useful during
    // first stage debugging but deprecated after we have the AI system and DN implemented.
    public void keyPressed(KeyEvent e) {
        int key = e.getKeyCode();

        if (key == KeyEvent.VK_UP) {
        	actionForward();
        }

        if (key == KeyEvent.VK_DOWN) {
            actionStop();
        }
        
        if (key == KeyEvent.VK_LEFT){
        	actionLeft();
        }
        
        if (key == KeyEvent.VK_RIGHT){
        	actionRight();
        }
    }

    // Reset the initial conditions of the agent.
    // This function is called during iterations of learning the same map.
    private void resetState() {
    	last_action = Action.STOP;
        x = init_x;
        y = init_y;
    }
    
    // Setter and getters for the xdir and ydir.
    public void setXDir(int x) {
        xdir = x;
    }

    public void setYDir(int y) {
        ydir = y;
    }

    public float getYDir() {
        return ydir;
    }
    
    // Get the motor pattern from the agent. This function is useful during debugging.
    public float[][][] getMotorPattern() {
    	float[][][] motorPattern = DNCaller.getInstance().getMotorPattern();
    	return motorPattern;
    }
    
    // Get the number of used neurons for each hidden layer.
    public int[] getUsedNeurons(){
    	return DNCaller.getInstance().getUsedNeurons();
    }
    
    // Learn the reward in Z area.
    public void learnReward(int value){
    	DNCaller.getInstance().learnReward(value);
    }
    
    public float getAngle(){
    	return angle;
    }

	public void learnCovertToCovert() {
		DNCaller.getInstance().learnCovertToCovert();
	}
	
	public void saveNet(String name){
		DNCaller.getInstance().saveDN(name);
	}
	
	public void loadNet(String name) throws ClassNotFoundException, IOException{
		DNCaller.getInstance().loadDN(name);
	}
	
	public void sendInfoToGui(){
		DNCaller.getInstance().sendInfoToGui();
	}
	
	public int getDisplayNum(){
		return DNCaller.getInstance().getDisplayNum();
	}
	
	public int getTopDownNum(){
		return DNCaller.getInstance().getTopDownNum();
	}

	public void trainWhereWhat(VisionLine[] visions, BufferedImage vision_image, env current_type, int current_loc,
			int current_scale) {
		DNCaller.getInstance().trainWhereWhat(visions, vision_image, current_type, current_loc, current_scale);		
	}

	public int[] testWhereWhat(VisionLine[] visions, BufferedImage vision_image, env current_type, int current_loc,
			int current_scale) {
		return DNCaller.getInstance().testWhereWhat(visions, vision_image, current_type, current_loc, current_scale);
	}
	
    public void rotateImage() throws Exception{

    	BufferedImage bi;
    	bi = ImageIO.read(new File("src/MazeInterface/Images/Agent.png"));
    	BufferedImage ci = rotateImageByDegrees(bi, angle-90);
    	ImageIcon tem = new ImageIcon(ci);
   	    image = tem.getImage();
    }
    
    public BufferedImage rotateImageByDegrees(BufferedImage img, float angle) {

        double rads = Math.toRadians(angle);
        double sin = Math.abs(Math.sin(rads)), cos = Math.abs(Math.cos(rads));
        int w = img.getWidth();
        int h = img.getHeight();
        int newWidth = (int) Math.floor(w * cos + h * sin);
        int newHeight = (int) Math.floor(h * cos + w * sin);

        BufferedImage rotated = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = rotated.createGraphics();
        AffineTransform at = new AffineTransform();
  //      at.translate((newWidth - w) / 2, (newHeight - h) / 2);

        int x = w / 2;
        int y = h / 2;

        at.rotate(rads, x, y);
        g2d.setTransform(at);
        g2d.drawImage(img, 0, 0, null);
        g2d.dispose();

        return rotated;
    }
    
	public static boolean getKey() throws IOException{
		boolean a = false;
		System.out.println("Please continue....");
		int b = System.in.read();
		if(b == 10){
			a = true;
		}
		return a;
	}
}
