/*
 * Board is the main part of the environment. 
 * 1) store all environment related objects like walls, rewards, and the agent.
 * 2) paint all objects and related information in the window.
 * 3) collision checking, success checking.
 * 4) keep track of the learning schedule.
 * 5) keep track of the maps. 
 */

package MazeInterface;

import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.RenderingHints;
import java.awt.Toolkit;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.geom.AffineTransform;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;
import java.util.Timer;
import java.util.TimerTask;
import javax.swing.JPanel;

import GUI.I_GUI;
import GUI.Neuron_Visualization;
import MazeInterface.Agent.Action;

public class Board extends JPanel implements Commons {

	private Timer timer;
	private String message = "Game Over";
	private Agent agent;
	private Schedule schedule;
	private Map map;
	private Brick bricks[];
	private Obstacle obstacles[];
	private Destination destinations[];
	private Reward rewards[];
	private TrafficLight lights[];
	private VisionLine visions[];
	private BufferedImage vision_image;
	private int currentGridLoc[];
	private boolean ingame;
	private boolean get_back_flag;
	private float current_distance;
	private float[] predicted_value;
	private ArrayList<Point> path;

	// These parameters deal with individual lessons
	private int learn_count;
	private mode supervised_flag;
	private int supervision_count;
	private boolean planning_flag;
	private boolean train_flag;
	private boolean block_change_flag;
	private boolean[] means_learned_flag;
	private int light_interval_count;

	// These parameters deal with the entire teaching
	private int lesson_iteration;
	private int current_lesson;
	private int emergent_concept;
	private int total_steps;
	private int emergent_destination;
	private int old_destination;
	
	// These parameters deal with where what teaching
	private boolean train_where_what_flag = where_what_flag;
	private boolean test_where_what_flag  = false;
	private int[] background_type = {env.OPEN.ordinal(), env.WALL.ordinal()};
	private int[] foreground_type = {env.OBST.ordinal(), env.LIGHT_STOP.ordinal(), env.LIGHT_PASS.ordinal()};
	//private int[] scale = {11, 16};
	//private int[] scale = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
	private int[] scale = { 0, 1, 2, 3, 4, 5};
	private int[] motor_location = { 0, 1, 2, 3, 4, 5};
	private int[] location = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
			24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42};
	private int current_type_id = 0;
	private int current_scale_id = 0;
	private int current_motor_scale_id = 0;
	private int current_loc_id = 0;
	private int current_motor_loc_id = 0;
    private int where_what_total = 200;
    private int train_where_what_count = 0;
    private int test_where_what_count = 20;
    private int[] error_where_what_scale = {0, 0, 0};
    private Random rand;
    
	/*
	 * This part deals with GUI specific parameters. If the GUI flag is true
	 * then would wait for the GUI signal. Default value for the continue_count
	 * is -1.
	 */
	private Neuron_Visualization mVisual;
	final int CONTINUE_NOT_INITIALIZED = -1;
	final int CONTINUE_TILL_END = -2;
	final int main_portNumber = 3000;
	final int flag_portNumber = 4000;
	private boolean paused = false;
	private long time_taken = 0;

	int continue_count = CONTINUE_NOT_INITIALIZED;
	private ServerSocket mainServerSocket = null;
	private ServerSocket flagServerSocket = null;
	private Socket mainClientSocket = null;
	private Socket flagClientSocket = null;
	private PrintWriter string_out = null;
	private DataOutputStream flag_out = null;
	private DataOutputStream data_out = null;
	private BufferedReader in = null;
	private String inputLine, outputLine;
	int display_y_zone = 1, display_num = 1, display_start_id = 0, display_z_zone_1 = 1, display_z_zone_2 = 1;
 // add iterations for skill training
	int iter = 0;
	int[] intial_random = {0, -13,-6, 6, 13};
	int[] angle_random = {0, 0, 0};
	
	public Board() throws IOException {
		initBoard();
	}

	public void connectSocket() throws IOException {
		// SetUp socket server
		if (use_socket_gui_flag) {
			mainServerSocket = new ServerSocket(main_portNumber);
			flagServerSocket = new ServerSocket(flag_portNumber);
			mainClientSocket = mainServerSocket.accept();
			flagClientSocket = flagServerSocket.accept();
			string_out = new PrintWriter(mainClientSocket.getOutputStream(), true);
			data_out = new DataOutputStream(mainClientSocket.getOutputStream());
			flag_out = new DataOutputStream(flagClientSocket.getOutputStream());

			in = new BufferedReader(new InputStreamReader(mainClientSocket.getInputStream()));

			// Initiate conversation with client
			outputLine = "Server Connected.";
			string_out.println(outputLine);
		}
	}

	private void initBoard() throws IOException {
		connectSocket();
		addKeyListener(new TAdapter());
		schedule = new Schedule("maps/teaching_schedule");
		agent = new Agent(schedule.getNumSkills(), schedule.getNumDestinations());
		vision_image = new BufferedImage(Agent.vision_num, Agent.vision_num * 3 /4, BufferedImage.TYPE_INT_RGB);
		timer = new Timer();
		timer.scheduleAtFixedRate(new ScheduleTask(), DELAY, PERIOD);
		setFocusable(true);
		setDoubleBuffered(true);
		ingame = true;
		learn_count = 0;
		current_lesson = 0;
		supervision_count = 0;
		emergent_concept = 0;
		emergent_destination = 0;
		old_destination = NULLVALUE;
		total_steps = 0;
		planning_flag = false;
		means_learned_flag = new boolean[schedule.getNumDestinations()];
		for (int i = 0; i < means_learned_flag.length; i++){
			means_learned_flag[i] = false;
		}
		rand = new Random(1234);
	}

	@Override
	public void addNotify() {
		super.addNotify();
		gameInit();
	}

	private void gameInit() {
        
		if (learn_count == lesson_iteration && lesson_iteration != 0) {
			current_lesson++;
			learn_count = 0;
			
			if(current_lesson == schedule.getNumSkills()) {
				System.out.println("save!!");
				agent.saveNet("maps/network1.ser");
			}
			//add iterations for skill training; 9 is original lessons && (current_lesson == schedule.getNumSkills())
			//if((iter < 4)&& (current_lesson == 2)){
			//	current_lesson = 0;
			//	iter = iter+1;
			//}
			
			if (current_lesson == schedule.getNumLessons()){
				stopGame();
				return;
			}	
		}
		String map_name = schedule.getName(current_lesson);
		lessonType type = schedule.getType(current_lesson);
		System.out.println(map_name);
		map = new Map(map_name, type);
		map.config();
		lesson_iteration = schedule.getIteration(current_lesson);
		train_flag = false;
		get_back_flag = false;
		light_interval_count = 0;

		if (map.supervised()){
			supervised_flag = SUPERVISED;
		} else if (map.getType() == PLANNING){
			supervised_flag = PLAN;
	    } else {
			supervised_flag = TEST;
			if ((learn_count == lesson_iteration - 1) && (lesson_iteration > 1)){
 				map.noisyGps(current_lesson - schedule.getNumSkills());
			}
		};

		bricks = new Brick[map.wall_list.size()];
		obstacles = new Obstacle[map.obst_list.size()];
		destinations = new Destination[map.dest_list.size()];
		rewards = new Reward[map.rwrd_list.size()];
		lights = new TrafficLight[map.light_list.size()];
		path = new ArrayList<Point>();
		current_distance = 0;
		
		// Pass in numLessions to initialize concept neuron.
		if (agent == null) {
			
		}
		if(current_lesson - schedule.getNumSkills()<0){            
			agent.setInitAgentLoc(map.getStartingLocS(intial_random[iter], 0), map.getStartingGpsAngle());			
		}
		else{
			Random random = new Random();
			int max1 = 5;
			int min1 = -5;
	        int rw = random.nextInt(max1)%(max1-min1+1) + min1;
	        int rh = random.nextInt(max1)%(max1-min1+1) + min1;
	        int max2 = 2;
			int min2 = 0;
			int a_index = random.nextInt(max2)%(max2-min2+1) + min2;
			int a = angle_random[a_index];
			if(rw <= -3){a = 0;}
	        
			agent.setInitAgentLoc(map.getStartingLocS(0, 0), map.getStartingGpsAngle());
		}		
		visions = new VisionLine[Agent.vision_num];
		currentGridLoc = new int[2];

		for (int i = 0; i < Agent.vision_num; i++) {
			visions[i] = new VisionLine(agent.getX(), agent.getY());
		}
		for (int i = 0; i < bricks.length; i++) {
			bricks[i] = new Brick(map.wall_list.get(i).x * 50, map.wall_list.get(i).y * 50);
		}
		for (int i = 0; i < obstacles.length; i++) {
			obstacles[i] = new Obstacle(map.obst_list.get(i).x * 50, map.obst_list.get(i).y * 50);
		}
		for (int i = 0; i < destinations.length; i++) {
			destinations[i] = new Destination(map.dest_list.get(i).x * 50, map.dest_list.get(i).y * 50);
		}
		for (int i = 0; i < rewards.length; i++) {
			rewards[i] = new Reward(map.rwrd_list.get(i).x * 50, map.rwrd_list.get(i).y * 50);
		}
		for (int i = 0; i < lights.length; i++) {
			if (map.light_list.get(i).type == URLT){
			    lights[i] = new TrafficLight(map.light_list.get(i).x * 50 + 30, map.light_list.get(i).y * 50);
			}
			if (map.light_list.get(i).type == LLLT){
				lights[i] = new TrafficLight(map.light_list.get(i).x * 50, map.light_list.get(i).y * 50 + 30);
			}
			if (map.light_list.get(i).type == LRLT){
				lights[i] = new TrafficLight(map.light_list.get(i).x * 50 + 30, map.light_list.get(i).y * 50 + 30);
			}
		}
		updateVision();
		System.out.print("the diff of angle: "+agent.getGpsDiff());
		learn_count++;
	}

	private void updateLoc(int x, int y, int gridWidth, int gridHeight) {
		currentGridLoc[0] = Math.floorDiv(y, gridHeight);
		currentGridLoc[1] = Math.floorDiv(x, gridWidth);
		//int temp = Math.floorMod(x, gridWidth);

		agent.updateGPS(map.getGpsAngle(currentGridLoc[0], currentGridLoc[1]));
		
	}

	private void updateVision() {
		updateLoc(agent.getX(), agent.getY(), bricks[0].getWidth(), bricks[0].getHeight());
		int max1 = (int)Agent.vision_range/20;
		int min1 = -max1;
		Random random = new Random();
		
		try {
			agent.rotateImage();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		for (int i_vision = 0; i_vision < Agent.vision_num; i_vision++) {
			visions[i_vision].start_x = agent.getX() + agent.getWidth()/2;
			visions[i_vision].start_y = agent.getY() + agent.getHeight()/2;
			visions[i_vision].end_x = visions[i_vision].start_x;
			visions[i_vision].end_y = visions[i_vision].start_y;
			visions[i_vision].type = OPEN;
			float angle = agent.angle + (float)Agent.visino_angle[i_vision];
			int range=Agent.vision_range;
			if(current_lesson >= schedule.getNumSkills()) {
				int noise = random.nextInt(max1)%(max1-min1+1)+min1;
				range=Agent.vision_range + noise;
			}
			for (int i_range = 0; i_range < range; i_range++) {
				boolean hit_brick = false;
				boolean hit_obstacle = false;
				boolean hit_light = false;
				visions[i_vision].end_x = visions[i_vision].start_x + (int) (i_range * Math.cos(Math.toRadians(angle)));
				visions[i_vision].end_y = visions[i_vision].start_y + (int) (i_range * Math.sin(Math.toRadians(angle)));

				// Check if collided with things.
				for (int i_brick = 0; i_brick < bricks.length; i_brick++) {
					float current_distance_x = Math
							.abs(visions[i_vision].end_x - (bricks[i_brick].x + bricks[i_brick].getWidth() / 2));
					float current_distance_y = Math
							.abs(visions[i_vision].end_y - (bricks[i_brick].y + bricks[i_brick].getHeight() / 2));
					if (current_distance_x <= bricks[i_brick].getWidth() / 2
							&& current_distance_y <= bricks[i_brick].getHeight() / 2) {
						hit_brick = true;
						visions[i_vision].type = WALL;
						break;
					}
				}

				for (int i_obst = 0; i_obst < obstacles.length; i_obst++) {
					float current_distance_x = Math
							.abs(visions[i_vision].end_x - (obstacles[i_obst].x + obstacles[i_obst].getWidth() / 2));
					float current_distance_y = Math
							.abs(visions[i_vision].end_y - (obstacles[i_obst].y + obstacles[i_obst].getHeight() / 2));
					if (current_distance_x <= obstacles[i_obst].getWidth() / 2
							&& current_distance_y <= obstacles[i_obst].getHeight() / 2) {
						hit_obstacle = true;
						visions[i_vision].type = OBST;
						break;
					}
				}
				
				for (int i_light = 0; i_light < lights.length; i_light++){
					float current_distance_x = Math
							.abs(visions[i_vision].end_x - (lights[i_light].x + lights[i_light].getWidth() / 2));
					float current_distance_y = Math
							.abs(visions[i_vision].end_y - (lights[i_light].y + lights[i_light].getHeight() / 2));
					if (current_distance_x <= lights[i_light].getWidth() / 2
							&& current_distance_y <= lights[i_light].getHeight() / 2) {
						hit_light = true;
						if (lights[i_light].getStatus() == true) {
						    visions[i_vision].type = LIGHT_PASS;
						} else {
							visions[i_vision].type = LIGHT_STOP;
						}
						break;
					}
				}

				if (hit_brick || hit_obstacle || hit_light) {
					break;
				}
			}
		}
		
		if (vision_2D_flag) {
			updateVisionImage();
		}
	}

	private void updateVisionImage() {
		int vision_height = vision_image.getHeight();
		int vision_width  = vision_image.getWidth();
		for (int i = 0; i < vision_width; i++) {
			if (visions[i].type == OPEN) {
				for (int j = 0; j < vision_height; j++){
					vision_image.setRGB(i, j, VISIONCOLORS[OPEN.ordinal()].getRGB());
				}
			} else if (visions[i].type == WALL) {
				for (int j = 0; j < vision_height; j++){
					vision_image.setRGB(i, j, VISIONCOLORS[WALL.ordinal()].getRGB());
				}
			} else if (visions[i].type == LIGHT_PASS){
				for (int j = 0; j < 15; j++){
					vision_image.setRGB(i, j, VISIONCOLORS[LIGHT_PASS.ordinal()].getRGB());
				}
				for (int j = 15; j < vision_height; j++){
					vision_image.setRGB(i, j, VISIONCOLORS[WALL.ordinal()].getRGB());
				}
			} else if (visions[i].type == LIGHT_STOP){
				for (int j = 0; j < 15; j++){
					vision_image.setRGB(i, j, VISIONCOLORS[LIGHT_STOP.ordinal()].getRGB());
				}
				for (int j = 15; j < vision_height; j++){
					vision_image.setRGB(i, j, VISIONCOLORS[WALL.ordinal()].getRGB());
				}
			} else if (visions[i].type == OBST){
				for (int j = 0; j < vision_height - 15; j++){
					vision_image.setRGB(i, j, VISIONCOLORS[OPEN.ordinal()].getRGB());
				}
				for (int j = 15; j < vision_height; j++){
					vision_image.setRGB(i, j, VISIONCOLORS[OBST.ordinal()].getRGB());
				}
			} else {
				throw new java.lang.Error("type not updated");
			}
		}
	}

	@Override
	public void paintComponent(Graphics g) {
		super.paintComponent(g);

		Graphics2D g2d = (Graphics2D) g;

		g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

		g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);

		if (ingame) {
			drawObjects(g2d);
		} else {
			gameFinished(g2d);
		}

		Toolkit.getDefaultToolkit().sync();
	}

	private void drawObjects(Graphics2D g2d) {

		g2d.drawImage(agent.getImage(), agent.getX(), agent.getY(), agent.getWidth(), agent.getHeight(), this);

		if (train_where_what_flag == false && test_where_what_flag == false){
	
			for (int i = 0; i < map.wall_list.size(); i++) {
				g2d.drawImage(bricks[i].getImage(), bricks[i].getX(), bricks[i].getY(), bricks[i].getWidth(),
						bricks[i].getHeight(), this);
			}
	
			for (int i = 0; i < map.dest_list.size(); i++) {
				g2d.drawImage(destinations[i].getImage(), destinations[i].getX(), destinations[i].getY(),
						destinations[i].getWidth(), destinations[i].getHeight(), this);
			}
	
			for (int i = 0; i < map.obst_list.size(); i++) {
				g2d.drawImage(obstacles[i].getImage(), obstacles[i].getX(), obstacles[i].getY(), obstacles[i].getWidth(),
						obstacles[i].getHeight(), this);
			}
	
			for (int i = 0; i < map.rwrd_list.size(); i++) {
				g2d.drawImage(rewards[i].getImage(), rewards[i].getX(), rewards[i].getY(), rewards[i].getWidth(),
						rewards[i].getHeight(), this);
			}
			
			for (int i = 0; i < map.light_list.size(); i++) {
				g2d.drawImage(lights[i].getImage(), lights[i].getX(), lights[i].getY(), lights[i].getWidth(),
						lights[i].getHeight(), this);
			}
			// Draw GPS arrow
			if (map.getGps(currentGridLoc[0], currentGridLoc[1]) != NONE){
			AffineTransform oldtrans = new AffineTransform();
			AffineTransform trans = new AffineTransform();
	
			trans.setToIdentity();
			trans.rotate(Math.toRadians(map.getGpsAngle(currentGridLoc[0], currentGridLoc[1])),
					agent.getX() + agent.getWidth() / 2, agent.getY() + agent.getHeight() / 2);
			g2d.setTransform(trans);
			g2d.drawImage(agent.heading.getImage(), agent.getX() + agent.getWidth() / 2,
					agent.getY() + agent.getHeight() / 2 - agent.heading.getHeight() / 2, agent.heading.getWidth(),
					agent.heading.getHeight(), null);
			trans.setToIdentity();
			g2d.setTransform(oldtrans);
			}
			// Paint agent path
			for (int i = 0; i < path.size() - 1; i++) {
				int start_x = path.get(i).x;
				int start_y = path.get(i).y;
				int end_x = path.get(i + 1).x;
				int end_y = path.get(i + 1).y;
				if (path.get(i + 1).mode) {
					g2d.setColor(Color.pink);
				} else {
					g2d.setColor(Color.gray);
				}
				g2d.drawLine(start_x, start_y, end_x, end_y);
			}
		}

		// Paint vision lines.
		for (int i = 0; i < visions.length; i++) {
			g2d.setColor(VISIONCOLORS[visions[i].type.ordinal()]);
			g2d.drawLine(visions[i].start_x, visions[i].start_y, visions[i].end_x, visions[i].end_y);
		}

		// Vision Panel
		Font font = new Font("Verdana", Font.PLAIN, 12);
		int currentY = VISIONTITLEY;
		g2d.setColor(Color.BLACK);
		g2d.setFont(font);
		g2d.drawString(VISIONTITLE, VISIONTITLEX, currentY);
		FontMetrics fm = g2d.getFontMetrics();
		String vision_string;
		for (int i = 0; i < agent.vision_order.length; i++) {
			int curr_order = agent.vision_order[i];
			vision_string = String.format("%.4f", visions[curr_order].getLength());
			Rectangle2D rect = fm.getStringBounds(vision_string, g2d);
			currentY += VISIONLINESPACE;
			g2d.setColor(VISIONCOLORS[visions[curr_order].type.ordinal()]);
			g2d.fillRect(VISIONTITLEX, currentY - fm.getAscent(), (int) rect.getWidth(), (int) rect.getHeight());
			g2d.setColor(Color.WHITE);
			g2d.drawString(vision_string, VISIONTITLEX, currentY);
		}
		

		// Display used number of neurons
		int[] used_neurons = agent.getUsedNeurons();
		String string_used = "Used: ";
		for (int i = 0; i < used_neurons.length; i++) {
			string_used += Integer.toString(used_neurons[i]) + " ";
		}
		currentY += VISIONLINESPACE;
		g2d.setColor(Color.BLACK);
		g2d.drawString(string_used, VISIONTITLEX, currentY);
		
		String string_time = "Update time: " + time_taken;
		currentY += VISIONLINESPACE;
		g2d.drawString(string_time, VISIONTITLEX, currentY);
		

		if (train_where_what_flag == false && test_where_what_flag == false){
			// GPS Info
			String gps_string = "GPS Diff: " + Integer.toString(agent.getGpsDiff());
			currentY += VISIONLINESPACE;
			g2d.drawString(gps_string, VISIONTITLEX, currentY);
			
			// 3rd input zone: block change flag.
			String block_string = "Block changed: " + Boolean.toString(block_change_flag);
			currentY += VISIONLINESPACE;
			g2d.setColor(Color.BLACK);
			g2d.drawString(block_string, VISIONTITLEX, currentY);
	
			// If testing
			String test_string;
			currentY += VISIONLINESPACE;
			if (supervised_flag != SUPERVISED) {
				test_string = "Testing!";
			} else {
				test_string = "Training, count : " + learn_count;
			}
			g2d.drawString(test_string, VISIONTITLEX, currentY);
	
			// Show number of supervisions
			String string_supervision = "Supervision: " + supervision_count;
			currentY += VISIONLINESPACE;
			g2d.drawString(string_supervision, VISIONTITLEX, currentY);
			
			// Show emergent concept during testing
			if (supervised_flag != SUPERVISED) {
				String string_concept = "Skill: " + emergent_concept;
				currentY += VISIONLINESPACE;
				g2d.drawString(string_concept, VISIONTITLEX, currentY);
			}
			
			String string_type;
			if (map.getType() == SKILL) {
				string_type = "Training skill";
			} else {
				string_type = "Training destination";
			}
			currentY += VISIONLINESPACE;
			g2d.drawString(string_type, VISIONTITLEX, currentY);
			
			String string_steps = "Steps: " + Integer.toString(total_steps);
			currentY += VISIONLINESPACE;
			g2d.drawString(string_steps, VISIONTITLEX, currentY);
			
			if (supervised_flag == PLAN) {
				String string_plan = "Planning values: ";
				currentY += VISIONLINESPACE;
				g2d.drawString(string_plan, VISIONTITLEX, currentY);
				for (int i = 0; i < predicted_value.length - 1; i++) {
					string_plan = "Means " + Integer.toString(i) + ": ";
					currentY += VISIONLINESPACE;
					g2d.drawString(string_plan, VISIONTITLEX, currentY);
					string_plan = "Firing cost id: " + String.format("%.2f", predicted_value[i]);
					currentY += VISIONLINESPACE;
					g2d.drawString(string_plan, VISIONTITLEX, currentY);
				}
			}
		}
		
		if (train_where_what_flag || test_where_what_flag){
			String string_where_what = "Current Loc: " + motor_location[current_motor_loc_id];
			currentY += VISIONLINESPACE;
			g2d.drawString(string_where_what, VISIONTITLEX, currentY);
			
			string_where_what = "Current Type: " + foreground_type[current_type_id];
			currentY += VISIONLINESPACE;
			g2d.drawString(string_where_what, VISIONTITLEX, currentY);
			
			string_where_what = "Current Scale: " + scale[current_motor_scale_id];
			currentY += VISIONLINESPACE;
			g2d.drawString(string_where_what, VISIONTITLEX, currentY);
			
			string_where_what = "Trained: " + train_where_what_count;
			currentY += VISIONLINESPACE;
			g2d.drawString(string_where_what, VISIONTITLEX, currentY);
		}
		
		if (test_where_what_flag){
			String string_where_what = "Error Loc: " + error_where_what_scale[0];
			currentY += VISIONLINESPACE;
			g2d.drawString(string_where_what, VISIONTITLEX, currentY);
			
			string_where_what = "Error Type: " + error_where_what_scale[1];
			currentY += VISIONLINESPACE;
			g2d.drawString(string_where_what, VISIONTITLEX, currentY);
			
			string_where_what = "Error Scale: " + error_where_what_scale[2];
			currentY += VISIONLINESPACE;
			g2d.drawString(string_where_what, VISIONTITLEX, currentY);
			
			string_where_what = "Tested: " + test_where_what_count;
			currentY += VISIONLINESPACE;
			g2d.drawString(string_where_what, VISIONTITLEX, currentY);
		}
		
		if (vision_2D_flag){
			currentY += VISIONLINESPACE;
			g2d.drawImage(vision_image, VISIONTITLEX, currentY, vision_image.getWidth() * 2, vision_image.getHeight() * 2, this);
		}
	}

	private void gameFinished(Graphics2D g2d) {

		Font font = new Font("Verdana", Font.BOLD, 18);
		FontMetrics metr = this.getFontMetrics(font);

		g2d.setColor(Color.BLACK);
		g2d.setFont(font);
		g2d.drawString(message, (map.map_height - metr.stringWidth(message)) / 2, map.map_width / 2);
	}

	private class TAdapter extends KeyAdapter {

		@Override
		public void keyPressed(KeyEvent e) {
			// agent.keyPressed(e);			
			int key = e.getKeyCode();
			if (key == KeyEvent.VK_SPACE){
				paused = !paused;
			}
			if (key == KeyEvent.VK_ENTER){ 
		        gameInit(); 
		    }
			if (key == KeyEvent.VK_V){
				paused = !paused;
				Scanner sc = new Scanner(System.in);
				if (paused == true){
				    mVisual = new Neuron_Visualization();
				    agent.sendInfoToGui();
					mVisual.updateDrawP();
					int visual_area = 1;
					int display_id = 1;
					mVisual.setVisualarea(visual_area);
					if(visual_area==0){
				        mVisual.setIndex(display_id, agent.getDisplayNum());}
					else{
						 mVisual.setIndex(display_id, agent.getTopDownNum());
					}
				    mVisual.updateInfopanel();
				    mVisual.updateDrawP();
				    mVisual.startHint();
				    mVisual.setVisible(true);
				}
			}
		}
	}

	private class ScheduleTask extends TimerTask {

		@Override
		public void run() {
			if (train_where_what_flag && !paused){
				boolean obj = updateWhereWhatVision();
				env current_type;
				if(obj) {
					current_type = env.values()[foreground_type[current_type_id]];
				}
				else {
					current_type = null;
				}				
				int current_loc  = motor_location[current_motor_loc_id];
				int current_scale = scale[current_motor_scale_id];
				long start_time = java.lang.System.nanoTime();
				agent.trainWhereWhat(visions, vision_image, current_type, current_loc, current_scale);
				long end_time = java.lang.System.nanoTime();
				time_taken = (end_time - start_time)/1000000;				
				train_where_what_count ++;
				/*
				boolean g = true;
				while(g) {
					try {
						if(getKey()) {
							g = false;
						}
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
				*/
				if (train_where_what_count > where_what_total){
					train_where_what_flag = false;
				}
				resetLocTypeScale();
				//updateLocTypeScale();
				//updateSocket();
				repaint();
			} else if (test_where_what_flag && !paused){
				boolean obj2 = updateWhereWhatVision();
				env current_type;
				if(obj2) {
					current_type = env.values()[foreground_type[current_type_id]];
				}
				else {
					current_type = null;
				}	
				int current_loc  = motor_location[current_motor_loc_id];
				int current_scale = scale[current_motor_scale_id];
				// where, what, scale
				int[] result = agent.testWhereWhat(visions, vision_image, current_type, current_loc, current_scale);
				if (result[0] != current_loc){
					error_where_what_scale[0] ++;
				}
				if (result[1] != current_type.ordinal()){
					error_where_what_scale[1] ++;
				}
				if (result[2] != current_scale){
					error_where_what_scale[2] ++;
				}
				
				test_where_what_count ++;
				if (test_where_what_count > where_what_total){
					test_where_what_flag = false;
				}
				resetLocTypeScale();
				//updateLocTypeScale();
				//updateSocket();
				repaint();
			}
			else if (!paused){
			if (supervised_flag == SUPERVISED) {
				// If the agent's emergent action is not the same as we intended, supervised and increment the counter.
				if (currentGridLoc[0] == map.getInitGridLoc()[0] && currentGridLoc[1] == map.getInitGridLoc()[1] &&
					Math.abs(agent.getAngle() - map.getInitGpsAngle())<=5){
					train_flag = true;
				}
				long start_time = java.lang.System.nanoTime();
				supervision_count += agent.getSupervision(visions, vision_image, map.getGps(currentGridLoc[0], currentGridLoc[1]), 
						                                  current_lesson, schedule.getNumSkills(), map.getPreviousSkill(), 
						                                  train_flag, current_distance, block_change_flag);
				long end_time = java.lang.System.nanoTime();
				time_taken = (end_time - start_time)/1000000;
				path.add(new Point((int)(agent.getX() + 0.5 * agent.getWidth()), 
						           (int)(agent.getY() + 0.5 * agent.getHeight()), train_flag));
			} else if (supervised_flag == TEST) {
				if (get_back_flag == false){
					if (current_lesson >= schedule.getNumSkills() && 
						means_learned_flag[current_lesson - schedule.getNumSkills()] == false){
					    agent.learnMeansLoop(current_lesson - schedule.getNumSkills());
					    means_learned_flag[current_lesson - schedule.getNumSkills()] = true;
					}
				    emergent_concept = agent.getEmergentAction(visions, vision_image, map.getGps(currentGridLoc[0], currentGridLoc[1]), 
						                                   current_lesson, schedule.getNumSkills(), map.getPreviousSkill(),
						                                   current_distance, block_change_flag);
				    path.add(new Point((int)(agent.getX() + 0.5 * agent.getWidth()), 
					           (int)(agent.getY() + 0.5 * agent.getHeight()), true));
				} else {
					agent.getSupervision(visions, vision_image, map.getGps(currentGridLoc[0], currentGridLoc[1]), 
                                         current_lesson, schedule.getNumSkills(), map.getPreviousSkill(), false,
                                         current_distance, block_change_flag);
					path.add(new Point((int)(agent.getX() + 0.5 * agent.getWidth()), 
					           (int)(agent.getY() + 0.5 * agent.getHeight()), false));
				}
			} else if (supervised_flag == PLAN) {
				PlanResult plan = agent.plan(visions, vision_image, map.getGps(currentGridLoc[0], currentGridLoc[1]), block_change_flag);
				emergent_destination = plan.result.destination;
				System.out.println("emergent_destination: "+emergent_destination);
				predicted_value = new float[plan.values.length];
				for (int i = 0; i < plan.values.length; i++) {
					predicted_value[i] = plan.values[i];
				}
				path.add(new Point((int)(agent.getX() + 0.5 * agent.getWidth()), 
				           (int)(agent.getY() + 0.5 * agent.getHeight()), false));
				
				// If emergent_destination changes, load new map gps as the destination changes.
				if (emergent_destination != old_destination && emergent_destination != NULLVALUE) {
					String new_name = schedule.getName(schedule.getNumSkills() + emergent_destination);
					map.resetGps(new_name);
				}
				old_destination = emergent_destination;
			}
			
			//updateSocket();

			boolean moved = agent.move();
			if (moved) {
				current_distance += step_value ;
			}
			total_steps ++ ;
			updateMap(); // if get back and returned to starting point, then init game
			updateVision();
			checkCollision();
			repaint();
			float agent_x = agent.getX() + 0.5f * agent.getWidth();
			float agent_y = agent.getY() + 0.5f * agent.getHeight();
			float brick_w = bricks[0].getWidth();
			float brick_h = bricks[0].getHeight();
			if (agent_x/brick_w == Math.floor(agent_x/brick_w) || 
				agent_y/brick_h == Math.floor(agent_y/brick_h)){
				block_change_flag = true;
			} else {
				block_change_flag = false;
			}
		}
		}
	}
	
	private void updateSocket(){
		if (use_socket_gui_flag) {
			if (continue_count == 0 || continue_count == CONTINUE_NOT_INITIALIZED) {
				if (continue_count == 0) {
					try {
						System.out.println("Count: 0");
						DNCaller.getInstance().sendOverSocket(string_out, data_out, display_y_zone, display_num,
								display_start_id, display_z_zone_1, display_z_zone_2);
						flushSockets();
					} catch (IOException e) {
						e.printStackTrace();
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
				try {
					while ((inputLine = in.readLine()) != null) {
						System.out.println(inputLine);
						if (inputLine.equals("End")) {
							stopGame();
							return;
						} else if (inputLine.equals("Start")) {
							continue_count = CONTINUE_TILL_END;
							break;
						} else if (inputLine.equals("Continue")) {
							String temp_string = in.readLine();
							continue_count = Integer.parseInt(temp_string);
							System.out.println("Count: " + Integer.toString(continue_count));
							readDisplayParam();
							break;
						} else if (inputLine.equals("Next") || inputLine.equals("Previous")
								|| inputLine.equals("Update")) {
							readDisplayParam();
							DNCaller.getInstance().sendOverSocket(string_out, data_out, display_y_zone, display_num,
									display_start_id, display_z_zone_1, display_z_zone_2);
							flushSockets();
						} else {
							outputLine = "Unrecognized command";
							string_out.println(outputLine);
						}
					}
				} catch (NumberFormatException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			continue_count--;
		}
	}
	
	public void updateLocTypeScale() {
		// update scale, loc and type id
		current_loc_id ++;
		if (current_loc_id >= location.length || (location[current_loc_id] + scale[current_scale_id] - 1>= Agent.vision_num)){
			current_loc_id = 0;
			current_type_id ++;
		}
		if (current_type_id >= foreground_type.length){
			current_type_id = 0;
			current_scale_id ++;
		}
		if (current_scale_id >= scale.length){
			current_scale_id = 0;
		}
	}

	public void resetLocTypeScale() {
		current_type_id = 0;
		current_loc_id = 0;
		current_scale_id = 0;
	}
	
	private boolean updateWhereWhatVision() {
		// generate vision and vision image according to current_type_id, current_loc_id, current_scale_id
		// first, generate random input vision lines
		current_motor_loc_id = 0;
		current_motor_scale_id = 0;
		boolean ol = false;
		boolean lr = false;
		boolean lg = false;
		boolean ot = false;
		int startObj = 0;
		int endObj = 0;
		int startLgt = 0;
		int endLgt = 0;
		
		int nObj = randInt(2,10);
		int nLgt = randInt(2,10);
		int startIn = 0;
		int endIn = randInt(10, 30);
		env current_vision_type = env.values()[background_type[randInt(0, background_type.length)]];	
		float current_vision_length = Agent.vision_range;
		//env current_vision_type = env.values()[background_type[1]];  //set to wall
		while(endIn <= Agent.vision_num) {
			if(current_vision_type == OPEN) {
				current_vision_length = Agent.vision_range;	
				int oblen = (endIn-startIn)/2;
				if(nObj > 6) {
					startObj = startIn+randInt(0, oblen-4);
					endObj = startObj+randInt(6, oblen+3);					
					nObj = 1;
					ot = true;
					ol = true;
					current_loc_id = startObj;
					current_scale_id = endObj-startObj;
				}
			}
			else {
			//	current_vision_length = randFloat(agent.safe_distance, Agent.vision_range);				
				if(nLgt > 4 && nLgt <= 7 && startIn > 0) {				
					startLgt = startIn;
					endLgt =  Math.min(randInt(startLgt+3,startLgt+12),endIn-3);
					nLgt = 1;				
					ol = true;
					current_loc_id = startLgt;
					current_scale_id = endLgt-startLgt;
				}
				if(nLgt > 7 && endIn < Agent.vision_num-1) {									
					endLgt = endIn;
					startLgt = Math.max(randInt(endLgt-12, endLgt-3), startIn+3);
					nLgt = 1;
					ol = true;
					current_loc_id = startLgt;
					current_scale_id = endLgt-startLgt;
				}
			}
			for (int i = startIn; i < endIn; i++){				
				float angle = agent.angle + (float)Agent.visino_angle[i];
               /*
				if(current_vision_type == WALL) {
    				if(i <  Agent.vision_num/2) {
    					if(current_vision_length < Agent.vision_range)
    					{current_vision_length += i*3;}
    					else {current_vision_length = Agent.vision_range;}
    				}
    				else {
    					if(current_vision_length >= agent.safe_distance+3) {
    						current_vision_length -= (i-Agent.vision_num/2)*3;
    					}
    					else {current_vision_length = agent.safe_distance;}
    				} 
                }*/
				visions[i].setLength(current_vision_length, angle);
				visions[i].setType(current_vision_type);				
			}
			
			if(nLgt == 1) {
				int curr_type = randInt(1,10);
				if(curr_type > 5) {
					current_vision_type = LIGHT_STOP;
					lr = true;
				}
				else {
					current_vision_type = LIGHT_PASS;
					lg = true;
				}
				for (int i = startLgt; i < endLgt; i++){					                    
					visions[i].setType(current_vision_type);				
				}
				nLgt -= 1;
			}
			
			if(nObj == 1) {
			//	current_vision_length = randFloat(agent.safe_distance, Agent.vision_range);
				for (int i = startObj; i < endObj; i++){
					current_vision_type = OBST;
					float angle = agent.angle + (float)Agent.visino_angle[i];
    				/*
					if(i <  Agent.vision_num/2) {
    					if(current_vision_length <= Agent.vision_range)
    					{current_vision_length += i*2;}
    					else {current_vision_length = Agent.vision_range;}
    				}
    				else {
    					if(current_vision_length >= agent.safe_distance+2) {
    						current_vision_length -= (i-Agent.vision_num/2)*2;
    					}
    					else {current_vision_length = agent.safe_distance;}
    				} 
    				*/
					visions[i].setLength(current_vision_length, angle);
					visions[i].setType(current_vision_type);				
				}
				nObj -= 1;
			}
			
			startIn = endIn;
            if(startIn >= Agent.vision_num) {
            	endIn = Agent.vision_num+1;
            }            
            else {
    			endIn = startIn+randInt(10, 30);
    			if(endIn > Agent.vision_num-5) {
    				endIn = Agent.vision_num;
    			}
            }

			if(current_vision_type == OPEN || current_vision_type == OBST) {
				current_vision_type = WALL;
			}
			else {
				current_vision_type = OPEN;
			}
		}
		if(lg == true) {
			current_type_id = 2;
		}
		if(ot == true) {
			current_type_id = 0;
		}
		if(lr == true) {
			current_type_id = 1;
		}

    	if(current_scale_id <=5){
    		current_motor_scale_id = 1;
    	}
    	else if(current_scale_id >5 && current_scale_id <=10){
    		current_motor_scale_id = 2;
    	}
    	else if(current_scale_id >10 && current_scale_id <=15){
    		current_motor_scale_id = 3;
    	}
    	else if(current_scale_id >15 && current_scale_id <=20){
    		current_motor_scale_id = 4;
    	}
    	else{
    		current_motor_scale_id = 5;
    	}
 
    	if(current_loc_id < 8){
    		current_motor_loc_id = 1;
    	}
    	else if(current_loc_id > 34){
    		current_motor_loc_id = 5;
    	}
    	else if(current_loc_id >= 8 && current_loc_id < 16){
    		current_motor_loc_id = 2;
    	}
    	else if(current_loc_id > 26 && current_loc_id <= 34){
    		current_motor_loc_id = 4;
    	}
    	else{
    		current_motor_loc_id = 3;
    	}
    	
		updateVisionImage();
		return ol;
	}
	
	private void updateMap(){
		if (get_back_flag == true && currentGridLoc[0] == 0 && 
			currentGridLoc[1] == 1){
			gameInit();
		}
		light_interval_count ++ ;
		if (Math.floorMod(light_interval_count, traffic_light_interval) == 0){
			for (int i = 0; i < lights.length; i++){
				lights[i].revertStatus();
			}
		}
	}

	private void flushSockets() throws IOException {
		data_out.flush();
		// flag_out.writeInt(1);
		flag_out.flush();
	}

	private void readDisplayParam() throws IOException {
		String temp_string = in.readLine();
		display_y_zone = Integer.parseInt(temp_string);

		temp_string = in.readLine();
		display_num = Integer.parseInt(temp_string);

		temp_string = in.readLine();
		display_start_id = Integer.parseInt(temp_string);

		temp_string = in.readLine();
		display_z_zone_1 = Integer.parseInt(temp_string);

		temp_string = in.readLine();
		display_z_zone_2 = Integer.parseInt(temp_string);
	}

	private void stopGame() {
		ingame = false;
		timer.cancel();
		System.out.print(agent.max_landmark_size + ", " + agent.min_landmark_size);
	}

	private void checkCollision() {
		for (int i = 0; i < bricks.length; i++) {
			if (agent.getRect().intersects(bricks[i].getRect())) {
				message = "Hit walls";
				gameInit();
			}
		}

		for (int i = 0; i < obstacles.length; i++) {
			if (agent.getRect().intersects(obstacles[i].getRect())) {
				message = "Hit obstacles";
				gameInit();
			}
		}

		for (int i = 0; i < destinations.length; i++) {
			if (agent.getRect().intersects(destinations[i].getRect())) {
				message = "Arrived";
				if (get_back_flag == false){
				    get_back_flag = true;
				    map.updateGpsBackWard();
				}
				train_flag = false;
			}
		}

		for (int i = 0; i < rewards.length; i++) {
			if (agent.getRect().intersects(rewards[i].getRect())) {
				if (get_back_flag == false){
					// current_distance += map.getValue();
				    agent.learnReward(map.getValue());
				    if (current_lesson < schedule.getNumLessons() - 1){
				        agent.learnPlanning(map.getPlanSequence(), current_lesson - schedule.getNumSkills(), map.getValue(), 
				         		            current_distance);
				        if (means_learned_flag[0] == true && means_learned_flag[1] == true && means_learned_flag[2] == true){
				            agent.learnCovertToCovert();
				            agent.learnMoreLess();
				        }
				    }
				}
				message = "Arrived";
				if (current_lesson != schedule.getNumLessons() - 1){
				    if (get_back_flag == false){
				        get_back_flag = true;
				        map.updateGpsBackWard();
				    }
				    train_flag = false;
				} else {
			        agent.saveNet("maps/network2.ser");
				    gameInit();
				}
			}
		}
	}
	
	public int randInt(int min, int max) {
	    int randomNum = rand.nextInt(max - min) + min;

	    return randomNum;
	}
	
	public float randFloat(float min, float max) {
	    float randomNum = rand.nextFloat() * (max - min) + min;
	    return randomNum;
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
