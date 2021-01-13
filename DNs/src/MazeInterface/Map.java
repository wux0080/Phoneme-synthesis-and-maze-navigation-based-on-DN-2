package MazeInterface;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class Map implements Commons{
	private lessonType type;
    public int map_width;  // pixels
    public int map_height; // pixels
    private int map_grid_width;
    private int map_grid_height;
    private int map_starting_height;
    private int map_starting_width;
    private boolean supervised_flag;
    private gps map_starting_angle;
    private int previous_skill;
    private int value;
    private int planning_length;
    private int[] planning;
    public ArrayList<ObjPos> wall_list;
    public ArrayList<ObjPos> obst_list;
    public ArrayList<ObjPos> dest_list;
    public ArrayList<ObjPos> rwrd_list;
    public ArrayList<ObjPos> light_list;

	private env[][] env_layout;
	
	private gps[][] env_gps;
	
	private gps[][] reverse_gps;
	
	public boolean supervised(){
		return supervised_flag;
	}
	
	public int[] getInitGridLoc(){
		return new int[] {map_starting_height, map_starting_width};
	}
	
	public int[] getInitLoc(){
		return new int[]{(int) ((map_starting_width + 0.5) * BRICK_SIZE), 
				         (int) ((map_starting_height + 0.5) * BRICK_SIZE)};
	}
	
	public int[] getStartingLoc(){
		return new int[]{(int) ((1 + 0.5) * BRICK_SIZE), 
		         (int) ((0 + 0.5) * BRICK_SIZE)};
	}
	
	public int[] getStartingLocS(int a, int b){
		return new int[]{(int) ((1 + 0.5) * BRICK_SIZE)+a, 
		         (int) ((0 + 0.5) * BRICK_SIZE)+b};
	}
	
	public Map(String filename, lessonType lesson_type){
		type = lesson_type;
		Scanner reader;
		try {
			reader = new Scanner(new File(filename));
			int parameterCount = 0;

			while (reader.hasNext()) {
				String temp = reader.next();
                
				if (temp.charAt(0) != '/'){
					switch (parameterCount) {
					case 0:
						supervised_flag = Boolean.parseBoolean(temp);
						parameterCount ++ ;
						break;
					case 1:
						map_grid_height = Integer.parseInt(temp);
						parameterCount ++ ;
						break;
					case 2:
						map_grid_width = Integer.parseInt(temp);
						parameterCount ++ ;
						break;
					case 3:
						map_starting_height = Integer.parseInt(temp);
						parameterCount ++ ;
						break;
					case 4:
						map_starting_width = Integer.parseInt(temp);
						parameterCount ++ ;
						break;
					case 5:
						map_starting_angle = Commons.parseGpsString(temp);
						parameterCount ++ ;
						break;
					case 6:
						previous_skill = Integer.parseInt(temp);
						if (previous_skill == -1) {
							previous_skill = NULLVALUE;
						}
						parameterCount ++ ;
						break;
					case 7:
						planning_length = Integer.parseInt(temp);
						parameterCount ++ ;
						break;
					case 8:
						env_layout = new env[map_grid_height][map_grid_width];
						for (int i = 0; i < map_grid_height; i++){
							reader.nextLine();
							for (int j = 0; j < map_grid_width; j++){
								temp = reader.next();
								temp.replaceAll(",", "");
								env_layout[i][j] = Commons.parseEnvString(temp);
								if (env_layout[i][j] == RWRD){
									String value_string = temp.replaceAll("RWRD", "");
									value = Integer.parseInt(value_string);
								}
							}
						}
						parameterCount ++ ;
						break;
					case 9:
						env_gps = new gps[map_grid_height][map_grid_width];
						for (int i = 0; i < map_grid_height; i++){
							reader.nextLine();
							for (int j = 0; j < map_grid_width; j++){
								temp = reader.next();
								temp.replaceAll(",", "");
								env_gps[i][j] = Commons.parseGpsString(temp);
							}
						}
						parameterCount ++ ;
						break;
					case 10:
						reverse_gps = new gps[map_grid_height][map_grid_width];
						for (int i = 0; i < map_grid_height; i++){
							reader.nextLine();
							for (int j = 0; j < map_grid_width; j++){
								temp = reader.next();
								temp.replaceAll(",", "");
								reverse_gps[i][j] = Commons.parseGpsString(temp);
							}
						}
						parameterCount ++ ;
						break;
					case 11:
						planning = new int[planning_length];
						reader.nextLine();
						for (int i = 0; i < planning_length; i++){
							temp = reader.next();
							temp.replaceAll(",", "");
							planning[i] = Integer.parseInt(temp);
						}
						parameterCount ++ ;
						break;
					}
				} else {
					reader.nextLine();
				}
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public env[][] getLayOut(){
		return env_layout;
	}
	
	public class ObjPos{
		public ObjPos(env type, int x, int y){
			this.type = type;
			this.x = x;
			this.y = y;
		}
		public env type;
		public int x;
		public int y;
	}
	
	public void config(){
		map_height = env_layout.length * Commons.BRICK_SIZE;
		map_width = env_layout[0].length * Commons.BRICK_SIZE;
		
		wall_list = new ArrayList<ObjPos>();
		obst_list = new ArrayList<ObjPos>();
		dest_list = new ArrayList<ObjPos>();
		rwrd_list = new ArrayList<ObjPos>();
		light_list = new ArrayList<ObjPos>();
		
		Random r = new Random();
		
		for(int i = 0; i < env_layout.length; i++){
			for (int j = 0; j < env_layout[i].length; j++){
				if (env_layout[i][j] == RAND){
					// RAND can be wall, obstacle, or open
					int result = r.nextInt(3);
					switch (result){
					case 0:
						env_layout[i][j] = WALL;
						break;
					case 1:
						env_layout[i][j] = OBST;
						break;
					case 2:
						env_layout[i][j] = OPEN;
						break;
					}
					
				}
				if (env_layout[i][j] == WALL){
					wall_list.add(new ObjPos(WALL, j, i));
				}
				if (env_layout[i][j] == DEST){
					dest_list.add(new ObjPos(DEST, j, i));
				}
				if (env_layout[i][j] == OBST){
					obst_list.add(new ObjPos(OBST, j, i));
				}
				if (env_layout[i][j] == RWRD){
					rwrd_list.add(new ObjPos(RWRD, j, i));
				}
				if (env_layout[i][j] == URLT){
					wall_list.add(new ObjPos(WALL, j, i));
					light_list.add(new ObjPos(URLT, j, i));
				}
				if (env_layout[i][j] == LLLT){
					wall_list.add(new ObjPos(WALL, j, i));
					light_list.add(new ObjPos(LLLT, j, i));
				}
				if (env_layout[i][j] == LRLT){
					wall_list.add(new ObjPos(WALL, j, i));
					light_list.add(new ObjPos(LRLT, j, i));
				}
			}
		}
	}
	
	public int getInitGpsAngle(){
		return gps_angles[map_starting_angle.ordinal()];
	}
	
	public int getStartingGpsAngle(){
		return gps_angles[DOWN.ordinal()];
	}
	
	public int getStartingGpsAngleS(int a){
		return gps_angles[DOWN.ordinal()]+a;
	}
	
	public int getGpsAngle(int i, int j){
		System.out.println(Integer.toString(i) + Integer.toString(j));
		return gps_angles[env_gps[i][j].ordinal()];
	}
	
	public gps getGps(int i, int j){
		return env_gps[i][j];
	}
	
	public lessonType getType() {
		return type;
	}
	
	public gps[][] getGps(){
		return env_gps;
	}
	
	public int getPreviousSkill(){
		return previous_skill;
	}
	
	public int getValue(){
		return value;
	}
	
	public void resetGps(String map_name) {
		Map temp_map = new Map(map_name, type);
		env_gps = temp_map.getGps();
	}
	
	public int[] getPlanSequence(){
		return planning;
	}
	
	public void updateGpsBackWard(){
		for (int i = 0; i < env_gps.length; i++){
			for (int j = 0; j < env_gps[0].length; j++){
				env_gps[i][j] = reverse_gps[i][j];
			}
		}
	}

	public void noisyGps(int dest_num) {
		if (dest_num == 0){
		    env_gps[3][1] = NONE;
		    env_gps[3][2] = NONE;
		    env_gps[6][4] = NONE;
		} else if (dest_num == 1){
			env_gps[1][3] = NONE;
			env_gps[2][6] = NONE;
			env_gps[6][4] = NONE;
		}
	}
}
