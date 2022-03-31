package MazeInterface;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class Schedule implements Commons{
	private MapIteration[] lessons;
	private int num_lessons;
	private int num_skills;
	private int num_destinations;
	private int num_planning;
	public Schedule(String schedule_file){
		Scanner reader;
		try {
			reader = new Scanner(new File(schedule_file));
			int parameterCount = 0;

			while (reader.hasNext()) {
				String temp = reader.next();
                
				if (temp.charAt(0) != '/'){
					switch (parameterCount) {
					case 0:
						num_skills = Integer.parseInt(temp);
						parameterCount ++ ;
						break;
					case 1:
						num_destinations = Integer.parseInt(temp);
						parameterCount ++ ;
						break;
					case 2:
						num_planning = Integer.parseInt(temp);
						num_lessons = num_destinations + num_skills + num_planning;
						lessons = new MapIteration[num_lessons];
						parameterCount ++ ;
						break;
					case 3:
						for (int i = 0; i < num_lessons; i++){
							reader.nextLine();
							String file_name = reader.next();
							file_name.replaceAll(",", "");
							int iteration = Integer.parseInt(reader.next());
							lessons[i] = new MapIteration(file_name, iteration);
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
	
	public String getName(int i){
		return lessons[i].getName();
	}
	
	public int getNumLessons(){
		return lessons.length;
	}
	
	public int getIteration(int i){
		return lessons[i].getIteration();
	}
	
    private class MapIteration {
    	private String map_name;
    	private int iterations;
    	
    	MapIteration(String name, int i){
    		map_name = name;
    		iterations = i;
    	}
    	
    	public String getName(){
    		return map_name;
    	}
    	
    	public int getIteration(){
    		return iterations;
    	}
    }

	public lessonType getType(int current_lesson) {
		if(current_lesson < num_skills) {
			return SKILL;
		} else if (current_lesson < num_skills + num_destinations){
			return DESTINATION;
		} else {
			return PLANNING;
		}
	}

	public int getNumSkills() {
		return num_skills;
	}
	
	public int getNumDestinations(){
		return num_destinations;
	}
}
