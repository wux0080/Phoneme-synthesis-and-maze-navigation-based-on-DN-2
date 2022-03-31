package MazeInterface;
import javax.swing.ImageIcon;

public class Reward extends Sprite {

    private boolean destroyed;

    public Reward(int x, int y) {
        
        this.x = x;
        this.y = y;

        ImageIcon ii = new ImageIcon(this.getClass().getResource("Images/Reward.png"));
        image = ii.getImage();

        i_width = image.getWidth(null);
        i_heigth = image.getHeight(null);
    }

    public boolean isDestroyed() {
        
        return destroyed;
    }

    public void setDestroyed(boolean val) {
        
        destroyed = val;
    }
}
