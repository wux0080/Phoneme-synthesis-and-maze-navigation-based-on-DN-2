package MazeInterface;
import javax.swing.ImageIcon;

public class Brick extends Sprite {

    private boolean destroyed;

    public Brick(int x, int y) {
        
        this.x = x;
        this.y = y;

        ImageIcon ii = new ImageIcon(this.getClass().getResource("Images/Brick.png"));
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
