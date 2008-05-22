package ual.gui;

import java.awt.*;
import javax.swing.*;

/**
 * Common predecessor of all the UAL application windows. 
 * 
 * @author  Nikolay Malitsky
 * @version $Id: ApplicationWindow.java,v 1.1 2003/02/28 20:41:40 ual Exp $
 */

public class ApplicationWindow extends JFrame {
    
    // Actions 
    // protected Actions actions;
    
    /* The Application's menu */
    protected JMenuBar menuBar;

    /* The Application's toolbar */
    protected transient JToolBar toolBar;
    
    /** Initializes window components */
    public void initWindow(){
    }
        
    /** Shows window */
    public void showWindow(){       
        pack(); 
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
        Dimension windowSize = getSize();
        setLocation((screenSize.width-windowSize.width)/2,
                    (screenSize.height-windowSize.height)/2);
        show();
        getRootPane().requestDefaultFocus();
    }
    
    // Returns the list of window actions    
    // public Actions getActions(){
    //    return actions;
    //}
   
    /** Returns the preferred window size */
    public Dimension getPreferredSize() {
        Dimension size = super.getPreferredSize();

        Dimension minimumSize = getMinimumSize();
        Dimension maximumSize = getMaximumSize(); 
        
        if (size.width < minimumSize.width) size.width = minimumSize.width;
        if (size.width > maximumSize.width) size.width = maximumSize.width;

        if (size.height < minimumSize.height) size.height = minimumSize.height;
        if (size.height > maximumSize.height) size.height = maximumSize.height;
        
        // return size;
        return minimumSize;
    }

    /** Returns the minimum window size */
    public Dimension getMinimumSize() {
        return new Dimension(700, 400);
    }

    /** Returns the maximum window size */
    public Dimension getMaximumSize() {
        return Toolkit.getDefaultToolkit().getScreenSize();
    }
}
