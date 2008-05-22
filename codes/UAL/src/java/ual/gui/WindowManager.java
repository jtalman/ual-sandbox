package ual.gui;


/**
 *  Manager of UAL application windows
 * 
 * @author  Nikolay Malitsky
 * @version $Id: WindowManager.java,v 1.1 2003/02/28 20:41:40 ual Exp $
 */

public class WindowManager {
    
    /** The main window of the UAL Simulation Facility */
    static ApplicationWindow mainWindow;
    
    /** Constructor */
    public WindowManager () {
    }
    
    /** Returns Main Window */
    public synchronized ApplicationWindow getMainWindow () {
        if (mainWindow == null) {
            mainWindow = new ual.gui.ApplicationWindow () {
		    public java.awt.Dimension getMinimumSize() {
			return new java.awt.Dimension(300, 100);
		    }
		};
	    mainWindow.setTitle("UAL Main Window");
	    mainWindow.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
            // mainWindow.addWindowListener(new IconifyManager());
        }
        return mainWindow;
    } 

    /** Initializes Main Window  */
    public  void initMainWindow () {
	ApplicationWindow aw = getMainWindow();
	aw.initWindow();
    }   

    /** Shows Main Window  */
    public  void showMainWindow () {
	ApplicationWindow aw = getMainWindow();
	aw.showWindow();
    }   
 
    
}
