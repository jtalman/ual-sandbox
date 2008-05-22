package ual.gui;


/**
 * Main class of the UAL Simulation Facility.
 * 
 * @author  Nikolay Malitsky
 * @version $Id: Main.java,v 1.1 2003/02/28 20:41:40 ual Exp $
 */

public final class Main extends NonGui {
    
    /** Initialization code. */
    public void run () {
        
        // do the non gui initialization
        super.run ();
        
        // initialize the main window
        initializeMainWindow (); 
        
    }
    
    /** Initializes the main window. */
    private void initializeMainWindow () {
 
        // Get the main window
	WindowManager wm = new WindowManager();
        ApplicationWindow aw = wm.getMainWindow();       
        
        // Initialize the main window
        wm.initMainWindow();
        
        // Open main window
        wm.showMainWindow();

    }
    
    /** Main program */
    public static void main(String[] args)  {
        
        // read command arguments
        NonGui.parseCommandLine (args);

        // show the welcome screen
        /* if (!noSplash) {
            splash = Splash.showSplash ();
        } */

        // initialize the TopManager   
        System.getProperties().put ("ual.gui.TopManager", "ual.gui.Main");
        TopManager manager = TopManager.getInstance ();

        // finish starting
        /*if (splash != null) {
        Splash.hideSplash(splash);
        splash = null;
        }*/
    }

}
