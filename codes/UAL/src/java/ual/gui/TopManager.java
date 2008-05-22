package ual.gui;


/**
 * Top Manager of the UAL Simulation Facility
 * 
 * @author  Nikolay Malitsky
 * @version $Id: TopManager.java,v 1.1 2003/02/28 20:41:40 ual Exp $
 */

public abstract class TopManager {
    
    /** Default top manager */
    private static TopManager defaultTopManager;
    

    /**
     * Returns the only instance of the Top Manager.
     */
    public static TopManager getInstance () {
        if (defaultTopManager != null) {
            return defaultTopManager;
        }

        return initializeTopManager ();
    }
            

    /** Initializes the top manager. */
    private static synchronized TopManager initializeTopManager () {
        
        if (defaultTopManager != null) {
            return defaultTopManager;
        }

        String className = System.getProperty("ual.gui.TopManager");

        try {
            Class c = Class.forName(className);
            defaultTopManager = (TopManager)c.newInstance();
        } catch (Exception ex) {
            ex.printStackTrace();
            return defaultTopManager;
        }

        // late initialization of the manager if needed
        if (defaultTopManager instanceof Runnable) {
            ((Runnable)defaultTopManager).run ();
        }

        return defaultTopManager;
    }
    
}
