
# LanePerception receives the characteristics of each line detection
class LanePerception:
    def __init__(self):
        # Was the line detected in the last iteration?
        self.was_detected_m = False
    
        # x values of the last n fits of the line
        self.recent_xfitted_m = []
        
        # average x values of the fitted line over the last n iterations
        self.bestx_m = None
        
        # polynomial coefficients averaged over the last n iterations
        self.best_fit_m = None
        
        # polynomial coefficients for the most recent fit
        self.current_fit_m = [np.array([False])]
        
        # radius of curvature of the line in some units
        self.radius_of_curvature_m = None
        
        # distance in meters of vehicle center from the line
        self.line_base_pos_m = None
        
        # difference in fit coefficients between last and new fits
        self.diffs_m = np.array([0,0,0], dtype = 'float')
        
        # x values for detected line pixels
        self.allx = None
        
        # y values for detected line pixels
        self.ally = None
        
    def sanity_check(self):
        """
            Some lines were found, before moving on, checks that the detection makes
            sense. Confirms detected lane lines are real by:
                - Checking they have similar curvature
                - Checking they are separated by approximately the right distance
                horizontally
                - Checking they are roughly parallel
        """
        
    def check_curvature(self, previous, current):
        """
            Checks if percentage error of previous and current radius of lane
            curvature is > 0.6, if true, then return true, so the "sliding windows
            search" is applied instead of "search from prior".
        """
        percent_error = (abs(current - previous))/previous
        # If percent error is > 60%, then current lane curvature isn't similar,
        # so redo sliding window search
        if percent_error > 0.6:
            redo_sliding_window_search = True # Re-apply sliding window search 
        else:
            redo_sliding_window_search = False # Continue to apply search from prior