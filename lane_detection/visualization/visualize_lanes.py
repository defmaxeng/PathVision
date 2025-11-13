from sympy import symbols, Eq, solve
import numpy as np
import cv2
import math

class Lane_Visual:

    
    def __init__(self, model_output, resolution, img, switch_direction=None):
        model_output = model_output.squeeze(0).detach().cpu().numpy() # Now in Shape (4, 48)
        self.model_output = model_output[[2, 0, 1, 3]] 
        self.img_width, self.img_height = resolution
        self.h_samples = self.generate_h_samples()
        self.markers = self.locate_markers() # lane_markers as list of points
        self.left_lane, self.center_lane, self.right_lane = self.locate_lanes() # lanes as lists of points
        self.img = img
        self.switch_direction = switch_direction
        if switch_direction is not None:
            self.lane_switch = self.switch_lanes(switch_direction) # lane_merge as list of points


    
    def locate_markers(self):
        # want to return 1 array of 4 arrays of points going over the lanes from left to right
        self.image_cutoffs = [self.find_image_cutoff(model_output_row) for model_output_row in self.model_output]
        regressions = []

        for row_index in range(len(self.model_output)):
            # Make sideways quadratic regressions
            y_values = self.h_samples[:self.image_cutoffs[row_index]]
            x_values = self.model_output[row_index][:self.image_cutoffs[row_index]]
            regressions.append((np.polyfit(y_values, x_values, 2)))


        # Use Middle Two lanes to find the vanishing point because they always exist
        self.vanishing_point = self.find_vanishing_point(regressions[1], regressions[2])
        
        
        
        all_pts = []
        ys = np.arange(self.vanishing_point, self.img_height) # ys stands for y smooth because after converting to regression and back, the lines are smoother
        for regression in regressions:
            a, b, c = regression

            # Convert each regression into a list of points
            # height goes from top to bottom (will later be limited to from vanishing point to out of bounds)
            xs = a * ys**2 + b * ys + c # xs stands for x smooth because after converting to regression and back to x values, the lines are smoother
            pts = np.stack([xs, ys], axis=1)
            all_pts.append(pts)
        return all_pts

    def locate_lanes(self):
        left_lane   = (self.model_output[0] + self.model_output[1]) / 2
        middle_lane = (self.model_output[1] + self.model_output[2]) / 2
        right_lane  = (self.model_output[2] + self.model_output[3]) / 2
        lanes = [left_lane, middle_lane, right_lane]

        lane_cutoffs = []
        lane_cutoffs.append(min(self.image_cutoffs[0], self.image_cutoffs[1])) # left
        lane_cutoffs.append(min(self.image_cutoffs[1], self.image_cutoffs[2])) # middle
        lane_cutoffs.append(min(self.image_cutoffs[2], self.image_cutoffs[3])) # right
        all_pts = []
        self.lane_regressions = []
        ys = np.arange(self.vanishing_point, self.img_height) # ys stands for y smooth because after converting to regression and back, the lines are smoother        
        for row_index in range(len(lanes)):
            # Make sideways quadratic regressions
            y_values = self.h_samples[:lane_cutoffs[row_index]]
            x_values = lanes[row_index][:lane_cutoffs[row_index]]
            a, b, c = np.polyfit(y_values, x_values, 2)
            self.lane_regressions.append((a, b, c))
            # Convert each regression into a list of points
            # height goes from top to bottom (will later be limited to from vanishing point to out of bounds)
            xs = a * ys**2 + b * ys + c # xs stands for x smooth because after converting to regression and back to x values, the lines are smoother
            pts = np.stack([xs, ys], axis=1)
            all_pts.append(pts)

        return all_pts

    def switch_lanes(self, switch_direction):
        # Original Lane
        a1, b1, c1 = self.lane_regressions[1]
        a2, b2, c2 = self.lane_regressions[0] if switch_direction == 'left' else self.lane_regressions[2]
        merge_low_bound = int(self.img_height * 0.75)
        merge_high_bound = int(self.img_height * 0.5)
        merge_range = merge_low_bound - merge_high_bound


        x_vals = []
        y_vals = [y for y in range(merge_high_bound, merge_low_bound)]

                
        # Generate x vals from y vals
        # Start building lane_switch_function. Using a cos wave. The (x_val, y_val) ordered pair renders top to bottom. 
        # So merging left would be a negative cosign function
        # must find amplitude and periodicity of the wave

        wave_period = math.pi / float(merge_range)

        for y_val in y_vals:
            x1 = a1*y_val**2 + b1*y_val + c1
            x2 = a2*y_val**2 + b2*y_val + c2
            xAvg = int((x1 + x2) / 2)
            
            sign = 1 if x2 > x1 else -1
            amplitude = abs(xAvg-x1)

            x_val = sign * amplitude * math.cos(wave_period * y_val) + xAvg

            x_vals.append(x_val)

        # Convert to Numpy to stack arrays
        x_vals = np.asarray(x_vals)
        y_vals = np.asarray(y_vals)
        
        # Stack the points
        
        return np.stack([x_vals, y_vals], axis=1)

    # Helper functions
    def generate_h_samples(self):
        ratio_lower_end = 1280/240
        ratio_higher_end = 1280/710
        lower_end = float(self.img_width)/ratio_lower_end
        higher_end = float(self.img_width)/ratio_higher_end
        return np.linspace(lower_end, higher_end, 48)

    def find_image_cutoff(self, lane_output):
        # Takes in a single lane's x values (model_column) and its y values (h_samples)
        # Uses a sliding window technique to see where the model starts printing trash (lane goes off the screen))
        # going from vanishing height to bottom, if the x values start moving in the opposite direction to the lane then that is where the data becomes trash
        # vanishing height is the height at which the lanes seemingly "vanish" into the horizon
        
        
        # if a lane is overwhelmingly given positive slopes (its a left lane) then it randomly starts giving negative, that is where the model turns gibberish
        positive_slopes = 0
        negative_slopes = 0
        incorrect_slopes = 0
        for prediction in range(len(lane_output)-4):
            a, b = np.polyfit(lane_output[prediction:prediction+4], self.h_samples[prediction:prediction+4], 1)
            if a > 0:
                positive_slopes += 1
                if negative_slopes > positive_slopes:
                    incorrect_slopes += 1
            else:
                negative_slopes += 1
                if positive_slopes > negative_slopes:
                    incorrect_slopes += 1
            
            if incorrect_slopes > 4:
                return prediction-4
        return 48

    def find_vanishing_point(self, r1, r2):
        a_zero, b_zero, c_zero = r1
        a_one, b_one, c_one = r2

        x, y = symbols('x y', real=True)
        
        eq1 = Eq(x, a_zero*y**2 + b_zero*y + c_zero)
        eq2 = Eq(x, a_one*y**2 + b_one*y + c_one)

        solutions = solve((eq1, eq2), (x, y))
        
        for solution in solutions:
            if solution[0] < self.img_width and solution[0] < self.img_height:
                return solution[1] +20
            
            elif solution[1] < self.img_width and solution[1] < self.img_height:
                return solution[1] + 20
        
        return int(self.img_height * 2 / 5)

    def get_image(
            self,
            markers=False, 
            center_lane=False, 
            left_lane=False, 
            right_lane=False, 
            lane_merge=False,
            visualize=False
            ):
        
        if markers: 
            for marker in self.markers: 
                marker = np.array(marker, dtype=np.float64)
                marker = np.round(marker).astype(np.int32).reshape(-1, 1, 2)

                cv2.polylines(self.img, [marker], isClosed=False, color=(203, 192, 255),
                        thickness=1, lineType=cv2.LINE_AA)
                
        
        if center_lane:
            self.center_lane = np.array(self.center_lane, dtype=np.float64)
            self.center_lane = np.round(self.center_lane).astype(np.int32).reshape(-1, 1, 2)

            cv2.polylines(self.img, [self.center_lane], isClosed=False, color=(255, 0, 0),
                        thickness=1, lineType=cv2.LINE_AA)
            
            
        if left_lane:
            self.left_lane = np.array(self.left_lane, dtype=np.float64)
            self.left_lane = np.round(self.left_lane).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(self.img, [self.left_lane], isClosed=False, color=(255, 0, 0),
                        thickness=1, lineType=cv2.LINE_AA)
            
        if right_lane:
            self.right_lane = np.array(self.right_lane, dtype=np.float64)
            self.right_lane = np.round(self.right_lane).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(self.img, [self.right_lane], isClosed=False, color=(255, 0, 0),
                        thickness=1, lineType=cv2.LINE_AA)
            
        if lane_merge:
            if self.switch_direction is not None:
                self.lane_switch = np.array(self.lane_switch, dtype=np.float64)
                self.lane_switch = np.round(self.lane_switch).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(self.img, [self.lane_switch], isClosed=False, color=(30, 180, 0),
                        thickness=1, lineType=cv2.LINE_AA)
        
            else:
                cv2.polylines(self.img, [self.center_lane], isClosed=False, color=(30, 180, 0),
                        thickness=1, lineType=cv2.LINE_AA)
            
        if visualize:
            cv2.imshow("Overlayed Image", self.img)
            cv2.waitKey(0)
        
        return self.img
        
        















    # #getters
    # def get_markers(self):
    #     pass

    # def get_center_lane(self):
    #     pass

    # def get_right_lane(self):
    #     pass

    # def get_left_lane(self):
    #     pass

    # def get_lane_switch(self):
    #     pass



