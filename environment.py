import random
import numpy as np
import gymnasium as gym
from scipy.ndimage import gaussian_filter, maximum_filter, label, center_of_mass
import yaml
from gymnasium import spaces

""" 
    This class represents a simplified simulation of the pSCT telescope
    for use in RL development.
"""
class pSCT_environment(gym.Env):

    def __init__(self,
                 seed=67,
                 resp_file="/Users/evanhilton/Desktop/VSCoding/RLAlignIndividual/pSCT_spec/P1_matrix.yml",

                 P1s = [1111, 1112, 1113, 1114, 1211, 1212, 1213, 1214, 1311, 1312, 1313, 1314, 1411, 1412, 1413, 1414],
                 n_panels=1,
                
                 action_clip=1.0,
                 action_scale=0.05,
                 delta_rxry_limit=0.1,

                 w_improve=0.2,
                 w_dist=5e-3,
                 w_action=5e3,
                 
                 max_steps=1024,
                 
                 init_scatter_pix=500.0,
                 init_rxry_scale=0.05,
                 
                 center=np.array([1612.2804, 1024.4423]),
                 img_size=128,
                 bg_level=0.02,
                 read_noise=0.01,
                 amplitude=1.0,

                 success_radius_pix=3.5,
                 det_smooth_sigma=1.2,
                 det_thresh_sigma=8.0,
                 det_max_peaks=64,
                 det_merge_radius_pix=2.0,
                 img_fov_pix=600.0):

        # useful
        self.rng = np.random.RandomState(seed)
        self.resp_file = resp_file

        # panel information
        self.P1s = P1s # a list of all panels
        self.n_panels = n_panels
        self.panels = random.sample(self.P1s, self.n_panels) # choose n_panels amount of random panels from the list

        # action information
        self.action_clip = action_clip
        self.action_scale = action_scale
        self.delta_rxry_limit = delta_rxry_limit
        self.rxry = None
        self.prev_action_mags = None
        self.current_panel_index = int(self.rng.randint(0, len(self.panels)))                # the panel that the agent controls is self.panels[self.current_panel_index]

        # reward shaping
        self.prev_cost = None
        self.w_improve = w_improve
        self.w_action = w_action
        self.w_dist = w_dist

        # state
        self.step_count = 0
        self.max_steps = max_steps
        self.base_offsets = None

        # centroid creation
        self.init_scatter_pix = init_scatter_pix
        self.init_rxry_scale = init_rxry_scale
        self.M_RxRy_inv = self.load_all_rx_ry_matrices(respfile=resp_file)
        self.true_centroids = None
        self.last_detected_fp = None

        # image creation
        self.center = center
        self.img_size = img_size
        self.bg_level = bg_level
        self.read_noise = read_noise
        self.amplitude = amplitude
        self.Y, self.X = np.mgrid[0:img_size, 0:img_size]

        # image detection
        self.success_radius_pix = success_radius_pix
        self.det_smooth_sigma = det_smooth_sigma
        self.det_thresh_sigma = det_thresh_sigma
        self.det_max_peaks = det_max_peaks
        self.det_merge_radius_pix = det_merge_radius_pix
        self.img_fov_pix = img_fov_pix
        self.centroids_at_center = 0

        # action / observation
        # Observation: single-channel image
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1, self.img_size, self.img_size),  # CHW for SB3 CNN
            dtype=np.float32,
        )

        # Action: (n_panels, 2) flattened
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            #shape=(2 * self.n_panels,),
            shape=(2,),                                 # we give the agent two buttons. Control changes depending on a situation
            dtype=np.float32,
        )
    # =================================== API ===================================
    """
    Please note that render() and close() are not declared. They are optionally a part of the API.

    render() is not defined because the observation returned by step() and reset()
    are inherently visual (an image). Any rendering or visualization that a user
    might want should be done in a seperate class. (It is done in this 
    project in visualize.py and visualizeDebug.py)
    
    close() is not defined because there is nothing to close. close() would need
    to be defined if we were doing any threading, using the internet, or using
    some other data gathering that requires having a connection.
    """
    
    """
        -Run one timestep of the environment's dynamics using the agent actions.
        For each panel in the simulation, the agent chooses to update its position
        by providing a rotation rx, ry. 
        -Takes each rotation and updates the location of the panel's corresponding 
        image and renders it.
        -Calculates and returns the reward corresponding with the provided action
        
        params:
        action (numpy.ndarray with shape (1, 2)):        panel rotations. the nth panel is rotated rx, ry = action[n]

        returns:
        observation (numpy.ndarray with shape (img_size, img_size)):    the new environment state
        reward (Float):                                                 how beneficial the action was
        terminated (bool):                                              whether the agent reaches the terminal state
        truncated (bool):                                               whether the agent reaches a state that should cause the simulation to stop early
        info (dict):                                                    contains debugging information
    """
    def step(self, action):
        # action shaping
        action = np.asarray(action, dtype=float) # makes sure action is the right type
        if action.ndim == 1:
            action = action.reshape(1, 2)
        assert action.shape == (1, 2)
        action = np.clip(action, -self.action_clip, self.action_clip)

        # update mirror positions
        delta = self.action_scale * action
        delta = np.clip(delta, -self.delta_rxry_limit, self.delta_rxry_limit)
        self.rx_ry[self.current_panel_index] = self.rx_ry[self.current_panel_index] + delta       # the nth panel rotations gets updated

        # create observation & detect centroids
        self._compute_true_centroids()
        observation = self._render_image()
        det = self._detected_fp_coords(observation)
        # num_at_center = self._get_num_at_center(det)
        # if(num_at_center > self.centroids_at_center):
        #     self.centroids_at_center = num_at_center    # raise detected centroids at center if there are more.
        #     self.current_panel_index = (self.current_panel_index + 1) % self.n_panels # make sure no index out of bounds

        # compute cost & reward shaping
        cost = self._cost_from_detected(det, delta)
        reward = -cost
        improve = self.prev_cost - cost
        reward += 0.5 * improve

        if self._success(det):
            reward += 100
        reward -= 0.1 # time penalty. incentivices fast solutions
        self.prev_cost = cost

        # episode tracking
        self.step_count += 1
        terminated = self._success(det)
        truncated = self._truncated()

        # info
        info = {}

        return observation.astype(np.float32)[None, :, :], reward, terminated, truncated, info

    """
        Resets the environment to an initial internal state, returning an initial observation and info.
        Places each panel's image to a new random location.

        params:
        seed (None):                Satisfies the API but we introduce our own PRNG (numpy random number generator) so keep as None
        options (optional dict):    optional debug information to include about how the environment is reset. RNG seed for example.

        returns:
        observation (numpy.ndarray with shape (n_panels, 2)):   the new environment state
        info (dict):                                            contains debugging information. Should be the same information as returned from step()
    """
    def reset(self, *, seed=None, options=None):
        self.step_count = 0

        # choose new random panels
        self.panels = random.sample(self.P1s, self.n_panels)

        # inital panel positions
        self.base_offsets = (self.rng.rand(self.n_panels, 2) - 0.5) * self.init_scatter_pix
        self.rx_ry = (self.rng.rand(self.n_panels, 2) - 0.5) * self.init_rxry_scale

        # create observation & detect centroids
        self._compute_true_centroids()
        observation = self._render_image()
        det = self._detected_fp_coords(observation)

        # initial cost for improvement shaping
        cost = self._cost_from_detected(det, np.zeros((self.n_panels, 2)))
        self.prev_cost = cost

        # choose new random starting panel
        self.current_panel_index = int(self.rng.randint(0, len(self.panels)))

        # info
        info = {}

        return observation.astype(np.float32)[None, :, :], info
    
    # ============================== Helper Functions ==============================

    """
        returns the cost of the given time step. high cost means the action was bad,
        low cost means the action was good.
    """
    def _cost_from_detected(self, pts_fp, action_applied):
        # distance term
        d = pts_fp - self.center[None, :]
        mean_r2 = float(np.mean(np.sqrt(np.sum(d**2, axis=1))))

        # success bonus
        # successBonus = 0
        # successBonus = -self.centroids_at_center * 5
        # if self._gaussian_outside_image():
        #     successBonus = 5
        # successBonus = 0
        # if self._success(pts_fp):
        #     successBonus = -10
        # if self._gaussian_outside_image():
        #     successBonus = 10
        
        cost = self.w_dist * mean_r2
        return cost
    
    """
        returns the number of centroids at the center of the screen. two centroids at the center
        of the screen registers as one centroid at the center of the screen, so this function
        also counts how many detected centroids exist on the screen and takes it into account for the count.
    """
    def _get_num_at_center(self, pts_fp):
        d = pts_fp - self.center[None, :]
        r = np.sqrt(np.sum(d**2, axis=1))
        r_sorted = np.sort(r)[: self.n_panels]
        detected_at_center = (r_sorted <= self.success_radius_pix).sum()
        return detected_at_center + (self.n_panels - len(pts_fp)) if detected_at_center > 0 else 0

    """
        Uses the detected focal plane coordinates of the gaussians to determine if the
        image is fully aligned. This happens when all detected points are within a certain
        radius of the center of the image.
    """
    def _success(self, pts_fp):
        d = pts_fp - self.center[None, :]
        r = np.sqrt(np.sum(d**2, axis=1))
        # success if the CLOSEST n_panels detections are all within radius
        r_sorted = np.sort(r)[: self.n_panels]
        return not bool(np.any(r_sorted > self.success_radius_pix))

    """
        An episode ends if the current step count is larger than the permitted episode length.
        An episode ends if any of the centroids leave the screen.
    """
    def _truncated(self):
        return self._gaussian_outside_image() or self.step_count >= self.max_steps
    
    """
        Returns true if any of the true gaussian positions leave the image, false otherwise.
    """
    def _gaussian_outside_image(self):
        for (fx, fy) in self.true_centroids:
            if ((np.abs(fx - self.center[0]) > self.init_scatter_pix / 2) or (np.abs(fy - self.center[1]) > self.init_scatter_pix / 2)):
                return True
        return False

    """
        Computes the true positions of each image created by each panel.
        Updates self.true_centroids to represent the new centroid positions (if panels have been moved).
        Everything is computed in focal plane coordinates
    """
    def _compute_true_centroids(self):
        centroids = np.zeros((self.n_panels, 2), float)
        for i, panel in enumerate(self.panels):
            rx, ry = self.rx_ry[i]
            dx, dy = self.calc_dx_dy(rx, ry, self.M_RxRy_inv[panel]) # focal plane coords
            base_xy = self.center + self.base_offsets[i]
            centroids[i] = base_xy + np.array([dx, dy])
        self.true_centroids = centroids

    """
        Uses the focal plane coordinates of the true centroids to create an image in
        pixel coordinate space (as would be seen by a camera during alignment). The
        returned image is a numpy float array with shape (self.img_size, self.img_size)
    """
    def _render_image(self):
        img = np.zeros((self.img_size, self.img_size), float)
        params = np.zeros((self.n_panels, 6), float)
        for i, (x_fp, y_fp) in enumerate(self.true_centroids):
            u, v = self._fp_to_uv(x_fp, y_fp)
            centerX, centerY = self._fp_to_uv(self.center[0], self.center[1])
            r0 = np.sqrt((u-centerX)**2+(v-centerY)**2)
            params[i][0] = u
            params[i][1] = v
            params[i][2] = self.amplitude
            params[i][3] = (np.tanh(0.17 * r0 - 3)) + 2
            params[i][4] = 0.006 * r0 + 1.00495
            params[i][5] = np.arctan((v - centerY) / (u - centerX)) if (u - centerX) != 0 else np.pi / 2
            
        img = self.add_gaussians_batch(img, params)

        # detector noise and background
        img += self.bg_level
        img += self.rng.normal(scale=self.read_noise, size=img.shape)

        # normalize to [0,1] (good for PPO)
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        
        return img
    
    """
        loops over each gaussian to be added to the image and adds it. This uses the optimized
        helper method add_gaussian() which adds a gaussian to the image.
    """
    def add_gaussians_batch(self, img, params):
        for g in params:
            x0, y0, A, s_r, s_t, p = g
            img = self.add_gaussian(img, x0, y0, A, s_r, s_t, p)
        return img
    
    """
        For optimization purposes, we only consider a small box around where
        we generate the gaussian. Values far away from the gaussian are basically
        zero, so theres no point wasting computation time calculating multiple exponentials
        that will inevitably be ~10^<-5
    """
    def add_gaussian(self, img, x0, y0, A, s_r, s_t, p):
        H, W = img.shape

        # cutoff
        k = 3.5
        R = int(np.ceil(k * max(s_r, s_t)))

        x_min = max(0, int(x0 - R))
        x_max = min(W, int(x0 + R + 1))
        y_min = max(0, int(y0 - R))
        y_max = min(H, int(y0 + R + 1))

        # local grid
        xs = np.arange(x_min, x_max)
        ys = np.arange(y_min, y_max)
        X, Y = np.meshgrid(xs, ys)

        # shift
        c = X - x0
        d = Y - y0

        # rotation
        cp = np.cos(p)
        sp = np.sin(p)

        a = c * cp + d * sp
        b = -c * sp + d * cp

        G = A * np.exp(
            -0.5 * ((a / s_r) ** 2 + (b / s_t) ** 2)
        )

        img[y_min:y_max, x_min:x_max] += G

        return img

    """
        Detects centroids from the given image and converts the detected points
        into focal plane coordinates. If there are no detected points, it returns
        an empty numpy array which still maintains the correct column number.
    """
    def _detected_fp_coords(self, img):
        pts_uv = self._detect_centroids_uv(img)
        if len(pts_uv) == 0:
            self.last_detected_fp = np.zeros((0, 2), float)
            return self.last_detected_fp

        x_fp, y_fp = self._uv_to_fp(pts_uv[:, 0], pts_uv[:, 1])
        pts_fp = np.vstack([x_fp, y_fp]).T
        self.last_detected_fp = pts_fp
        return pts_fp

    """
        given the image, this function finds the centroid of each gaussian and
        stores and returns the pixel coordinate of each. 
    """
    def _detect_centroids_uv(self, img):
        """
        Simple detector:
          1) smooth
          2) threshold (robust sigma)
          3) local maxima
          4) connected-component COM to merge plateaus/blobs
        Returns centroids in image pixel coords (u,v).
        """
        sm = gaussian_filter(img, self.det_smooth_sigma)

        # robust background/sigma from median + MAD
        med = np.median(sm)
        mad = np.median(np.abs(sm - med))
        sigma = 1.4826 * mad if mad > 0 else np.std(sm) + 1e-9

        thr = med + self.det_thresh_sigma * sigma
        mask = sm > thr

        if not np.any(mask):
            return np.zeros((0, 2), float)

        # local maxima among a 3x3 neighborhood
        mx = (sm == maximum_filter(sm, size=3)) & mask

        # label maxima regions (plateaus)
        lab, nlab = label(mx)
        if nlab == 0:
            return np.zeros((0, 2), float)

        com = center_of_mass(sm, lab, np.arange(1, nlab + 1))
        # com is list of (v,u) because array indexing is (row,col)
        pts = np.array([(u, v) for (v, u) in com], float)

        # keep strongest peaks if too many
        if pts.shape[0] > self.det_max_peaks:
            # score by sm at nearest integer pixel
            ui = np.clip(np.round(pts[:, 0]).astype(int), 0, self.img_size - 1)
            vi = np.clip(np.round(pts[:, 1]).astype(int), 0, self.img_size - 1)
            score = sm[vi, ui]
            keep = np.argsort(score)[-self.det_max_peaks:]
            pts = pts[keep]

        # optional merge close detections
        if pts.shape[0] >= 2 and self.det_merge_radius_pix > 0:
            pts = self._merge_close_points(pts, self.det_merge_radius_pix)
        
        return pts
    
    """
        used as a helper function for _detect_centroids_uv(). If two detected peaks
        are too close together, they are merged into one detected peak.
    """
    def _merge_close_points(self, pts, r):
        keep = []
        used = np.zeros(len(pts), dtype=bool)
        for i in range(len(pts)):
            if used[i]:
                continue
            d = np.sqrt(np.sum((pts - pts[i]) ** 2, axis=1))
            grp = np.where(d <= r)[0]
            used[grp] = True
            keep.append(np.mean(pts[grp], axis=0))
        return np.array(keep, float)

    """
        converts given focal plane coordinates to pixel coordinates
    """
    def _fp_to_uv(self, x_fp, y_fp):
        """focal-plane pixels -> image pixels"""
        half = self.img_fov_pix / 2.0
        dx = x_fp - self.center[0]
        dy = y_fp - self.center[1]
        u = (dx + half) / (2 * half) * (self.img_size - 1)
        v = (dy + half) / (2 * half) * (self.img_size - 1)
        return u, v

    """
        converts given pixel coordinates to focal plane coordinates
    """
    def _uv_to_fp(self, u, v):
        """image pixels -> focal-plane pixels"""
        half = self.img_fov_pix / 2.0
        dx = (u / (self.img_size - 1)) * (2 * half) - half
        dy = (v / (self.img_size - 1)) * (2 * half) - half
        x_fp = self.center[0] + dx
        y_fp = self.center[1] + dy
        return x_fp, y_fp

    """
        Uses the response matrix M_RxRy_inv to convert rotation coordinates to focal plane coordinates.
        The response matrix M_RxRy_inv should be given as the real response matrix obtained
        experimentally on a real telescope.
    """
    def calc_dx_dy(self, rx, ry, M_RxRy_inv):
        M_RxRy = np.linalg.inv(M_RxRy_inv)
        dx, dy = np.matmul(M_RxRy, np.array([rx, ry]))
        return dx, dy
    
    """
        Helper method to load a file which hopefully contains all the response matrices
        for each panel.
    """
    def load_all_rx_ry_matrices(self, respfile, verbose=False):
        with open(respfile) as f:
            respM_yaml = yaml.safe_load(f)
        return respM_yaml