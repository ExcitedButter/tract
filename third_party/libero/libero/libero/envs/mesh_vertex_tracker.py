"""
Mesh Vertex Tracker for specific vertex IDs in LIBERO Environments
This wrapper tracks specific vertices by their global IDs and projects them to camera views.
"""

import numpy as np
import gym
from gym import spaces


class MeshVertexTracker(gym.Wrapper):
    """
    Wrapper that tracks specific mesh vertices by their IDs.
    Returns both 3D positions and 2D projections for specified camera views.
    """
    
    def __init__(self, env, 
                 agentview_vertex_ids=None, 
                 eye_in_hand_vertex_ids=None,
                 use_full_vertices=False,
                 max_vertices_per_mesh=50,
                 img_height=128,
                 img_width=128):
        """
        Args:
            env: Base environment
            agentview_vertex_ids: List of vertex IDs to track for agentview camera
            eye_in_hand_vertex_ids: List of vertex IDs to track for eye-in-hand camera
            use_full_vertices: If True, extract all vertices without subsampling
            max_vertices_per_mesh: Maximum vertices per mesh when subsampling (default 50)
            img_height: Height of camera images for projection (default 128)
            img_width: Width of camera images for projection (default 128)
        """
        super().__init__(env)
        
        # Store vertex IDs to track for each camera
        self.agentview_vertex_ids = agentview_vertex_ids or []
        self.eye_in_hand_vertex_ids = eye_in_hand_vertex_ids or []
        
        # Subsampling configuration
        self.use_full_vertices = use_full_vertices
        self.max_vertices_per_mesh = max_vertices_per_mesh
        
        # Image dimensions for projection
        self.img_height = img_height
        self.img_width = img_width
        
        # Build vertex ID to mesh mapping on first reset
        self.vertex_mapping = None
        self.all_vertices_3d = None
        
        # Modify observation space if it exists
        if hasattr(env, 'observation_space'):
            self.observation_space = self._modify_obs_space(env.observation_space)
        
    def _modify_obs_space(self, obs_space):
        """Add tracked vertex spaces to observation space."""
        if isinstance(obs_space, spaces.Dict):
            new_spaces = dict(obs_space.spaces)
            
            # Maximum number of points we might track
            max_agentview = len(self.agentview_vertex_ids)
            max_eye_in_hand = len(self.eye_in_hand_vertex_ids)
            
            # Add spaces for tracked vertices
            if max_agentview > 0:
                new_spaces['agentview_tracked_vertices_3d'] = spaces.Box(
                    low=-10.0, high=10.0, shape=(max_agentview, 3), dtype=np.float32
                )
                new_spaces['agentview_tracked_vertices_2d'] = spaces.Box(
                    low=0, high=1000, shape=(max_agentview, 2), dtype=np.float32
                )
                new_spaces['agentview_tracked_vertex_ids'] = spaces.Box(
                    low=0, high=100000, shape=(max_agentview,), dtype=np.int32
                )
                
            if max_eye_in_hand > 0:
                new_spaces['eye_in_hand_tracked_vertices_3d'] = spaces.Box(
                    low=-10.0, high=10.0, shape=(max_eye_in_hand, 3), dtype=np.float32
                )
                new_spaces['eye_in_hand_tracked_vertices_2d'] = spaces.Box(
                    low=0, high=1000, shape=(max_eye_in_hand, 2), dtype=np.float32
                )
                new_spaces['eye_in_hand_tracked_vertex_ids'] = spaces.Box(
                    low=0, high=100000, shape=(max_eye_in_hand,), dtype=np.int32
                )
                
            return spaces.Dict(new_spaces)
        return obs_space
    
    def reset(self, **kwargs):
        """Reset environment and add tracked vertices to observation."""
        obs = self.env.reset(**kwargs)
        
        # Build vertex mapping on first reset
        if self.vertex_mapping is None:
            self._build_vertex_mapping()
        
        # Add tracked vertices for each camera
        if self.agentview_vertex_ids:
            tracked_data = self._get_tracked_vertices('agentview', self.agentview_vertex_ids)
            obs['agentview_tracked_vertices_3d'] = tracked_data['vertices_3d']
            obs['agentview_tracked_vertices_2d'] = tracked_data['vertices_2d']
            obs['agentview_tracked_vertex_ids'] = tracked_data['vertex_ids']
            
        if self.eye_in_hand_vertex_ids:
            tracked_data = self._get_tracked_vertices('robot0_eye_in_hand', self.eye_in_hand_vertex_ids)
            obs['eye_in_hand_tracked_vertices_3d'] = tracked_data['vertices_3d']
            obs['eye_in_hand_tracked_vertices_2d'] = tracked_data['vertices_2d']
            obs['eye_in_hand_tracked_vertex_ids'] = tracked_data['vertex_ids']
        
        return obs
    
    def _world_vertex_at_id(self, vertex_id: int):
        """World position for one mapped vertex (O(1); avoids updating all vertices each step)."""
        if self.vertex_mapping is None or vertex_id not in self.vertex_mapping:
            return None
        env = self._get_base_env()
        sim = env.sim
        vertex_info = self.vertex_mapping[vertex_id]
        geom_id = vertex_info["geom_id"]
        local_vertex = vertex_info["local_vertex"]
        geom_pos = sim.data.geom_xpos[geom_id].copy()
        geom_mat = sim.data.geom_xmat[geom_id].reshape(3, 3).copy()
        return (local_vertex @ geom_mat.T) + geom_pos

    def step(self, action):
        """Step environment and add tracked vertices to observation."""
        obs, reward, done, info = self.env.step(action)

        # Add tracked vertices for each camera
        if self.agentview_vertex_ids:
            tracked_data = self._get_tracked_vertices('agentview', self.agentview_vertex_ids)
            obs['agentview_tracked_vertices_3d'] = tracked_data['vertices_3d']
            obs['agentview_tracked_vertices_2d'] = tracked_data['vertices_2d']
            obs['agentview_tracked_vertex_ids'] = tracked_data['vertex_ids']
            
        if self.eye_in_hand_vertex_ids:
            tracked_data = self._get_tracked_vertices('robot0_eye_in_hand', self.eye_in_hand_vertex_ids)
            obs['eye_in_hand_tracked_vertices_3d'] = tracked_data['vertices_3d']
            obs['eye_in_hand_tracked_vertices_2d'] = tracked_data['vertices_2d']
            obs['eye_in_hand_tracked_vertex_ids'] = tracked_data['vertex_ids']
        
        return obs, reward, done, info
    
    def _build_vertex_mapping(self):
        """Build mapping from vertex IDs to mesh vertices."""
        env = self._get_base_env()
        sim = env.sim
        
        # Store all vertices with their global IDs
        self.vertex_mapping = {}
        self.all_vertices_3d = []
        
        global_vertex_id = 0
        
        # Extract all mesh vertices
        for geom_id in range(sim.model.ngeom):
            geom_type = sim.model.geom_type[geom_id]
            
            if geom_type == 7:  # mjGEOM_MESH
                mesh_id = sim.model.geom_dataid[geom_id]
                
                # Get vertex range for this mesh
                vert_start = sim.model.mesh_vertadr[mesh_id]
                if mesh_id < sim.model.nmesh - 1:
                    vert_end = sim.model.mesh_vertadr[mesh_id + 1]
                else:
                    vert_end = len(sim.model.mesh_vert)
                
                # Store local vertices for this mesh
                local_vertices = sim.model.mesh_vert[vert_start:vert_end].reshape(-1, 3)
                
                # Sample vertices based on configuration
                if not self.use_full_vertices and len(local_vertices) > self.max_vertices_per_mesh:
                    indices = np.linspace(0, len(local_vertices)-1, self.max_vertices_per_mesh, dtype=int)
                    local_vertices = local_vertices[indices]
                
                # Store mapping for each vertex
                for vertex in local_vertices:
                    self.vertex_mapping[global_vertex_id] = {
                        'geom_id': geom_id,
                        'local_vertex': vertex.copy()
                    }
                    global_vertex_id += 1
            else:
                # Non-mesh geometry - store center as single vertex
                self.vertex_mapping[global_vertex_id] = {
                    'geom_id': geom_id,
                    'local_vertex': np.zeros(3)  # Center point
                }
                global_vertex_id += 1
        
        pass  # Vertex mapping built successfully
    
    def _update_vertex_positions(self):
        """Update current 3D positions of all vertices."""
        env = self._get_base_env()
        sim = env.sim
        
        self.all_vertices_3d = []
        
        for vertex_id, vertex_info in self.vertex_mapping.items():
            geom_id = vertex_info['geom_id']
            local_vertex = vertex_info['local_vertex']
            
            # Get current transformation
            geom_pos = sim.data.geom_xpos[geom_id].copy()
            geom_mat = sim.data.geom_xmat[geom_id].reshape(3, 3).copy()
            
            # Transform to world coordinates
            world_vertex = (local_vertex @ geom_mat.T) + geom_pos
            self.all_vertices_3d.append(world_vertex)
        
        self.all_vertices_3d = np.array(self.all_vertices_3d)
    
    def _get_tracked_vertices(self, camera_name, vertex_ids):
        """Get 3D positions and 2D projections for tracked vertices."""
        env = self._get_base_env()
        sim = env.sim
        
        # Collect vertices for the specified IDs (only tracked IDs, not full scene)
        vertices_3d = []
        valid_ids = []

        for vid in vertex_ids:
            world = self._world_vertex_at_id(vid)
            if world is not None:
                vertices_3d.append(world)
                valid_ids.append(vid)
        
        if len(vertices_3d) == 0:
            # Return empty arrays if no valid vertices
            return {
                'vertices_3d': np.zeros((0, 3), dtype=np.float32),
                'vertices_2d': np.zeros((0, 2), dtype=np.float32),
                'vertex_ids': np.array([], dtype=np.int32)
            }
        
        vertices_3d = np.array(vertices_3d)
        
        # Project to 2D for the specified camera
        vertices_2d = self._project_to_camera(vertices_3d, camera_name)
        
        return {
            'vertices_3d': vertices_3d.astype(np.float32),
            'vertices_2d': vertices_2d.astype(np.float32),
            'vertex_ids': np.array(valid_ids, dtype=np.int32)
        }
    
    def _project_to_camera(self, vertices_3d, camera_name):
        """Project 3D vertices to 2D camera coordinates."""
        env = self._get_base_env()
        sim = env.sim
        
        # Get camera parameters
        camera_id = sim.model.camera_name2id(camera_name)
        fovy = sim.model.cam_fovy[camera_id]
        cam_pos = sim.data.cam_xpos[camera_id].copy()
        cam_mat = sim.data.cam_xmat[camera_id].reshape(3, 3).copy()
        
        # Use instance image dimensions
        img_height = self.img_height
        img_width = self.img_width
        
        # Convert fovy to focal length
        fovy_rad = np.deg2rad(fovy)
        f = img_height / (2 * np.tan(fovy_rad / 2))
        
        # Transform to camera coordinates
        points_cam = vertices_3d - cam_pos
        points_cam = points_cam @ cam_mat
        points_cam[:, 2] = -points_cam[:, 2]  # Flip Z axis
        
        # Project to 2D
        valid = points_cam[:, 2] > 0.01  # Points in front of camera
        
        points_2d = np.zeros((len(vertices_3d), 2))
        if np.any(valid):
            x_proj = (points_cam[valid, 0] * f) / points_cam[valid, 2] + img_width / 2
            y_proj = (points_cam[valid, 1] * f) / points_cam[valid, 2] + img_height / 2
            y_proj = img_height - y_proj  # Flip Y axis
            
            points_2d[valid, 0] = x_proj
            points_2d[valid, 1] = y_proj
        
        return points_2d
    
    def _get_grid_points(self, camera_name):
        """Generate a grid of points in the camera view."""
        # Generate grid in the actual image space
        img_size = self.img_height  # Use configured height
        
        grid_x = np.linspace(20, img_size-20, self.grid_size)
        grid_y = np.linspace(20, img_size-20, self.grid_size)
        
        grid_2d = []
        for y in grid_y:
            for x in grid_x:
                grid_2d.append([x, y])
        grid_2d = np.array(grid_2d)
        
        # For 3D, we'd need to unproject or sample from scene
        # For now, return dummy 3D positions
        grid_3d = np.zeros((len(grid_2d), 3))
        
        return grid_3d, grid_2d
    
    def _get_base_env(self):
        """Get the base environment with sim access."""
        env = self.env
        
        # Navigate through wrappers to find base env with sim
        while hasattr(env, 'env') and not hasattr(env, 'sim'):
            env = env.env
        
        if not hasattr(env, 'sim'):
            raise RuntimeError("Could not find environment with sim access")
        
        return env


def create_tracking_env(task_suite_name, task_name, img_h=128, img_w=128, 
                        agentview_vertex_ids=None, eye_in_hand_vertex_ids=None,
                        use_full_vertices=False, max_vertices_per_mesh=50):
    """
    Create a LIBERO environment with vertex tracking.
    
    Args:
        task_suite_name: Name of the task suite
        task_name: Specific task name
        agentview_vertex_ids: List of vertex IDs to track for agentview
        eye_in_hand_vertex_ids: List of vertex IDs to track for eye-in-hand
        use_full_vertices: If True, extract all vertices without subsampling
        max_vertices_per_mesh: Maximum vertices per mesh when subsampling (default 50)
    
    Returns:
        Wrapped environment with vertex tracking
    """
    from libero import benchmark, get_libero_path
    from libero.envs import OffScreenRenderEnv
    import os
    
    # Get task details
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task_from_name(task_name)
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), 
                                  task.problem_folder, task.bddl_file)
    
    # Create base environment
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=img_h,
        camera_widths=img_w,
        render_gpu_device_id=-1
    )
    
    # Add vertex tracking
    env = MeshVertexTracker(
        env,
        agentview_vertex_ids=agentview_vertex_ids,
        eye_in_hand_vertex_ids=eye_in_hand_vertex_ids,
        use_full_vertices=use_full_vertices,
        max_vertices_per_mesh=max_vertices_per_mesh
    )
    
    return env


if __name__ == "__main__":
    # Test the vertex tracker with specific vertex IDs
    
    # Your specified vertex IDs
    # For agentview: track eye-in-hand vertices
    agentview_ids = [2965, 3034, 2955, 3055, 3040, 2981, 2966, 3052, 3022]
    
    # For eye-in-hand: track agentview vertices  
    eye_in_hand_ids = [2955, 3055, 2991, 3041, 2789, 2758, 2774, 2965, 3034]
    
    print("Creating environment with vertex tracking...")
    env = create_tracking_env(
        task_suite_name="libero_goal",
        task_name="put_the_bowl_on_the_plate",
        agentview_vertex_ids=agentview_ids,
        eye_in_hand_vertex_ids=eye_in_hand_ids
    )
    
    print("\nResetting environment...")
    obs = env.reset()
    
    print("\nChecking tracked vertices:")
    if 'agentview_tracked_vertices_3d' in obs:
        print(f"Agentview tracking {len(obs['agentview_tracked_vertices_3d'])} vertices")
        print(f"  3D shape: {obs['agentview_tracked_vertices_3d'].shape}")
        print(f"  2D shape: {obs['agentview_tracked_vertices_2d'].shape}")
        print(f"  Tracked IDs: {obs['agentview_tracked_vertex_ids']}")
    
    if 'eye_in_hand_tracked_vertices_3d' in obs:
        print(f"\nEye-in-hand tracking {len(obs['eye_in_hand_tracked_vertices_3d'])} vertices")
        print(f"  3D shape: {obs['eye_in_hand_tracked_vertices_3d'].shape}")
        print(f"  2D shape: {obs['eye_in_hand_tracked_vertices_2d'].shape}")
        print(f"  Tracked IDs: {obs['eye_in_hand_tracked_vertex_ids']}")
    
    # Take a step to see updated positions
    print("\nTaking a step...")
    action = np.zeros(7)  # Dummy action
    obs, reward, done, info = env.step(action)
    
    print("\nAfter step:")
    if 'agentview_tracked_vertices_2d' in obs:
        print(f"Agentview first vertex 2D: {obs['agentview_tracked_vertices_2d'][0]}")
    if 'eye_in_hand_tracked_vertices_2d' in obs:
        print(f"Eye-in-hand first vertex 2D: {obs['eye_in_hand_tracked_vertices_2d'][0]}")
    

    action = np.zeros(7)  # Dummy action
    obs, reward, done, info = env.step(action)
    
    print("\nAfter step:")
    if 'agentview_tracked_vertices_2d' in obs:
        print(f"Agentview first vertex 2D: {obs['agentview_tracked_vertices_2d'][0]}")
    if 'eye_in_hand_tracked_vertices_2d' in obs:
        print(f"Eye-in-hand first vertex 2D: {obs['eye_in_hand_tracked_vertices_2d'][0]}")


    env.close()
    print("\n✅ Vertex tracking test complete!")