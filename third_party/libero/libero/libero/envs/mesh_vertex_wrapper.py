"""
Mesh Vertex Extraction Wrapper for LIBERO Environments
This wrapper extracts ALL mesh vertices (not just geometry centers) from MuJoCo simulation.
"""

import numpy as np
import gym
from gym import spaces


class MeshVertexWrapper(gym.Wrapper):
    """
    Wrapper that extracts mesh vertices from the simulation.
    Can extract ALL vertices or only specific vertex IDs.
    """
    
    def __init__(self, env, max_points=10000, include_non_mesh_geoms=True,
                 specific_vertex_ids=None):
        """
        Args:
            env: Environment to wrap
            max_points: Maximum number of points to extract
            include_non_mesh_geoms: Whether to include non-mesh geometry centers
            specific_vertex_ids: List of specific global vertex IDs to extract
                                If provided, ONLY these vertices are returned
                                e.g., [2955, 3055, 2991, 3041, 2774, 2965, 3034]
        """
        super().__init__(env)
        self.max_points = max_points
        self.include_non_mesh_geoms = include_non_mesh_geoms
        self.specific_vertex_ids = specific_vertex_ids
        
        # Modify observation space if it exists
        if hasattr(env, 'observation_space'):
            self.observation_space = self._modify_obs_space(env.observation_space)
        
    def _modify_obs_space(self, obs_space):
        """Add mesh vertex spaces to observation space."""
        if isinstance(obs_space, spaces.Dict):
            new_spaces = dict(obs_space.spaces)
            
            # Determine number of points based on mode
            if self.specific_vertex_ids is not None:
                num_points = len(self.specific_vertex_ids)
            else:
                num_points = self.max_points
            
            # Add mesh vertex observations
            new_spaces['mesh_vertices_3d'] = spaces.Box(
                low=-10.0, high=10.0, shape=(num_points, 3), dtype=np.float32
            )
            new_spaces['mesh_vertex_labels'] = spaces.Box(
                low=0, high=255, shape=(num_points,), dtype=np.uint8
            )
            new_spaces['mesh_vertex_geom_ids'] = spaces.Box(
                low=-1, high=1000, shape=(num_points,), dtype=np.int32
            )
            new_spaces['mesh_vertex_count'] = spaces.Box(
                low=0, high=num_points, shape=(1,), dtype=np.int32
            )
            
            return spaces.Dict(new_spaces)
        return obs_space
    
    def reset(self, **kwargs):
        """Reset environment and add mesh vertices to observation."""
        obs = self.env.reset(**kwargs)
        
        # Extract mesh vertices
        mesh_data = self._extract_mesh_vertices()
        obs['mesh_vertices_3d'] = mesh_data['vertices']
        obs['mesh_vertex_labels'] = mesh_data['labels']
        obs['mesh_vertex_geom_ids'] = mesh_data['geom_ids']
        obs['mesh_vertex_count'] = mesh_data['count']
        
        return obs
    
    def step(self, action):
        """Step environment and add mesh vertices to observation."""
        obs, reward, done, info = self.env.step(action)
        
        # Extract mesh vertices
        mesh_data = self._extract_mesh_vertices()
        obs['mesh_vertices_3d'] = mesh_data['vertices']
        obs['mesh_vertex_labels'] = mesh_data['labels']
        obs['mesh_vertex_geom_ids'] = mesh_data['geom_ids']
        obs['mesh_vertex_count'] = mesh_data['count']
        
        return obs, reward, done, info
    
    def _extract_all_mesh_vertices(self):
        """Backward compatibility alias for _extract_mesh_vertices()."""
        return self._extract_mesh_vertices()
    
    def _extract_mesh_vertices(self):
        """
        Extract mesh vertices from MuJoCo simulation.
        If specific_vertex_ids is provided, returns ONLY those vertices.
        Otherwise returns all vertices up to max_points.
        """
        # Navigate to environment with sim
        env = self.env
        while hasattr(env, 'env') and not hasattr(env, 'sim'):
            env = env.env
        
        # Determine output size
        if self.specific_vertex_ids is not None:
            output_size = len(self.specific_vertex_ids)
        else:
            output_size = self.max_points
        
        if not hasattr(env, 'sim'):
            # Return empty data if we can't access sim
            return {
                'vertices': np.zeros((output_size, 3), dtype=np.float32),
                'labels': np.zeros(output_size, dtype=np.uint8),
                'geom_ids': np.full(output_size, -1, dtype=np.int32),
                'count': np.array([0], dtype=np.int32)
            }
        
        sim = env.sim
        
        all_vertices = []
        all_labels = []
        all_geom_ids = []
        global_vertex_counter = 0  # Track global vertex ID
        
        # If specific IDs requested, prepare storage
        if self.specific_vertex_ids is not None:
            specific_vertices = np.zeros((len(self.specific_vertex_ids), 3), dtype=np.float32)
            specific_labels = np.zeros(len(self.specific_vertex_ids), dtype=np.uint8)
            specific_geom_ids = np.full(len(self.specific_vertex_ids), -1, dtype=np.int32)
        
        # Define label mapping
        LABEL_MAPPING = {
            'robot': 1,
            'gripper': 2,
            'object': 3,
            'table': 4,
            'mount': 5,
            'floor': 6,
            'wall': 7,
            'visualization': 8,
            'unknown': 9
        }
        
        # Process each geometry
        for geom_id in range(sim.model.ngeom):
            geom_name = sim.model.geom_id2name(geom_id)
            geom_type = sim.model.geom_type[geom_id]
            
            # Get label for this geometry
            label = self._categorize_geometry(geom_name, LABEL_MAPPING)
            
            # Get transformation for this geometry
            geom_pos = sim.data.geom_xpos[geom_id].copy()
            geom_mat = sim.data.geom_xmat[geom_id].reshape(3, 3).copy()
            
            # Check if this is a mesh geometry (type 7)
            if geom_type == 7:  # mjGEOM_MESH
                mesh_id = sim.model.geom_dataid[geom_id]
                
                # Get mesh vertex range
                vert_start = sim.model.mesh_vertadr[mesh_id]
                if mesh_id < sim.model.nmesh - 1:
                    vert_end = sim.model.mesh_vertadr[mesh_id + 1]
                else:
                    vert_end = len(sim.model.mesh_vert)
                
                # Extract local vertices
                local_vertices = sim.model.mesh_vert[vert_start:vert_end].reshape(-1, 3)
                
                # Transform to world coordinates
                world_vertices = (local_vertices @ geom_mat.T) + geom_pos
                
                # Add all vertices from this mesh
                for vertex_idx, vertex in enumerate(world_vertices):
                    # If specific vertices requested, check if this is one we want
                    if self.specific_vertex_ids is not None:
                        if global_vertex_counter in self.specific_vertex_ids:
                            # Found a specific vertex we want
                            idx_in_list = self.specific_vertex_ids.index(global_vertex_counter)
                            specific_vertices[idx_in_list] = vertex
                            specific_labels[idx_in_list] = label
                            specific_geom_ids[idx_in_list] = geom_id
                    else:
                        # Extract all vertices mode
                        all_vertices.append(vertex)
                        all_labels.append(label)
                        all_geom_ids.append(geom_id)
                        
                        if len(all_vertices) >= self.max_points:
                            break
                    
                    global_vertex_counter += 1
            
            elif self.include_non_mesh_geoms:
                # For non-mesh geometries, just add the center point
                if self.specific_vertex_ids is None:
                    all_vertices.append(geom_pos)
                    all_labels.append(label)
                    all_geom_ids.append(geom_id)
                
                # Non-mesh geoms don't increment global_vertex_counter
                # since they don't have actual vertices
            
            if self.specific_vertex_ids is None and len(all_vertices) >= self.max_points:
                break
        
        # Prepare output based on mode
        if self.specific_vertex_ids is not None:
            # Return only specific vertices
            return {
                'vertices': specific_vertices,
                'labels': specific_labels,
                'geom_ids': specific_geom_ids,
                'count': np.array([len(self.specific_vertex_ids)], dtype=np.int32)
            }
        else:
            # Return all vertices (original behavior)
            num_points = len(all_vertices)
            vertices_array = np.zeros((self.max_points, 3), dtype=np.float32)
            labels_array = np.zeros(self.max_points, dtype=np.uint8)
            geom_ids_array = np.full(self.max_points, -1, dtype=np.int32)
            
            if num_points > 0:
                actual_count = min(num_points, self.max_points)
                vertices_array[:actual_count] = np.array(all_vertices[:actual_count])
                labels_array[:actual_count] = np.array(all_labels[:actual_count])
                geom_ids_array[:actual_count] = np.array(all_geom_ids[:actual_count])
            
            return {
                'vertices': vertices_array,
                'labels': labels_array,
                'geom_ids': geom_ids_array,
                'count': np.array([min(num_points, self.max_points)], dtype=np.int32)
            }
    
    def _categorize_geometry(self, geom_name, label_mapping):
        """Categorize geometry based on its name."""
        if geom_name is None:
            return label_mapping.get('unknown', 9)
        name_lower = geom_name.lower()
        
        # Check for specific categories
        if 'robot' in name_lower or 'panda' in name_lower or 'link' in name_lower:
            if 'finger' in name_lower or 'hand' in name_lower or 'gripper' in name_lower:
                return label_mapping['gripper']
            return label_mapping['robot']
        elif 'table' in name_lower:
            return label_mapping['table']
        elif 'floor' in name_lower:
            return label_mapping['floor']
        elif 'wall' in name_lower:
            return label_mapping['wall']
        elif 'mount' in name_lower:
            return label_mapping['mount']
        elif 'visual' in name_lower or 'collision' in name_lower:
            return label_mapping['visualization']
        else:
            # Likely an object
            return label_mapping['object']