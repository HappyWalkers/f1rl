import numpy as np
import os

CAR_SCALE = 5.

def get_render_callback(renderers):
    def render_callback(env_renderer):
        for renderer in renderers:
            renderer.render(env_renderer)
    return render_callback

# class ScanRenderer():
#     def __init__(self, scan, color, angle_min, angle_max):
#         self.scan = scan
#         self.color = color
#         self.angle_min = angle_min
#         self.angle_max = angle_max
#         self.dimension = len(scan)
#         self.drawn_scan = []

#     def save_scan(self, scan, pose):
#         self.scan = scan.copy()
#         self.pose = pose.copy()
#         self.lidar_angles = np.linspace(self.angle_min, self.angle_max, self.dimension) * np.pi / 180 + pose[2]

#     def rays2world(self, distance):
#         # convert lidar scan distance to 2d locations in space
#         x = distance * np.cos(self.lidar_angles)
#         y = distance * np.sin(self.lidar_angles)
#         return x, y

#     def render(self, e):
#         x, y = self.rays2world(self.scan)
#         x = (x + self.pose[0]) * PLOT_SCALE
#         y = (y + self.pose[1]) * PLOT_SCALE

#         for i in range(self.dimension):
#             if len(self.drawn_scan) < self.dimension:
#                 b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [x[i] , y[i], 0.]),
#                                 ('c3B/stream', self.color))
#                 self.drawn_scan.append(b)
#             else:
#                 self.drawn_scan[i].vertices = [x[i], y[i], 0.]

# class SteerRenderer():
#     def __init__(self, pose, control, color) -> None:
#         self.pose = pose
#         self.control = control
#         self.bars = []
#         self.color = color

#     def update(self, pose, control):
#         self.pose = pose.copy()
#         self.control = control.copy()
        
#     def reset(self):
#         pass
    
#     def render(self, e):
#         # vehicle shape constants
#         BAR_LENGTH = 0.7 * PLOT_SCALE
#         BAR_WIDTH = 0.05 * PLOT_SCALE
#         draw_pose = np.zeros(3)

#         draw_pose[2] = self.pose[4] + self.pose[2] * np.pi/2
#         draw_pose[0] = self.pose[0] + BAR_LENGTH/2 * np.cos(draw_pose[2])
#         draw_pose[1] = self.pose[1] + BAR_LENGTH/2 * np.sin(draw_pose[2])
        
        
#         if len(self.bars) < 1:
#             vertices_np = get_vertices(np.array([0., 0., 0.]), BAR_LENGTH, BAR_WIDTH)
#             vertices = list(vertices_np.flatten())
#             car = e.batch.add(4, GL_QUADS, None, ('v2f', vertices), ('c3B', self.color * 4))
#             self.bars.append(car)
#         else:
#             vertices_np = PLOT_SCALE * get_vertices(np.float64(draw_pose), BAR_LENGTH, BAR_WIDTH)
#             vertices = list(vertices_np.flatten())
#             self.bars[0].vertices = vertices
#             self.bars[0].colors = self.color * 4
            

# class AcceRenderer():
#     def __init__(self, pose, control) -> None:
#         self.pose = pose
#         self.control = control
#         self.bars = []

#     def update(self, pose, control):
#         self.pose = pose.copy()
#         self.control = control.copy() / 3
        
#     def reset(self):
#         pass
    
#     def render(self, e):
#         # vehicle shape constants
#         BAR_LENGTH = np.abs(self.control[1]) * PLOT_SCALE
#         BAR_WIDTH = 0.05 * PLOT_SCALE
#         draw_pose = np.zeros(3)
#         draw_pose[2] = self.pose[4] + np.pi
#         draw_pose[0] = self.pose[0] + BAR_LENGTH/2 * np.cos(draw_pose[2])
#         draw_pose[1] = self.pose[1] + BAR_LENGTH/2 * np.sin(draw_pose[2])
        
        
#         if self.control[1] > 0:
#             self.color = [0, 255, 0]
#         else:
#             self.color = [255, 0, 0]
        
#         if len(self.bars) < 1:
#             vertices_np = get_vertices(np.array([0., 0., 0.]), BAR_LENGTH, BAR_WIDTH)
#             vertices = list(vertices_np.flatten())
#             car = e.batch.add(4, GL_QUADS, None, ('v2f', vertices), ('c3B', self.color * 4))
#             self.bars.append(car)
#         else:
#             vertices_np = PLOT_SCALE * get_vertices(np.float64(draw_pose), BAR_LENGTH, BAR_WIDTH)
#             vertices = list(vertices_np.flatten())
#             self.bars[0].vertices = vertices
#             self.bars[0].colors = self.color * 4

class WaypointRenderer():
    
    def __init__(self, waypoints, color, point_size=5, mode='point') -> None:
        self.point_color = color
        self.waypoints = waypoints
        self.point_size = point_size
        self.mode = mode
        self.waypoint_render = None
        
    def update(self, waypoints):
        self.waypoints = waypoints.copy()
        
    def reset(self):
        pass

    def render(self, e):
        points = np.stack([self.waypoints[:,0], self.waypoints[:,1]], axis=1)
        if self.waypoint_render is None:
            if self.mode == 'point':
                self.waypoint_render = e.render_points(points, color=self.point_color, size=self.point_size)
            else:
                self.waypoint_render = e.render_closed_lines(points, color=self.point_color, size=self.point_size)
        else:
            self.waypoint_render.setData(points)
        
                
class MapWaypointRenderer():
    
    def __init__(self, waypoints, color=[255, 255, 255], point_size=4, mode='point') -> None:
        self.point_color = tuple(color)
        self.waypoints = waypoints
        self.point_size = point_size
        self.position = [0, 0]
        self.mode = mode
        self.waypoint_render = None
        
    def update(self, position):
        self.position = position.copy()
    
    def render(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        points = np.stack([self.waypoints[:,1], self.waypoints[:,2]], axis=1)
        if self.waypoint_render is None: # Render only once if QT is used
            if self.mode == 'point':
                self.waypoint_render = e.render_points(points, color=self.point_color, size=self.point_size)
            else:
                self.waypoint_render = e.render_closed_lines(points, color=self.point_color, size=self.point_size)

class TrackRenders:
    def __init__(self, track, width, color=[255, 255, 255], boundary_color=[255, 0, 0], point_size=4, render_mode='point', max_points=150):
        import jax
        import jax.numpy as jnp
        waypoints = track.waypoints
        waypoints_render_subsample = waypoints[np.arange(waypoints.shape[0], step=waypoints.shape[0]//max_points), :] # render only max_points waypoints
        self.map_waypoint_renderer = MapWaypointRenderer(waypoints_render_subsample, color=color, mode=render_mode)
        waypoints_boundary = jnp.concatenate([waypoints_render_subsample[:, 0:1], track.vmap_frenet_to_cartesian_jax(jnp.concatenate([waypoints_render_subsample[:, 0:1], 
                                                                                        jnp.ones_like(waypoints_render_subsample[:, 0:1]) * width/2,
                                                                                        jnp.zeros_like(waypoints_render_subsample[:, 0:1])], axis=1))], axis=1)
        self.map_waypoint_renderer_l = MapWaypointRenderer(waypoints_boundary, color=boundary_color, mode=render_mode)
        waypoints_boundary = jnp.concatenate([waypoints_render_subsample[:, 0:1], track.vmap_frenet_to_cartesian_jax(jnp.concatenate([waypoints_render_subsample[:, 0:1], 
                                                                                            jnp.ones_like(waypoints_render_subsample[:, 0:1]) * -width/2,
                                                                                            jnp.zeros_like(waypoints_render_subsample[:, 0:1])], axis=1))], axis=1)
        self.map_waypoint_renderer_r = MapWaypointRenderer(waypoints_boundary, color=boundary_color, mode=render_mode)
    
    def list(self):    
        return [self.map_waypoint_renderer, self.map_waypoint_renderer_l, self.map_waypoint_renderer_r]
    
    def update(self, pose):
        self.map_waypoint_renderer.update(pose)
        self.map_waypoint_renderer_l.update(pose)
        self.map_waypoint_renderer_r.update(pose)
    