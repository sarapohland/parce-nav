import numpy as np
from scipy.spatial.transform import Rotation as R

def heading_from_quaternion(orientation):
    quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
    heading = R.from_quat(quaternion).as_euler('zyx')[0]
    return heading

def robot_to_world(robot_paths, state):
    if robot_paths.ndim == 1:
        xs, ys, thetas = robot_paths[0], robot_paths[1], robot_paths[2]
    elif robot_paths.ndim == 2:
        xs, ys, thetas = robot_paths[:,0], robot_paths[:,1], robot_paths[:,2]
    elif robot_paths.ndim == 3:
        xs, ys, thetas = robot_paths[:,:,0], robot_paths[:,:,1], robot_paths[:,:,2]
    else:
        raise NotImplementedError('Robot path has an unfamiliar number of dimensions.')
    
    x_r, y_r, theta_r = state
    world_xs = x_r + xs * np.cos(theta_r) - ys * np.sin(theta_r)
    world_ys = y_r + xs * np.sin(theta_r) + ys * np.cos(theta_r)
    world_thetas = thetas + theta_r

    if robot_paths.ndim == 1:
        world_paths = np.array([world_xs, world_ys, world_thetas])
    elif robot_paths.ndim == 2:
        world_paths = np.concatenate((world_xs[:,np.newaxis], world_ys[:,np.newaxis], world_thetas[:,np.newaxis]), axis=1)
    else:
        world_paths = np.concatenate((world_xs[:,:,np.newaxis], world_ys[:,:,np.newaxis], world_thetas[:,:,np.newaxis]), axis=2)
    
    return world_paths

def world_to_robot(world_paths, state):
    x_r, y_r, theta_r = state

    if world_paths.ndim == 1:
        x, y, theta = world_paths[0], world_paths[1], world_paths[2]
        d = np.sqrt((x - x_r)**2 + (y - y_r)**2)
        robot_theta = np.arctan2(y - y_r, x - x_r) - theta_r
        robot_x = d * np.cos(robot_theta)
        robot_y = d * np.sin(robot_theta)
        return np.array([robot_x, robot_y, robot_theta])

    if world_paths.ndim == 2:
        xs, ys, thetas = world_paths[:,0], world_paths[:,1], world_paths[:,2]
    elif world_paths.ndim == 3:
        xs, ys, thetas = world_paths[:,:,0], world_paths[:,:,1], world_paths[:,:,2]
    else:
        raise NotImplementedError('Robot path has an unfamiliar number of dimensions.')
    
    d = [np.sqrt((x - x_r)**2 + (y - y_r)**2) for (x, y) in zip(xs, ys)]
    robot_thetas = np.array([np.arctan2(y - y_r, x - x_r) - theta_r for (x, y) in zip(xs, ys)])
    robot_xs = d * np.cos(robot_thetas)
    robot_ys = d * np.sin(robot_thetas)

    if world_paths.ndim == 2:
        robot_paths = np.concatenate((robot_xs[:,np.newaxis], robot_ys[:,np.newaxis], robot_thetas[:,np.newaxis]), axis=1)
    else:
        robot_paths = np.concatenate((robot_xs[:,:,np.newaxis], robot_ys[:,:,np.newaxis], robot_thetas[:,:,np.newaxis]), axis=2)
    
    return robot_paths