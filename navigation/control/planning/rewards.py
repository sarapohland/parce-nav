import numpy as np

def goal_cost(xtraj, xG, intermed_weight=1.0, final_weight=1.0):
    xG = xG[:2]
    x0 = xtraj[:,0,:2]
    xpath = xtraj[:,:,:2]
    N, H, _ = np.shape(xpath)

    goal_dist = np.linalg.norm(xpath - xG, axis=2)

    weights = np.concatenate((np.full((N,H-1), intermed_weight), np.full((N,1), final_weight)), axis=1)
    goal_dist *= weights
    return np.sum(goal_dist, axis=1)

def goalx_cost(xtraj, xG, intermed_weight=1.0, final_weight=1.0):
    xG = xG[0]
    x0 = xtraj[:,0,0]
    xpath = xtraj[:,:,0]
    N, H = np.shape(xpath)

    goal_dist = np.abs(xpath - xG)

    weights = np.concatenate((np.full((N,H-1), intermed_weight), np.full((N,1), final_weight)), axis=1)
    goal_dist *= weights
    return np.sum(goal_dist, axis=1)

def goaly_cost(xtraj, xG, intermed_weight=1.0, final_weight=1.0):
    xG = xG[1]
    x0 = xtraj[:,0,1]
    xpath = xtraj[:,:,1]
    N, H = np.shape(xpath)

    goal_dist = np.abs(xpath - xG)

    weights = np.concatenate((np.full((N,H-1), intermed_weight), np.full((N,1), final_weight)), axis=1)
    goal_dist *= weights
    return np.sum(goal_dist, axis=1)

def path_cost(xtraj, xG, intermed_weight=None, final_weight=None):
    xG = xG[:2]
    x0 = xtraj[:,0,:2]
    xpath = xtraj[:,:,:2]
    N, H, _ = np.shape(xpath)

    path_length = np.linalg.norm(np.diff(xpath, axis=1), axis=2)
    return np.sum(path_length, axis=1)

def heading_cost(xtraj, xG, intermed_weight=None, final_weight=None):
    N, H, _ = np.shape(xtraj)

    goal_ang = np.arctan2(xG[1] - xtraj[:,:,1], xG[0] - xtraj[:,:,0])
    ang_diff = np.abs(goal_ang - xtraj[:,:,2])
    ang_diff[ang_diff > np.pi] = 2*np.pi - ang_diff[ang_diff > np.pi]
    ang_diff = ang_diff**2

    weights = np.concatenate((np.full((N,H-1), intermed_weight), np.full((N,1), final_weight)), axis=1)
    ang_diff *= weights
    return np.sum(ang_diff, axis=1)

def distance_reward(xtraj, xG, intermed_weight=1.0, final_weight=1.0):
    xG = xG[:2]
    x0 = xtraj[:,0,:2]
    xpath = xtraj[:,:,:2]
    N, H, _ = np.shape(xpath)

    goal_diff = (xG - x0)[:,:,np.newaxis]
    path_diff = xpath - x0[:,np.newaxis,:]
    progress = np.matmul(path_diff, goal_diff).reshape((N,-1))
    progress /= np.linalg.norm(xG - x0, axis=1)[:,np.newaxis]

    weights = np.concatenate((np.full((N,H-1), intermed_weight), np.full((N,1), final_weight)), axis=1)
    progress *= weights
    return np.sum(progress, axis=1)

def normalize(rewards):
    min = np.min(rewards)
    max = np.max(rewards)
    return (rewards - min) / (max - min)