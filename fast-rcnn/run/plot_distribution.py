import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open("roidb_targets.pickle","r") as f:
        roidb = pickle.load(f)

    i = 0
    nb_gt_rois = np.count_nonzero(roidb[i]['gt_classes'])
    nb_yolo_rois = len(roidb[i]['gt_classes']) - nb_gt_rois
    
    bbox_targets = roidb[i]['bbox_targets']

    labels = bbox_targets[:,0]
    indices = np.nonzero(labels)

    delta_x = bbox_targets[:,1][indices]
    delta_y = bbox_targets[:,2][indices]
    delta_w = bbox_targets[:,3][indices]
    delta_h = bbox_targets[:,4][indices]
    
    df = pd.DataFrame({'delta_x': delta_x, 'delat_y': delta_y,'delta_w':delta_w, 'delta_h':delta_h},columns = ['delta_x', 'delta_y', 'delta_w', 'delta_h'])
    #df = pd.DataFrame({'delta_h': delta_h},columns = ['delta_h'])
    
    #df = pd.DataFrame(bbox_targets[:,1:5], columns=['delta_x', 'delta_y', 'delta_w', 'delta_h'])
    #df.plot.bar()
    df.plot.hist(alpha = 0.5,bins = 20)
    plt.show()
    
