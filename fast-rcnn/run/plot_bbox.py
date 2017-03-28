import numpy as np
import pickle
import cv2

def _bbox_pred(boxes, box_deltas):
    """Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1e-14
    heights = boxes[:, 3] - boxes[:, 1] + 1e-14
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = box_deltas[:, 0::4]
    dy = box_deltas[:, 1::4]
    dw = box_deltas[:, 2::4]
    dh = box_deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

if __name__ == "__main__":
    with open("roidb.pickle","r") as f:
        roidb = pickle.load(f)

    i = 0
    img_path = roidb[i]['image']
    nb_gt_rois = np.count_nonzero(roidb[i]['gt_classes'])
    nb_yolo_rois = len(roidb[i]['gt_classes']) - nb_gt_rois
    
    # read image from its absolute path
    img_example = cv2.imread(img_path)
    height, width, _ = img_example.shape
    print img_example.shape
    

    img = roidb[i]['image_data'].transpose(1,2,0)
    #img = img[:,:,::-1]
    #img[:, :, 0] += 103.939 
    #img[:,:,1] += 116.779
    #img[:,:,2] += 123.68
    #img = img[:,:,::-1]
    img[:, :, 0] += 123.68 
    img[:,:,1] += 116.779
    img[:,:,2] += 103.939
    
    img = img.astype(np.uint8)
    
    
    img = img.copy()

    height, width,_ = img.shape
    print img.shape
    
    

    gt_boxes = np.zeros((nb_gt_rois,4))
    yolo_boxes = np.zeros((nb_yolo_rois,4))
    bbox_targets = np.zeros((nb_gt_rois + nb_yolo_rois,4))

    gt_boxes[:,0]= roidb[i]['box_normalized'][0:nb_gt_rois,0]*width
    gt_boxes[:,1]= roidb[i]['box_normalized'][0:nb_gt_rois,1]*height
    gt_boxes[:,2]= roidb[i]['box_normalized'][0:nb_gt_rois,2]*width
    gt_boxes[:,3]= roidb[i]['box_normalized'][0:nb_gt_rois,3]*height
  
    yolo_boxes[:,0] = roidb[i]['box_normalized'][nb_gt_rois:,0]*width
    yolo_boxes[:,1] = roidb[i]['box_normalized'][nb_gt_rois:,1]*height
    yolo_boxes[:,2] = roidb[i]['box_normalized'][nb_gt_rois:,2]*width
    yolo_boxes[:,3] = roidb[i]['box_normalized'][nb_gt_rois:,3]*height

    boxes = np.concatenate((gt_boxes, yolo_boxes),axis = 0)
    #print boxes.shape
    
    bbox_targets = roidb[i]['bbox_targets']
    #print bbox_targets.shape
    pred_boxes = _bbox_pred(boxes, bbox_targets[:,1:5])
    print 'num of gt_boxes: '+str(nb_gt_rois)   
    # draw the gound truth rois
    for ind in xrange(nb_gt_rois):
        xmin, ymin, xmax, ymax = gt_boxes[ind].astype(np.int32)
        print xmin, ymin, xmax, ymax
        cv2.rectangle(img, (xmin,ymin),(xmax,ymax),(0,255,0),5)
    
    
    # draw yolo proposed rois (BACKGROUND BOXES RMOVED)
    for ind in xrange(nb_yolo_rois):
        xmin, ymin, xmax, ymax = yolo_boxes[ind].astype(np.int32)
        # remove background boxes
        if bbox_targets[ind+nb_gt_rois][0] == 0:
            continue
        #print xmin, ymin, xmax, ymax
        cv2.rectangle(img, (xmin, ymin),(xmax, ymax),(0,0, 225),2)
    
    
    # draw predicted boxes (BACKGOURND BOXES REMOVED)
    for ind in xrange(nb_gt_rois+nb_yolo_rois):
        xmin, ymin, xmax, ymax = pred_boxes[ind].astype(np.int32)
        # remove background boxes
        if bbox_targets[ind][0] == 0:
            continue
        print xmin,ymin,xmax,ymax
        cv2.rectangle(img, (xmin, ymin),(xmax, ymax),(255,0,0 ),2)
    


    
    #cv2.imwrite("all_boxes.png",img)
    cv2.imshow("image",img)
   
    
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    
    
