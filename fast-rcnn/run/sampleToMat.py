import pdb
import sys
import argparse
import json
from scipy.io import savemat


# read result data generated from darknet_yolo.
def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="dataset name. eg. voc_2012_test or voc_2012_trainval")
    ap.add_argument("-i","--imageList",required=True,help="name list of the images")
    args = vars(ap.parse_args())
    return args


def main():
    args = parse_arguments()
    datasetName = args['dataset']
    print 'dataset name:' + datasetName
    imageListFile = args['imageList']
    print 'imageListFile:' + imageListFile
    
    classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

    imageList = []
    with open(imageListFile,'r') as infile:
        for line in infile:
            imageList.append(line.strip())
    
    dic = {}
    for classname in classes:
        path = '/nfs/isicvlnas01/users/zekunl/projects/darknet_2016_Sep_02/results/'+datasetName+'_'+classname+'.txt'
        print 'loading '+path
        with open(path, 'r') as infile:
            for line in infile:
                [image,prob,xmin,ymin,xmax,ymax] = line.split()
                box = [float(xmin),float(ymin),float(xmax),float(ymax)]
                if image in imageList: # check if ROI info of this image is needed
                    if image in dic: # if key already added in the dictionary
                        dic[image].append(box)
                    else: # if key not added in the dictionary
                        dic[image] = [box]
		
    print 'dictionary size ' + str(len(dic))

    list_image_box = [ [image,box] for image, box in dic.items() ]
    sorted_list = sorted(list_image_box,key=lambda x: x[0])
    
    #split key,values <image,box> in dic
    images = zip(*sorted_list)[0] 
    boxes = zip(*sorted_list)[1]
    
    # --save file --
    savemat('/nfs/isicvlnas01/users/zekunl/projects/keras-fast-rcnn/fast-rcnn/data/yolo_data/'+datasetName+'.mat',mdict={'boxes':boxes},oned_as='row')
    print 'rois saved as Mat'
    # print statistics
    cnt_boxes = [len(img_box) for img_box in boxes]
    print 'average number of boxes per image:'
    print sum(cnt_boxes)/float(len(cnt_boxes))
    print 'min number of boxes for one image:'
    print min(cnt_boxes)
    print 'max number of boxes for one image:'
    print max(cnt_boxes)

if __name__ =="__main__":
	main()
