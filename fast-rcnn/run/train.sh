#!/bin/bash
echo "HOSTNAME:"
echo ${HOSTNAME}
echo "SGE_GPU:"
echo ${SGE_GPU}
echo "TMPDIR:"
echo ${TMPDIR}
CUDA_VISIBLE_DEVICES=${SGE_GPU}
echo "CUDA_VISIBLE_DEVICES:"
echo ${CUDA_VISIBLE_DEVICES}

export PATH="/nfs/isicvlnas01/share/anaconda/bin/:/nfs/isicvlnas01/users/zekunl/zlib/caffe/:/usr/local/cuda/bin:~/bin:/usr/local/bin:/usr/ucb/:/sbin:/usr/sbin:/usr/local/sbin:/usr/hosts:/usr/games"

export  LD_LIBRARY_PATH="/usr/local/cuda/lib64/:/nfs/isicvlnas01/share/SGE_ROOT/lib/linux-x64/:/nfs/isicvlnas01/share/cudnn-7.5-linux-x64-v5.0-ga/lib64/"

# -------modify accordingly ------------
# data copied from datastore has to be compatible with '--imdb' param input 
cd ${TMPDIR}

cp /nfs/isicvlnas01/users/zekunl/datastore/VOCdevkit/VOC2012.tar.gz ${TMPDIR}
tar -xzf VOC2012.tar.gz

cd /nfs/isicvlnas01/users/zekunl/projects/keras-fast-rcnn/fast-rcnn/

# if gpu allocated successfully
# remeber to modify lib/keras-model/fastrcnn.py for gpu allocation
time python -m pdb tools/train_net.py \
    --imdb voc_2012_trainval \
    --data ${TMPDIR}/VOC2012 \
    --proposal yolo --targetnorm 1 \
    --mergedense 1 \
    --numbboxout 4 \
    --pool slicepool\ 
    --outdir 

#--weights output/weights/samperepoch1000_maxpool_bgfilter_lr0.0001-train-00-0.76.hdf5 
