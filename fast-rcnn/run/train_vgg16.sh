#!/bin/bash

echo "SGE_GPU:"
echo ${SGE_GPU}
echo "TMPDIR:"
echo ${TMPDIR}
set CUDA_VISIBLE_DEVICES=${SGE_GPU}
echo "CUDA_VISIBLE_DEVICES:"
echo ${CUDA_VISIBLE_DEVICES}

export PATH="/nfs/isicvlnas01/share/anaconda/bin/:/nfs/isicvlnas01/users/zekunl/zlib/caffe/:/usr/local/cuda/bin:~/bin:/usr/local/bin:/usr/ucb/:/sbin:/usr/sbin:/usr/local/sbin:/usr/hosts:/usr/games"

export  LD_LIBRARY_PATH="/usr/local/cuda/lib64/:/nfs/isicvlnas01/share/SGE_ROOT/lib/linux-x64/:/nfs/isicvlnas01/share/cudnn-7.5-linux-x64-v5.0-ga/lib64/"

cd /nfs/isicvlnas01/users/zekunl/projects/keras-fast-rcnn/fast-rcnn/


time python -m pdb tools/train_net.py --gpu ${SGE_GPU} --imdb voc_2012_train
#  --solver models/VGG16/solver.prototxt \
#  --weights data/imagenet_models/VGG16.v2.caffemodel \
#  --imdb voc_2012_train

#time ./tools/test_net.py --gpu $1 \
#  --def models/VGG16/test.prototxt \
#  --net output/default/voc_2007_trainval/vgg16_fast_rcnn_iter_40000.caffemodel \
#  --imdb voc_2007_test
