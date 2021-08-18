img=/gogz/trainingH/

# loadmodel=./models/kitti_best_model/model_best_0.0172.tar
loadmodel=AANET.pth
devices=0,1
savepath=./aa_origin
model=aanet
python predict_mid.py --image $img  \
--loadmodel $loadmodel --devices $devices --savepath $savepath  --vis --model $model  \
--feature_type aanet  --feature_pyramid_network 
