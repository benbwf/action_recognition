# test the TSN and TSM on Kinetics using 8-frame, you should get top-1 accuracy around:
# TSN: 68.8%
# TSM: 71.2%

# test TSN
# python test_models.py kinetics \
#     --weights=pretrained/TSM_kinetics_RGB_resnet50_avg_segment5_e50.pth \
#     --test_segments=8 --test_crops=1 \
#     --batch_size=64

# test TSM
python test_models.py ite \
    --weights=checkpoint/TSM_ite_RGB_resnet50_shift8_blockres_avg_segment8_e50/ckpt.best.pth.tar \
    --test_segments=8 --test_crops=1 \
    --batch_size=8 --csv_file ./test_results/ite_8f_e50_test.csv --test_list /mnt/data/ite_dataset/test_videofolder.txt
