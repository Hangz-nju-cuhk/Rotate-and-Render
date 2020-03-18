python -u train.py  \
        --name rs_model \
        --dataset_mode allface \
        --load_size 400 \
        --crop_size 256 \
        --gpu_ids 0,1,2,3,4,5,6,7 \
        --netG rotatespade \
        --trainer rotatespade \
        --norm_D spectralsyncbatch \
        --norm_G spectralsyncbatch \
        --batchSize 3 \
        --model rotatespade \
        --dataset 'example' \
        --lambda_D 0.75 \
        --lambda_rotate_D 0.001 \
        --D_input concat \
        --netD multiscale \
        --label_nc 5 \
        --nThreads 3 \
        --no_html \
        --load_separately \
        --heatmap_size 2.5 \
        --heatmap_size 2.5 \
        --device_count 8 \
        --render_thread 3 \
        --chunk_size 1 1 1 1 1 \
        --no_gaussian_landmark \
        --landmark_align \
        --erode_kernel 15 \
        --pose_noise \
        # --face_vgg \
        # --G_pretrain_path ./checkpoints/latest_net_G.pth \
        # --D_pretrain_path ./checkpoints/latest_net_D.pth \
