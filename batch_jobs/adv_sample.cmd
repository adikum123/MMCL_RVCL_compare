python ssl_adv_sample.py \
    --model cnn_4layer_b \
    --encoder_checkpoint models/mmcl/rbf/cnn_4layer_b_C_1_rbf_auto.pkl \
    --linear_eval_checkpoint models/linear_evaluate/linear_cnn_4layer_b_C_1_rbf_auto.pkl \
    --dataset cifar-10 \
    --batch_size 256 \
    --kernel_type rbf \
    --num_iters 200 \
    --lr 1e-3 \
    --train_type test