python roberta/model.py \
--gradient_accumulation_steps 4 \
--per_gpu_train_batch_size 4 \
--num_train_epochs 50 \
--learning_rate 1e-5 \
--adam_epsilon 1e-6 \
--warmup_ratio 0.06 \
--weight_decay 0.1 \
--print_loss_step 50

