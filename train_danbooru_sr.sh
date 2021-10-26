
MODEL_FLAGS="--large_size 128 --small_size 32 --guide_size 128 --num_channels 64 --num_res_blocks 3 --use_attention False --learn_sigma True --dropout 0.0"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--crop_size 32 --use_fp16 True --lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"

OPENAI_LOGDIR="./danbooru2017_guided_sr_log" python scripts/pixel_guide_super_res_train.py --data_dir data/danbooru2017/anime --guide_dir data/danbooru2017/anime_sketch $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
