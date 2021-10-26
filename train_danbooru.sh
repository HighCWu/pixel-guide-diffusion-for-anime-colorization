
MODEL_FLAGS="--image_size 32 --guide_size 128 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.0"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--use_fp16 True --lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"

OPENAI_LOGDIR="./danbooru2017_guided_log" python scripts/pixel_guide_train.py --data_dir data/danbooru2017/anime --guide_dir data/danbooru2017/anime_sketch $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
