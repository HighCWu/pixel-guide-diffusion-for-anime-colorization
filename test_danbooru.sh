
MODEL_FLAGS="--image_size 32 --guide_size 128 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.0"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TEST_FLAGS="--batch_size 4"

OPENAI_LOGDIR="./danbooru2017_guided_test_log" python scripts/pixel_guide_sample.py --data_dir data/danbooru2017/anime --guide_dir data/danbooru2017/anime_sketch --timestep_respacing ddim25 --use_ddim True --model_path danbooru2017_guided_log/ema_0.9999_360000.pt $MODEL_FLAGS $DIFFUSION_FLAGS $TEST_FLAGS
