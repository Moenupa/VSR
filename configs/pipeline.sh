# python tools/test.py configs/edvr.py out2/edvrm_wotsa_x4_g8_60k_stm3k/iter_60000.pth --seed 3407 --out data/STM3k/test30/edvr/results.pkl --save-path data/STM3k/test30/edvr
# python tools/test.py configs/basicvsr.py out2/basicvsr_stm_test/iter_60000.pth --seed 3407 --out data/STM3k/test30/basicvsr/results.pkl --save-path data/STM3k/test30/basicvsr
# python tools/train.py configs/esrgan.py --seed 3407 --resume-from work_dirs/esrgan_x4c64b23g32_g1_100k_stm3k/iter_40000.pth
python tools/train.py configs/ganbasicvsr.py --seed 3407
python tools/test.py configs/ganbasicvsr.py out/ganbasicvsr_c64b20_1x30x8_lr5e-5_30k_stm3k/iter_30000.pth --seed 3407 --out data/STM3k/test30/ganbasicvsr/results.pkl --save-path data/STM3k/test30/ganbasicvsr