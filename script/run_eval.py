import os

run_names = ["romulan-phaser-63", "cool-field-64", "whole-fire-70", "whole-fire-70", "pretty-durian-73", "desert-dust-75"]
enc_downsample = [(3,2), (3,2,2), (3,2,2), (3,2,2), (3,2,2,2), (3,2,2,2)]
mask_span = [5, 5, 3, 5, 6, 3]

for run_name, enc_downsample, mask_span in zip(run_names, enc_downsample, mask_span):
    print("Running ", run_name, "with mask span ", mask_span)
    os.system(f"python evaluate_bendr.py --dataset Spampy --run_name {run_name} --enc_downsample {' '.join(map(str, enc_downsample))} --mask_span {mask_span}")
    print("FINISHED RUNNING ", run_name, "with mask span ", mask_span)
    print()