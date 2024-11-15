<<<<<<< Updated upstream
python train_all.py exp_name --dataset PACS --data_dir  --trial_seed 0 --algorithm ERM --checkpoint_freq 100 --alpha 0.001 --lr 3e-5 --weight_decay 1e-4 --resnet_dropout 0.5 --swad False
=======
python train_all.py exp_name --dataset PACS_Generated --data_dir "D:\Datasets\PACS_augmented.zip\diffuseMix_augmented" --steps=5000 --trial_seed 0 --algorithm ERM_DiffuseMix --checkpoint_freq 100 --lr 3e-5 --weight_decay 1e-4 --resnet_dropout 0.5 --swad False
>>>>>>> Stashed changes
