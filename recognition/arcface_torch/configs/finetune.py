from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "cosface"
config.network = "r34"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.01  # batch size is 512
config.rec = "/data/users/yangqiancheng/datasets/oppo_cropped"
config.num_classes = 425
config.num_image = 2000
config.num_epoch = 40
config.warmup_epoch = -1
config.decay_epoch = [8, 12, 15, 18]
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
