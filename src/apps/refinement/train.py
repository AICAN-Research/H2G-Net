from math import ceil
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from architectures.UNet import Unet
from DualAttentionUNet import DualAttentionUNet
from AttentionUNet import AttentionUNet
from DoubleUNet import build_model
from numpy.random import shuffle
from utils.batch_generator import mpBatchGeneratorCustom
import pandas as pd
from tensorflow.python.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.backend as K
from datetime import datetime, date
from tensorflow.python.keras.engine.training_generator import _make_enqueued_generator, convert_to_generator_like, _get_next_batch
from accum_optimizers import *
import os


if __name__ == "__main__":
    # current date + time
    curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
    curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))

    # use single GPU (first one)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    heatmap_guiding = True  # False
    arch = "unet"  # {unet, dagunet, agunet, doubleunet}
    if arch != "doubleunet":
        batch_size = 6  # 6, 2
        accum_steps = 4  # 8 (4 * 8 = 32 => effective batch size using ordinary batch size BS=4 and accumulated gradients AG=8)
    else:
        batch_size = 4  # 4
        accum_steps = 6  # 6, 8 (4 * 8 = 32 => effective batch size using ordinary batch size BS=4 and accumulated gradients AG=8)
    epochs = 1000
    img_size = 1024
    top = True  # False
    workers = 1  # 8
    use_multiprocessing = False  # False
    max_queue_size = 10  # 10
    use_bn = True  # True
    renorm = True  # False (whether to test BatchReNormalization)
    lr = 1e-4  # 1e-4, 1e-3 default for Adam
    hprob = True  # False
    dataset_type = "HUNT"  # HUNT (merged HUNT0 and HUNT1 datasets into one dataset), HUNT0, HUNT1, Bergen
    best_pw_heatmap = True  # False
    random_split = False  # whether to use predefined data split or generate new one for current training session

    deep_supervision = False
    if arch in ["dagunet", "agunet"]:
        deep_supervision = True

    if arch == "doubleunet":
        deep_supervision = False

    # augmentation
    train_aug = {'flip':1, 'rotate_ll':1, 'stain':1, 'mult':[0.8, 1.2]}
    val_aug = {}

    # set name for model and history
    name = curr_date + "_" + curr_time + "_" + "tumor_refinement_full_" + str(img_size) + "_arch_" + arch + "_heatmap_" + str(heatmap_guiding) + "_hprob_" + str(hprob) + "_bs_" + str(batch_size) + "_ag_" + str(accum_steps) + "_aug_" + \
        str(train_aug).replace(" ", "").replace("'", "").replace("{", "").replace("}", "") + "_" + str(val_aug).replace(" ", "").replace("'", "").replace("{", "").replace("}", "") + "_top_" + str(top) + \
        "_renorm_" + str(renorm) + "_bn_" + str(use_bn) + "_lr_" + str(lr) + "_dataset_" + dataset_type

    print("Training name chosen:", name)

    # fixed paths related to the local setup (need to be changed for future projects)
    gt_path = "/mnt/EncryptedPathology/pathology/phd/P2_AP_samlefil/exported_mask_tif_4_240821/"
    data_path = "/mnt/EncryptedPathology/pathology/datasets/131021_tumor_refinement_full_1024_cluster_True_dataset_type_HUNT_cluster_heatmap_True/"
    chosen_dataset = "dataset_070921_112027_tumor_classify_images_256_10x_vsi_train_augs_flip:1,rotate_ll:1,hsv:20,mult:[0.8,1.2]_model_clustering_dense_100_drp_0.5_lr_0.0001_bs_64_eps_200_classes_[0,1]_wsiSample_True.h5"
    save_model_path = "/mnt/EncryptedPathology/pathology/output/saved_models/"
    history_path = "/mnt/EncryptedPathology/pathology/output/history/"
    datasets_path = "/mnt/EncryptedPathology/pathology/output/datasets/"
    path_to_stata = "/mnt/EncryptedPathology/pathology/phd/ID_grad_histotype.dta"

    if random_split:
        # labels of each tumor
        tmp1 = os.listdir(gt_path)
        tmps = np.array([int(t.split(".tif")[0]) for t in tmp1])
        tmp = pd.read_stata(path_to_stata, convert_categoricals=False)
        ids = tmp["ID_deltaker"]
        grades_vals = tmp["GRAD"]
        out = [a in tmps for a in ids]
        ids = ids[out]
        grades_vals = grades_vals[out]
        grades = np.transpose(np.reshape(np.concatenate([ids, grades_vals]), (2, len(ids)))).astype(int)

        # split dataset into three depending on grade
        grade_split = []
        labs = grades[:, 1]
        for i in np.unique(labs):
            tmp = labs == i
            grade_split.append([i, grades[:,0][tmp]])

        # shuffle each column -> random sets
        for i in range(len(grade_split)):
            tmp = grade_split[i][1]
            shuffle(tmp)
            for j in range(len(tmp)):
                grade_split[i][1][j] = tmp[j]

        # assign two of each -> train,test 2*2*3 = 12 total -> rest training
        val = 5
        test_set = []
        for i in range(len(grade_split)):
            tmp = grade_split[i][1][:val]
            for j in tmp:
                test_set.append(data_path + str(j) + '.h5')

        val_set = []
        for i in range(len(grade_split)):
            tmp = grade_split[i][1][(val):int(2*val)]
            for j in tmp:
                val_set.append(data_path + str(j) + '.h5')

        train_set = []
        for i in range(len(grade_split)):
            tmp = grade_split[i][1][int(2*val):]
            for j in tmp:
                train_set.append(data_path + str(j) + '.h5')

        # save random generated data sets
        f = h5py.File((datasets_path+'dataset_' + name + '.h5'), 'w')
        f.create_dataset("test", data=np.array(test_set).astype('S200'), compression="gzip", compression_opts=4)
        f.create_dataset("val", data=np.array(val_set).astype('S200'), compression="gzip", compression_opts=4)
        f.create_dataset("train", data=np.array(train_set).astype('S200'), compression="gzip", compression_opts=4)
        f.close()
    else:
        curr_dataset = datasets_path + chosen_dataset
        f = h5py.File(curr_dataset, 'r')
        all_sets = []
        for sets in ["train", "val", "test"]:
            tmp = np.array(f[sets])
            tmps = []
            for t in tmp:
                tmp2 = np.array(f[sets + "/" + t])
                tmps.append(tmp2)
            wsi_list = tmps.copy()
            all_sets.append(wsi_list)
        f.close()

        train_dir, val_dir, test_dir = all_sets

        train_set = [data_path + str(s) + ".h5" for x in train_dir for s in x]
        val_set = [data_path + str(s) + ".h5" for x in val_dir for s in x]
        test_set = [data_path + str(s) + ".h5" for x in test_dir for s in x]

    # define model
    if arch == "unet":
        network = Unet(input_shape=(1024, 1024, 3 + int(heatmap_guiding)), nb_classes=2)
        network.encoder_spatial_dropout = None  # 0.1
        network.decoder_spatial_dropout = 0.1  # 0.1
        convs = [8, 16, 32, 64, 128, 128, 256, 256, 512, 256, 256, 128, 128, 64, 32, 16, 8]
        network.set_convolutions(convs)
        network.set_bn(use_bn)
        network.set_renorm(renorm)
        model = network.create()
    elif arch == "dagunet":
        network = DualAttentionUnet(input_shape=(1024, 1024, 3 + int(heatmap_guiding)), nb_classes=2, deep_supervision=True,
                                    input_pyramid=True, attention_guiding=False)
        network.decoder_dropout = 0.1
        convs = [8, 16, 32, 64, 128, 128, 256, 256]
        network.set_convolutions(convs)
        model = network.create()
    elif arch == "agunet":
        network = AttentionUnet(input_shape=(1024, 1024, 3 + int(heatmap_guiding)), nb_classes=2, deep_supervision=True,
                                input_pyramid=True)
        network.decoder_dropout = 0.1
        network.set_renorm(renorm)
        convs = [8, 16, 32, 64, 128, 128, 256, 256]
        network.set_convolutions(convs)
        model = network.create()
    elif arch == "doubleunet":
        encoder_filters = [8, 16, 32, 64, 128, 128, 256, 256]
        decoder_filters = encoder_filters[::-1]
        aspp_filter = 32
        model = build_model(input_shape=(1024, 1024, 3 + int(heatmap_guiding)), nb_classes=2, weights=None,
                            encoder_filters=encoder_filters, decoder_filters=decoder_filters, aspp_filter=aspp_filter)

        # for loss function
        network = Unet(input_shape=(1024, 1024, 3 + int(heatmap_guiding)), nb_classes=2)
    else:
        raise("Model architecture chosen was: " + arch + ", which is not of supported type. Please choose either of these archs: {unet, agunet, dagunet}")

    print(model.summary())

    # load model <- if want to fine-tune, or train further on some previously trained model
    #model.load_weights('/home/andre/Documents/Project/Andrep/lungmask/output/model_3d_11_01.h5', by_name=True)

    if arch == "unet":
        model.compile(
            optimizer=AdamAccumulated(accumulation_steps=accum_steps, learning_rate=lr),
            loss=network.get_dice_loss(),
            metrics=[network.get_dice_metric()],
        )
    elif arch == "doubleunet":
        model.compile(
            optimizer=AdamAccumulated(accumulation_steps=accum_steps, learning_rate=lr),
            loss=network.get_dice_loss(),
            metrics=[network.get_dice_metric()],
            loss_weights=[1 / 2] * 2  # two U-Nets, one loss for each output
        )
    else:
        model.compile(
            optimizer=AdamAccumulated(accumulation_steps=accum_steps, learning_rate=lr),
            loss=network.get_dice_loss_no_bg(),
            metrics=[network.get_dice_metric()],
            loss_weights=[1 / 7] * 7  # @TODO: This should be dynamically set
        )

    train_length = batch_length(train_set)
    val_length = batch_length(val_set)

    train_gen = mpBatchGeneratorCustom(train_set, batch_size=batch_size, aug=train_aug, N=train_length, classes=None, max_q_size=20, max_proc=8, heatmap_guiding=heatmap_guiding,
                                    deep_supervision=deep_supervision, hprob=hprob, arch=arch)
    val_gen = mpBatchGeneratorCustom(val_set, batch_size=batch_size, aug=val_aug, N=val_length, classes=None, max_q_size=20, max_proc=8, heatmap_guiding=heatmap_guiding,
                                    deep_supervision=deep_supervision, hprob=hprob, arch=arch)

    if top:
        if arch == "dagunet":
            save_best = ModelCheckpoint(
                save_model_path + '/unet_tumor_' + name + '.h5',
                monitor='val_conv2d_50_loss',
                # 'val_loss',  # @ TODO: Currently, monitoring val_loss, but should monitor top loss ONLY. Not relevant for UNet, but for DS models (multi-output)
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                mode='auto',
                period=1
            )

            early = EarlyStopping(
                monitor='val_conv2d_50_loss',
                min_delta=0,
                patience=100,
                verbose=1,
                mode="auto",
            )
        elif arch == "agunet":
            save_best = ModelCheckpoint(
                save_model_path + '/unet_tumor_' + name + '.h5',
                monitor='val_conv2d_63_loss',
                # 'val_loss',  # @ TODO: Currently, monitoring val_loss, but should monitor top loss ONLY. Not relevant for UNet, but for DS models (multi-output)
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                mode='auto',
                period=1
            )

            early = EarlyStopping(
                monitor='val_conv2d_63_loss',
                min_delta=0,
                patience=100,
                verbose=1,
                mode="auto",
            )

    if (arch in ["unet", "doubleunet"]) or (not top):
        curr_loss = 'val_loss' if arch == "unet" else 'val_refinement_dice'
        curr_mode = "auto" if arch == "unet" else "max"
        print("current monitor metric/loss: ", curr_loss)
        save_best = ModelCheckpoint(
            save_model_path + '/unet_tumor_' + name + '.h5',
            monitor=curr_loss,  # @ TODO: Currently, monitoring val_loss, but should monitor top loss ONLY. Not relevant for UNet, but for DS models (multi-output)
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode=curr_mode,
            period=1
        )

        early = EarlyStopping(
            monitor=curr_loss,
            min_delta=0,
            patience=100,
            verbose=1,
            mode=curr_mode,
        )


    # history logging
    csv_logger = CSVLogger(history_path + "history_" + name + ".csv", append=True, separator=';')

    # start training
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=int(ceil(train_length/batch_size)),
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=int(ceil(val_length/batch_size)),
        callbacks=[save_best, csv_logger, early],
        use_multiprocessing=use_multiprocessing,
        workers=workers,
        max_queue_size=max_queue_size
    )
