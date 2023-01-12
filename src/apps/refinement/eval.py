import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
import cv2
from tqdm import tqdm
from numpy.random import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support, recall_score, precision_score
import pandas as pd
from skimage.morphology import binary_closing, binary_opening, disk
from architectures.DualAttentionUNet import DualAttentionUNet
from medpy.metric.binary import hd95
import matplotlib as mpl
import openslide as ops
import matplotlib.colors as mcolors
import csv


def maxminscale(tmp):
    if (len(np.unique(tmp)) > 1):
        tmp = tmp - np.amin(tmp)
        tmp = tmp / np.amax(tmp)
    return tmp


def remove_copies(tmp):
    return np.unique(tmp).tolist()


def import_set(tmp, num=None, filter=False):
    with h5py.File(datasets_path + 'dataset_' + name + '.h5', 'r') as f:
        tmp = np.array(f[tmp])
        tmp = [tmp[i].decode("UTF-8") for i in range(len(tmp))]
        shuffle(tmp)
        if filter:
            tmp = remove_copies(tmp)
        if num != None:
            tmp = tmp[:num]
    return tmp


def DSC(pred, gt, smooth=0.):
    intersection1 = np.sum(pred * gt)
    union1 = np.sum(pred * pred) + np.sum(gt * gt)
    return (2. * intersection1 + smooth) / (union1 + smooth)


if __name__ == "__main__":
    # some pretrained model name
    name = "unet_tumor_181021_105110_tumor_refinement_full_1024_arch_unet_heatmap_True_hprob_True_bs_6_ag_4_aug_flip:1,rotate_ll:1,stain:1,mult:[0.8,1.2]__top_True_renorm_True_bn_True_lr_0.0001_dataset_HUNT"  # unet

    # setup GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    tissue_pred_flag = False
    if name == "tissue":
        tissue_pred_flag = True

    dataset_type = "HUNT"  # HUNT, HUNT0, HUNT1, Bergen
    lowres_model = False  # False
    patch_model = False  # False: Then will assume a refinement network has been used
    eval_res_flag = True
    tissue_flag = False  # False : To fix/filter GT based on tissue mask
    plot_flag = False  # False
    savefig_flag = False  # True
    filter_glass_gt = False  # False
    threshold_method = 1  # 1 if Otsu, else predefined (simple thresholding)

    # parse architecture used from model name
    arch = name.split("arch_")[-1].split("_")[0]

    if tissue_pred_flag:
        patch_model = False

    heatmap_guiding = False
    if (not patch_model) and (not lowres_model) and (not tissue_pred_flag):
        heatmap_guiding = eval(name.split("heatmap_")[-1].split("_")[0])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("heatmap guiding:", heatmap_guiding)
    print("filter glass gt:", filter_glass_gt)

    # training data
    raw_wsi_path = "/mnt/EncryptedPathology/pathology/images_tiff_tz_1024_jpeg_85/"

    # paths
    gt_path = "/mnt/EncryptedPathology/pathology/phd/P2_AP_samlefil/exported_mask_tif_4_240821/"
    data_path = "/mnt/EncryptedPathology/pathology/datasets/131021_tumor_refinement_full_1024_cluster_True_dataset_type_HUNT_cluster_heatmap_True/"
    chosen_dataset = "dataset_070921_112027_tumor_classify_images_256_10x_vsi_train_augs_flip:1,rotate_ll:1,hsv:20,mult:[0.8,1.2]_model_clustering_dense_100_drp_0.5_lr_0.0001_bs_64_eps_200_classes_[0,1]_wsiSample_True.h5"
    save_model_path = "/mnt/EncryptedPathology/pathology/output/saved_models/"
    history_path = "/mnt/EncryptedPathology/pathology/output/history/"
    results_path = "/mnt/EncryptedPathology/pathology/output/results/"
    datasets_path = "/mnt/EncryptedPathology/pathology/output/datasets/"
    saved_path = "/mnt/EncryptedPathology/pathology/output/predictions/" + name + "/"

    # labels of each tumor
    tmp1 = os.listdir(gt_path)
    tmps = np.array([int(t.split(".tif")[0]) for t in tmp1])
    path_to_stata = "/mnt/EncryptedPathology/pathology/phd/ID_grad_histotype.dta"
    tmp = pd.read_stata(path_to_stata, convert_categoricals=False)
    ids = tmp["ID_deltaker"]
    grades_vals = tmp["GRAD"]
    out = [a in tmps for a in ids]  # [a in B for a in A]
    ids = ids[out]
    grades_vals = grades_vals[out]
    grades = np.transpose(np.reshape(np.concatenate([ids, grades_vals]), (2, len(ids)))).astype(int)

    # skip specific one which does not have been given grade {1, 2, 3}
    grades = grades[grades[:, 0] != 76]

    # load pregenerated dataset split
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

    sets = test_set  # chose which split to use
    np.random.shuffle(sets)  # shuffle these -> does not impact result, just for debbugging

    # threshold for binarization of prediction
    th = 0.5

    # load trained model
    # define model
    if (not patch_model):
        if not tissue_pred_flag:
            model = load_model(save_model_path + name + ".h5", compile=False, custom_objects={"PAM": PAM, "CAM": CAM})
    
    recalls = []
    precisions = []
    f1s = []
    hd95s = []
    f1s_grades = [[], [], []]
    wsi_grades = []
    dsc_list = []
    dsc_tissue_list = []
    res1 = [[], []]
    res2 = [[], []]
    for cnt, path in enumerate(tqdm(sets)):
        curr_id = int(path.split("/")[-1].split(".")[0])
        curr_grade = grades[grades[:, 0] == curr_id, 1][0] - 1
        curr_saved_path = saved_path + str(curr_id) + ".h5"  # only relevant for patch models

        if eval_res_flag:
            # get low-res WSI of current patient at a specific magnification plane
            curr_plane = 5  # plane to assess performance (1 micron resolution)
            try:
                raw_wsi_reader = ops.OpenSlide(raw_wsi_path + str(curr_id) + ".tif")
            except Exception:
                print("something wrong with WSI...")
                continue
            raw_wsi = np.array(raw_wsi_reader.read_region((0, 0), curr_plane, raw_wsi_reader.level_dimensions[curr_plane]))[..., :3]
            raw_wsi_reader.close()

            # get corresponding GT at the same magnification level as the WSI
            gt_plane = curr_plane - 2
            try:
                raw_gt_reader = ops.OpenSlide(gt_path + str(curr_id) + ".tif")
                raw_gt = np.array(raw_gt_reader.read_region((0, 0), gt_plane, raw_gt_reader.level_dimensions[gt_plane]))[..., 0]
            except Exception:
                print("something wrong with GT...")
                continue
            raw_gt_reader.close()

            if np.count_nonzero(raw_gt) == 0:
                print("The GT was empty! No tumour annotated...")
                continue

            raw_gt = (raw_gt > 127).astype(np.float32)

        if patch_model:
            with h5py.File(curr_saved_path, 'r') as f:
                pred = np.array(f["pred"])

            pred[pred == 1] = 0
            pred[pred > 0] = 1

            data = raw_wsi.copy()
            gt = raw_gt.copy()
            tmp = data.copy()

            if plot_flag:
                fig, ax = plt.subplots(1, 3)
                ax[0].imshow(data)
                ax[1].imshow(pred)
                ax[2].imshow(gt)
                plt.show()
        else:
            with h5py.File(path, 'r') as f:
                data = np.array(f['data'])
                gt = np.array(f['label']).astype(int)
                if heatmap_guiding:
                    heatmap = np.array(f['heatmap'])
                    data = np.concatenate([data, heatmap], axis=-1)

            tmp = data[0]

        wsi_grades.append(curr_grade)

        # automatic tissue segmentation
        tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2HSV)
        tmp = tmp[..., 1]
        if threshold_method == 1:
            _, tissue = cv2.threshold((tmp * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # otsu
            tissue = (tissue > 0).astype(int)
            tissue = binary_closing(tissue, selem=disk(5)).astype(int)
        else:
            tissue = ((tmp * 255).astype(np.uint8) >= 20).astype(int)  # simple thresholding using predefined threshold
            tissue = binary_closing(tissue, selem=disk(5)).astype(int)
        tissue_orig = (tissue > 0).astype(int)
        tissue = binary_closing(tissue, selem=disk(3)).astype(int)

        if patch_model:
            pass
        else:
            if tissue_pred_flag:
                pred = tissue_orig.copy()
            else:
                if lowres_model:
                    print(data.shape)
                    pred = model.predict(data)
                else:
                    # if not patch model, actually run inference
                    pred = model.predict(data)

                    if arch == "doubleunet":
                        pred = pred[1]

                    # if arch != "unet": I should only get the first output
                    if arch not in ["unet", "doubleunet"]:
                        pred = pred[0]

                    if tissue_flag:
                        data = tissue.copy()

                data = data[0]
                pred = pred[0, ..., 1]
                gt = gt[0, ..., 1]

        pred_bin = (pred >= th).astype(int)

        if filter_glass_gt:
            # new gt, filter glass in pred and gt (only focus on tissue)
            pred_bin *= tissue
            gt *= tissue

        if not eval_res_flag:
            raw_wsi = data.copy()
            raw_gt = gt.copy()
        else:
            pred_bin = cv2.resize(pred_bin.astype(np.uint8), raw_wsi.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
            pred_bin = (pred_bin > 0.5).astype(np.float32)

        # calculate Dice
        dsc = DSC(pred_bin, raw_gt)
        dsc_list.append(dsc)
        res1[0].append(pred_bin)
        res1[1].append(gt)

        # calculate Recall, Precision, F1
        true_positives = np.sum(pred_bin * raw_gt)
        possible_positives = np.sum(raw_gt)
        predicted_positives = np.sum(pred_bin)
        pr_ = true_positives / possible_positives
        rec_ = true_positives / predicted_positives
        f1_ = 2 * (pr_ * rec_ / (pr_ + rec_))

        ret = [pr_, rec_, f1_]

        precisions.append(ret[0])
        recalls.append(ret[1])
        f1s.append(ret[2])
        f1s_grades[curr_grade].append(ret[2])

        # calculate the symmetric Hausdorff Distance (HD)
        hd95_ = 0  # hd95(pred_bin, raw_gt, voxelspacing=None, connectivity=1)
        hd95s.append(hd95_)

        names = ["WSI", "heatmap", "tissue", "conf", "pred, DSC: " + str(np.round(dsc, 4)), "gt"]

        if plot_flag:
            mpl.rc("font", size=8, **{'family': 'serif'})

            # create colormaps
            tissue_cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', [(1, 1, 1, 1), (0, 0, 1, 1)], N=2)
            tumor_cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', [(1, 1, 1, 1), (0, 1, 0, 1), (1, 0, 0, 1)], N=100)
            bin_tumor_cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', [(1, 1, 1, 1), (1, 0, 0, 1)], N=2)
            tumor_prob_cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', [(1, 1, 1, 1), (1, 0, 0, 1)], N=100)

            test = data[..., -1]
            test = (test >= 0.5).astype(np.bool)
            test = binary_closing(test, selem=disk(5)).astype(np.bool)
            test = binary_opening(test, selem=disk(5)).astype(np.float32)

            heatmap = data[..., -1]
            heatmap = cv2.resize(heatmap, raw_wsi.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            pred_bin = cv2.resize(pred, raw_wsi.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

            heatmap_prob = heatmap.copy()

            heatmap_bin = (heatmap >= 0.5).astype(np.float32)

            heatmap[heatmap != 0] += 1
            heatmap[(raw_gt == 1) & (heatmap >= 0.5)] += 1
            heatmap[(raw_gt == 1) & ((heatmap < 0.5))] += 0.5

            some_names = ["wsi", "heatmap", "ground truth", "glass, Dice: " + str(np.round(DSC(tissue, gt), 4)), "näive, Dice: " + str(np.round(DSC(test, gt), 4)), "proposed, Dice: " + str(np.round(dsc, 4))]

            fig, ax = plt.subplots(2, 3, gridspec_kw={'wspace': 0.15, 'hspace': 0.1}, squeeze=True)
            ax[0, 0].imshow(raw_wsi[..., :3])
            ax[0, 1].imshow(heatmap, cmap=tumor_cmap, vmin=0, vmax=3)
            ax[0, 2].imshow(raw_gt, cmap=bin_tumor_cmap)
            ax[1, 0].imshow(tissue, cmap=tissue_cmap)
            ax[1, 1].imshow(test, cmap=bin_tumor_cmap)
            ax[1, 2].imshow(pred_bin, cmap=bin_tumor_cmap)

            cnt = 0
            for i, n in enumerate(some_names):
                ax[cnt, int(i % 3)].set_title(some_names[i])
                ax[cnt, int(i % 3)].axis("off")
                if i == 2:
                    cnt += 1

            plt.tight_layout()
            if savefig_flag:
                fig.savefig("/home/andrep/workspace/pathology/output/tumor_seg_results/" + str(curr_id) + ".png", dpi=900, bbox_inches="tight")
            else:
                plt.show()

        res2[0].append(pred_bin)
        res2[1].append(gt)

    print()
    print("#"*80, "\n")

    print("Which model is used:", name)
    print("Total number of WSIs considered:", len(dsc_list))

    print('average DSC')
    print(np.mean(dsc_list), np.std(dsc_list))
    #print(np.mean(dsc_tissue_list), np.std(dsc_tissue_list))

    print("Recall, Precision, F1:")
    print(np.mean(recalls), np.std(recalls))
    print(np.mean(precisions), np.std(precisions))
    print(np.mean(f1s), np.std(f1s))

    print("95% HD:")
    print(np.mean(hd95s), np.std(hd95s))

    # DSC for each grade
    f1s_grades = np.array(f1s_grades)
    print("F1 for each grade:")
    print([str(np.mean(x)) + " +- " + str(np.std(x)) + " (" + str(len(x)) + ")" for x in f1s_grades])

    # save results to CSV
    names = ["Recall", "Precision", "F1", "HD95", "Grade"]
    results = [recalls, precisions, f1s, hd95s, wsi_grades]

    results = np.stack(results, axis=1)
    results = np.concatenate([np.expand_dims(names, axis=0), results], axis=0)

    np.savetxt(results_path + "eval_results_" + name + "_" + dataset_type + ".csv", results, fmt='%s', delimiter=",", newline="\n")
