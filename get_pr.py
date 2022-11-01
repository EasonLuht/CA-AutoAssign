import os
import mmcv
import numpy as np
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmcv import Config
from mmdet.datasets import build_dataset


CONFIG_FILE = [f"work_dirs_9/Autoassign_swin_AGM_CSM/autoassign_swin_Channel_AGM/CA-AutoAssign.py", f"works_dir_origin/autoassign/autoassign_swin/AutoAssign_Swin.py", f"work_dirs_9/TianChi/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco/Sparse R-CNN.py"
    , f"work_dirs_9/cascade_rcnn_r50_fpn_20e_coco/Cascade R-CNN.py", f"work_dirs/tood_r50_fpn_mstrain_2x_coco_tianchi/TOOD.py"]
RESULT_FILE = [f"results/CA-AutoAssign.pkl", f"results/AutoAssign_swin.pkl", f"results/Sparse_rcnn.pkl", f"results/Cascade_rcnn.pkl", f"results/TOOD.pkl"]

font1 = {
    # 'family': 'Times New Roman',
    'weight': 'normal',
    'size': 14,
}

def plot_pr_curve(config_files, result_files, metric="bbox"):
    """plot precison-recall curve based on testing results of pkl file.

        Args:
            config_files (list[list | tuple]): config file path.
            result_files (str): pkl file of testing results path.
            metric (str): Metrics to be evaluated. Options are
                'bbox', 'segm'.
    """
    n = len(config_files)
    for i in range(n):
        config_file = config_files[i]
        result_file = result_files[i]
        name = config_file.split('/')[-1].split('.')[0]
        cfg = Config.fromfile(config_file)
        # turn on test mode of dataset
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True

        # build dataset
        dataset = build_dataset(cfg.data.test)
        # load result file in pkl format
        pkl_results = mmcv.load(result_file)
        # convert pkl file (list[list | tuple | ndarray]) to json
        json_results, _ = dataset.format_results(pkl_results)
        # initialize COCO instance
        coco = COCO(annotation_file=cfg.data.test.ann_file)
        coco_gt = coco
        coco_dt = coco_gt.loadRes(json_results[metric])
        # initialize COCOeval instance
        coco_eval = COCOeval(coco_gt, coco_dt, metric)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # extract eval data
        precisions = coco_eval.eval["precision"]
        '''
        precisions[T, R, K, A, M]
        T: iou thresholds [0.5 : 0.05 : 0.95], idx from 0 to 9
        R: recall thresholds [0 : 0.01 : 1], idx from 0 to 100
        K: category, idx from 0 to ...
        A: area range, (all, small, medium, large), idx from 0 to 3 
        M: max dets, (1, 10, 100), idx from 0 to 2
        '''

        pr_array6 = precisions[0, :, 5, 0, 2]
        # pr_array6 = precisions[0, :, 6, 0, 2]
        # pr_array6 = precisions[0, :, :, 0, 2]
        # pr_array6 = np.mean(pr_array6, axis=-1)


        x = np.arange(0.0, 1.01, 0.01)
        # plot PR curve

        plt.plot(x, pr_array6, label=name)

    plt.xlabel("recall", font1)
    plt.ylabel("precison", font1)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.show()
    # plt.savefig("results/AP_50.png", dpi=1000)


if __name__ == "__main__":
    plot_pr_curve(config_files=CONFIG_FILE, result_files=RESULT_FILE, metric="bbox")
