import argparse
import os


def fixModel (cfg):
    print (cfg.model.type)
    if cfg.model.type == "CascadeRCNN":
        for k in range(len(cfg.model.roi_head.bbox_head)):
            cfg.model.roi_head.bbox_head[k]["num_classes"] = 1
    elif cfg.model.type == "TOOD":
        cfg.model.bbox_head["num_classes"] = 1
    elif cfg.model.type == "VFNet":
        cfg.model.bbox_head["num_classes"] = 1
    else:
        raise Exception ("Unknown model, cannot adapt number of classes.")
    return cfg


def fixTestData (cfg, args):
    # this is for testing only, i.e. FINAL testing
    testFile = "#"*50
    assert (args.fold == -1)
    assert (args.testset == args.testset)

    trainFile = None
    testFile = f"./annotations/{args.testset}_{args.margin}.json" # fake, we dont need this because there is no early stopping or whatever
    valFile = testFile

    cfg.train_dataloader.dataset.data_root = "."
    cfg.train_dataloader.dataset.ann_file = trainFile
    cfg.val_dataloader.dataset.data_root = "."
    cfg.val_dataloader.dataset.ann_file = valFile
    cfg.test_dataloader.dataset.data_root = "."
    cfg.test_dataloader.dataset.ann_file = testFile
    cfg.val_evaluator.ann_file = valFile
    cfg.test_evaluator.ann_file = testFile
    return cfg


def fixTrainData (cfg, args):
    # data
    trainFile = None
    if args.fold < 0:
        trainFile = f"./annotations/train_{args.margin}.json"
        valFile = f"./annotations/train_val_fold_0_{args.margin}.json" # fake, we dont need this because there is no early stopping or whatever
        testFile = f"./annotations/train_val_fold_0_{args.margin}.json" # fake, we dont need this because there is no early stopping or whatever
    else:
        trainFile =  f"./annotations/train_train_fold_{args.fold}_{args.margin}.json"
        valFile = f"./annotations/train_val_fold_{args.fold}_{args.margin}.json"
        testFile = f"./annotations/train_val_fold_{args.fold}_{args.margin}.json"

    cfg.train_dataloader.dataset.data_root = "."
    cfg.train_dataloader.dataset.ann_file = trainFile
    cfg.val_dataloader.dataset.data_root = "."
    cfg.val_dataloader.dataset.ann_file = valFile
    cfg.test_dataloader.dataset.data_root = "."
    cfg.test_dataloader.dataset.ann_file = testFile
    cfg.val_evaluator.ann_file = valFile
    cfg.test_evaluator.ann_file = testFile
    return cfg


#
