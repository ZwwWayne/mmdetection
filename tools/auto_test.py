import argparse
import json
import os
import subprocess
import traceback

import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet.core import coco_eval, results2json
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results, prog_bar.fps


def parse_results(json_log):
    last_line = subprocess.check_output(['tail', '-1', json_log])
    last_line = json.loads(last_line)
    metrics = dict()
    for k, v in last_line.items():
        if k in ('AR@100', 'bbox_mAP', 'segm_mAP'):
            metrics[k] = v
    return metrics


parser = argparse.ArgumentParser()
parser.add_argument('model_path')
args = parser.parse_args()

if os.path.exists('valid_results.json'):
    print('Please remove or rename the original valid_results.json')
    exit()

#  check if all models have corresponding configs
print('parse model infos')
print('-' * 40)
model_infos = []
for model_family in os.listdir(args.model_path):
    model_family_dir = os.path.join(args.model_path, model_family)
    for model in os.listdir(model_family_dir):
        model_info = dict()
        model_dir = os.path.join(model_family_dir, model)
        if model_family == 'configs':
            config = os.path.join(model_family, model)
        else:
            config = os.path.join('configs', model_family, model)

        records = os.listdir(os.path.join(model_dir))
        config = [r for r in records if r[-2:] == 'py']
        assert 0 < len(config) <= 1, 'check {} fails'.format(model_dir)
        config = os.path.join(model_dir, config[0])

        cpt = [r for r in records if r[-3:] == 'pth']
        assert 0 < len(cpt) <= 1, 'check {} fails'.format(model_dir)
        cpt = os.path.join(model_dir, cpt[0])

        log = [r for r in records if r[-4:] == 'json']
        assert 0 < len(log) <= 1, 'check {} fails'.format(model_dir)
        log = os.path.join(model_dir, log[0])

        eval_results = parse_results(log)
        model_info['config'] = config
        model_info['checkpoint'] = cpt
        model_info['train_results'] = eval_results
        model_infos.append(model_info)

print('collect total: {} models'.format(len(model_infos)))
print()

print('start valid models')

for i, model_info in enumerate(model_infos):
    try:
        config = model_info['config']
        cpt = model_info['checkpoint']
        print('valid [{}/{}] model'.format(i + 1, len(model_infos)))
        print('config: {}'.format(config))
        print('checkpoint: {}'.format(cpt))
        print('-' * 40)

        valid_results = dict()
        valid_results['config'] = config
        valid_results['checkpoint'] = cpt
        for k, v in model_info['train_results'].items():
            valid_results['train_{}'.format(k)] = v

        cfg = mmcv.Config.fromfile(config)
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True
        distributed = False
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        checkpoint = load_checkpoint(model, cpt, map_location='cpu')
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        model = MMDataParallel(model, device_ids=[0])
        outputs, fps = single_gpu_test(model, data_loader)

        eval_types = []
        train_metrics = model_info['train_results']
        if 'AR@100' in train_metrics.keys():
            eval_types.append('proposal_fast')
        if 'bbox_mAP' in train_metrics.keys():
            eval_types.append('bbox')
        if 'segm_mAP' in train_metrics.keys():
            eval_types.append('segm')

        mmcv.mkdir_or_exist('tmp')
        mmcv.dump(outputs, 'tmp/tmp.pkl')

        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = 'tmp/tmp.pkl'
                eval_results = coco_eval(result_file, eval_types, dataset.coco)
            else:
                if not isinstance(outputs[0], dict):
                    result_files = results2json(dataset, outputs,
                                                'tmp/tmp.pkl')
                    eval_results = coco_eval(result_files, eval_types,
                                             dataset.coco)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = args.out + '.{}'.format(name)
                        result_files = results2json(dataset, outputs_,
                                                    result_file)
                        eval_results = coco_eval(result_files, eval_types,
                                                 dataset.coco)

        for k, v in eval_results.items():
            valid_results['valid_{}'.format(k)] = v
        valid_results['inf_speed'] = round(fps, 1)
    except Exception:
        traceback.print_exc()
        valid_results['valid'] = None
        del model
        torch.cuda.empty_cache()

    with open('valid_results.json', 'a+') as f:
        mmcv.dump(valid_results, f, file_format='json')
        f.write('\n')
    print()
