import json
import argparse
import os
import math
import pickle
import sys

from PIL import Image
import evaluate
import torch
import numpy as np
from tqdm import tqdm
from pycocoevalcap.cider.cider import Cider

from model.lumina_mgpt.inference_solver import FlexARInferenceSolver
from model.lumina_mgpt.data.item_processor import FlexARItemProcessor
from dataset.gen_data import QA
from configs.nusc import tasks_config, generation_configs, do_generate

from utils import encode_mask, decode_plans, process_prompt
from planning_utils import PlanningMetric
from metrics import calc_l2, eval_qa

def main(args):
    anno_path = args.anno_path
    nusc_path = args.nusc_path
    save_path = args.save_path
    model_path = args.model_path
    data_path = args.data_path
    task = args.task

    with open(data_path, 'r') as f:
        data = json.load(f) # [[conv[qas]]]

    div = len(data) // args.split

    def qa_from_dict(d):
        qa = QA()
        qa.from_dict(d)
        return qa
    
    inferencer = FlexARInferenceSolver(
        model_path=model_path,
        precision='bf16'
    )
    processor = FlexARItemProcessor()
    
    inferencer.model.eval()
    logs = list()
    mask_tokens, mask_ids = encode_mask(processor)

    with torch.no_grad():
        for i, convs in tqdm(enumerate(data[div*args.id: div*(args.id+1)])):
            past_tokens, past_key_values = None, None
            try:
                for j, conv in enumerate(convs):
                    qas = list(map(qa_from_dict, conv)) # [QAs]
                    # print(qas)

                    for qa in qas:
                        # print(qa.q, qa.a)

                        if not tasks_config[task][qa.task][j]:
                            continue
                        
                        if do_generate[task][qa.task][j]:
                            # print('generating')
                            generated, past_tokens, past_key_values = inferencer.generate(
                                qas=[[qa.q, None]], 
                                locs=qa.coords[0],
                                past_key_values=past_key_values,
                                past_tokens=past_tokens,
                                logits_processor=inferencer.create_logits_processor(generation_configs[qa.task]['processor']),
                                **generation_configs[qa.task]['settings'],
                            )
                            
                            if qa.task == 'plan':
                                plans = '' + generated[0]
                                past_tokens = torch.cat((past_tokens, mask_ids), dim=1)
                                # argmax for example
                                ans_ids = inferencer.model(past_tokens).logits.argmax(-1).squeeze(0)
                                ans_ids = ans_ids[-27:-2].tolist()
                                plan_tokens = processor.tokenizer.tokenizer.decode(ans_ids)
                                plans += plan_tokens
                                plans = decode_plans(plans)
                                # log id for metric
                                logs.append({'plan': plans, 'gt': qa.plan, 'id': qa.id})
                            
                            elif qa.task == 'image' or qa.task == 'init':
                                generated[1][0].save(os.path.join(save_path, f'{i}-{j}.jpg'))
                            
                            else:
                                logs.append({'generate': generated[0], 'gt': qa.a, 'gt_locs': qa.coords[1], 'task': qa.task})
                                
                            # TODO: decode coords
                            
                                
                        else:
                            # print(qa.q, qa.a)
                            if qa.task == 'plan':
                                new_tokens = process_prompt(
                                    processor, [[qa.q, qa.a+mask_tokens]], 
                                    images=qa.image, 
                                    plans=[[qa.plan[0][0]]], 
                                    locs=qa.coords[0]+qa.coords[1], 
                                    bos=(qa.task=='init')
                                )     

                            else:
                                new_tokens = process_prompt(
                                    processor, [[qa.q, qa.a]], 
                                    images=qa.image, 
                                    plans=qa.plan, 
                                    locs=qa.coords[0]+qa.coords[1], 
                                    bos=(qa.task=='init')
                                )

                            if not past_tokens is None:
                                past_tokens = torch.cat((past_tokens, new_tokens), dim=1)
                            
                            else:
                                past_tokens = new_tokens
            except:
                continue
            
            if i % 20 == 0:            
                with open(os.path.join(save_path, f'log-{args.id}.json'), 'w') as f:
                    json.dump(logs, f)
                            
    # load the log here and resume.
    # with open(f'out/qa/log-{args.id}.json', 'r') as f:
    #   logs = json.load(f)

    # metrics for plan and qa
    metric_dict = dict()
    if task == 'plan':
        planning_metric = PlanningMetric(nusc_path)
        with open(anno_path, 'rb') as f:
            annos = pickle.load(f)
        future_seconds = 3
        l2, cnt = np.zeros(2*future_seconds), 0
        # coll
        colls = [0., 0., 0.]
        
        for log in logs:
            if 'plan' in log:
                l2 += np.array(calc_l2(log['plan'], log['gt'][0]))

                
                plan = torch.tensor(log['plan']).unsqueeze(0)
                gt_infos = annos['infos'][log['id']]
                gt_agent_boxes = np.concatenate([gt_infos['gt_boxes'], gt_infos['gt_velocity']], -1)
                gt_agent_feats = np.concatenate([gt_infos['gt_fut_traj'][:, :6].reshape(-1, 12), gt_infos['gt_fut_traj_mask'][:, :6], gt_infos['gt_fut_yaw'][:, :6], gt_infos['gt_fut_idx']], -1)
                bev_seg = planning_metric.get_birds_eye_view_label(gt_agent_boxes, gt_agent_feats, add_rec=True)
                # mask should be all ones
                gt_traj = gt_infos['gt_planning']
                gt_traj = torch.from_numpy(gt_traj[..., :2])
                seg = torch.from_numpy(bev_seg[1:]).unsqueeze(0)
                for jj in range(future_seconds):
                    cur_time = (jj+1)*2
                    _, coll = planning_metric.evaluate_coll(plan[:,:cur_time,:2], gt_traj[:,:cur_time,:], seg)
                    coll = coll.mean().item()
                    colls[jj] += coll

                cnt += 1

        for i in range(future_seconds):
            cur_time = (i+1)*2
            metric_dict[f'l2_{i+1}s'] = l2[:cur_time].sum().item() / cur_time / cnt
            metric_dict[f'coll_{i+1}s'] = colls[i] / cnt
        metric_dict['samples'] = cnt

    elif task == 'qa':
        predictions, references = [], []
        # batch_size = 1024

        for log in logs:
            if 'task' in log and log['task'] == 'qa':
                predictions.append(log['generate'])
                references.append(log['gt'])
                # add metrics
        qa_dict = eval_qa(predictions, references, 'meteor')
        for k, v in qa_dict.items():
            metric_dict[k] = v

    else:
        raise NotImplementedError()
    
    print(metric_dict)
    # TODO: support more evaluate metrics

    with open(os.path.join(save_path, f'result-{args.id}.json'), 'w') as f:
        json.dump(metric_dict, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="set paths and tasks")
    parser.add_argument('--anno_path', type=str, help='Path to the annotation file.')
    parser.add_argument('--nusc_path', type=str, help='Path to the Nuscenes Dataset.')
    parser.add_argument('--model_path', type=str, help='Path to the model.')
    parser.add_argument('--save_path', type=str, help='Path to save the results.')
    parser.add_argument('--data_path', type=str, help='Path to the processed data.')
    parser.add_argument('--task', type=str, help='Task to evaluate.')
    parser.add_argument('--split', type=int, default=1, help='Split the data for multi gpus.')
    parser.add_argument('--id', type=int, default=0)

    args = parser.parse_args()
    main(args)
