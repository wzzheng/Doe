from random import random
import json
import argparse
import os
import pickle
import sys

from tqdm import tqdm

from data_utils import process_coords, process_desc, check_range

class QA:
    def __init__(self, question=None, answer=None, task=None,
                 image=None, plan=None, coords=None, id=None):
        
        self.q = question
        self.a = answer
        self.image = image if not image is None else []
        self.plan = plan if not plan is None else []
        self.coords = coords if not coords is None else [[], []]
        self.task = task
        
        # other
        self.id = id
        # self.range = (-20, 60)
        
    def extract_coords(self, data_range=(-20, 60)):
        
        c_q, text_q = process_coords(self.q)
        c_a, text_a = process_coords(self.a)
        self.q = text_q
        self.a = text_a
        self.coords = [c_q, c_a]
        return len(c_q) > 0 and check_range(*data_range, c_q) and check_range(*data_range, c_a)
    
    def from_dict(self, d):
        for k, v in d.items():
            self.__setattr__(k, v)

    def insert_coords(self):
        # to do
        pass
            


def main(args):
    info_path = args.info_path
    qa_path = args.qa_path
    nusc_path = args.nusc_path
    save_path = args.save_path
    record_list = list()

    with open(info_path, 'rb') as f:
        data = pickle.load(f)['infos']

    with open('dataset/prompt.json', 'r') as f:
        prompt = json.load(f)

    conv = list()
    view, task = 'CAM_FRONT', 'val'

    max_length = args.max_length # 5 for planning

    for i, sample in tqdm(enumerate(data[:2])):
        info = list()
        token = sample['token']

        if len(conv) == max_length:
            record_list.append(conv)
            conv = list()

        if len(conv) == 0:
            if sample['frame_idx']+10 != data[i+10]['frame_idx']:
                continue

        # generate image
        info.append(QA(prompt['generate_scene'] if len(conv) == 0 else prompt['image'], 
                       '<|image|>', task='init' if len(conv) == 0 else 'image',
                       image=[os.path.join(nusc_path, sample['cams'][view]["data_path"][16:])]))
        
        # generate desc
        desc_path = os.path.join(qa_path, 'desc', task, f'{token}.json')
        with open(desc_path, 'r') as f:
            desc_action = json.load(f)
        desc = process_desc(desc_action['description'])
        action = desc_action['action']
        info.append(QA(prompt['desc'], desc, task='desc'))

        # generate counterfactual
        vqa_path = os.path.join(qa_path, 'vqa', task, f'{token}.json')
        if os.path.exists(vqa_path):
            with open(vqa_path, 'r') as f:
                vqas = json.load(f)
            for vqa in vqas:
                qa = QA(vqa['question'], vqa['answer'], task='cf')
                if qa.extract_coords():
                    info.append(qa)

        # generate qas
        conv_path = os.path.join(qa_path, 'conv', task, f'{token}.json')
        with open(conv_path, 'r') as f:
            convs = json.load(f)
        for qa in convs:
            info.append(QA(qa['question'], qa['answer'], task='qa'))

        # generate action
        info.append(QA(prompt['action'], action, task='action'))

        # generate plan
        plans = sample['gt_planning'][0].tolist()
        info.append(QA(prompt['plan'], '<|plan|>', task='plan', plan=[plans], id=i))
        
        info = list(map(vars, info))
        conv.append(info)
        # print(info)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(record_list, f, ensure_ascii=False, indent=4)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--info_path', type=str, help='Path to infos.pkl.')
    parser.add_argument('--qa_path', type=str, help='Path to conversation data.')
    parser.add_argument('--nusc_path', type=str, help='Path to nuscenes dataset.')
    parser.add_argument('--save_path', type=str, help='Path to save processed data.')
    parser.add_argument('--max_length', type=int, default=1, help='Frames of conversations.')

    args = parser.parse_args()
    main(args)