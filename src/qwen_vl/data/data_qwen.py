import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2
from .utils import prepare_image_inputs
import torch.nn.functional as F
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None

try:
    import moxing as mox
    import io
except:
    print("load moxing failed")


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path, max_samples: int=-1):
    with open(path, "r") as f:
        # return [json.loads(line) for line in f]
        ret = []
        for line in f:
            ret.append(json.loads(line))
            if max_samples !=-1 and len(ret) >= max_samples:
                break
    return ret


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw: List = [],
    visual_type: str = "image",
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."
    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index = 0
    input_ids, targets = [], []
    prompts, answers = [], []
    ori_convs = []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []
        prompt, answer = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]
            ori_conv = copy.deepcopy(conv)
            ori_convs.append(ori_conv)
            

            role = roles.get(role, role)
            if role == "user":
                visual_tag = f"<{visual_type}>"
                if visual_tag in content:
                    parts = content.split(visual_tag)
                    new_parts = []

                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])

                        replacement = (
                            "<|vision_start|>"
                            + f"<|{visual_type}_pad|>"
                            * grid_thw[visual_replicate_index]
                            + "<|vision_end|>"
                        )

                        new_parts.append(replacement)
                        visual_replicate_index += 1

                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)

            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)
        prompts.append(prompt)
        answers.append(answer)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)


    del tokenizer
    return dict(
        input_ids=input_ids,
        labels=targets,
        prompts=prompts,
        answers=answers,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"], max_samples=data_args.max_samples)
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                ann["data_path"] = data["data_path"]
                ann["tag"] = data["tag"]
            list_data_dict += annotations

        print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

        self.depart_smi_token = getattr(self.data_args, "depart_smi_token", False)
        self.smi_image_num = getattr(self.data_args, "smi_image_num", 8)
        self.smi_downsample_rate = getattr(self.data_args, "smi_downsample_rate", 2)

        self.setup_lengths()

    def setup_lengths(self):
        self.cached_lengths = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            if "image" in sample:
                image_num = len(sample["image"])
            elif "images" in sample:
                image_num = len(sample["images"])
            elif "video" in sample:
                image_num = getattr(self.data_args, "video_max_frames", 8)
            else:
                image_num = 0

            if self.depart_smi_token and image_num > self.smi_image_num:
                self.cached_lengths.append(image_num * 252 // (self.smi_downsample_rate**2) + cur_len)
            else:self.cached_lengths.append(image_num * 252 + cur_len)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        return self.cached_lengths

        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            if "image" in sample:
                image_num = len(sample["image"])
            elif "images" in sample:
                image_num = len(sample["images"])
            elif "video" in sample:
                image_num = getattr(self.data_args, "video_max_frames", 8)
            else:
                image_num = 0
            length_list.append(image_num * 252 + cur_len)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample, lenght in zip(self.list_data_dict, self.cached_lengths):
            tag = sample.get("tag", "2d")
            cur_len = -lenght if tag == "2d" else lenght
            length_list.append(cur_len)
        return length_list

        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            if "image" in sample:
                image_num = len(sample["image"])
            elif "images" in sample:
                image_num = len(sample["images"])
            elif "video" in sample:
                image_num = getattr(self.data_args, "video_max_frames", 8)
            else:
                image_num = 0
            cur_len += image_num*252
            tag = sample.get("tag", "2d")
            cur_len = -cur_len if tag == "2d" else cur_len
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        del processor
        return image_tensor, grid_thw
    
    def draw_visual_marks(self, images, spar_info):

        if spar_info is None:
            return
        info = json.loads(spar_info)
        task_type = info["type"]
        from .draw_marker import DRAW_FUNCTIONS
        draw_fn = DRAW_FUNCTIONS[task_type]
        if len(images) == 1:
            draw_fn(images[0], info)
        else:
            draw_fn(images, info)
        # for j, img in enumerate(images):
        #     # write to local
        #     img.save(f"images/img_{j}.jpg", format="JPEG")

    def process_video(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        del processor
        return video_tensor, grid_thw, second_per_grid_ts
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            # sample = self._get_item(i)

            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e
    
    def read_video_images(self, source):
        # read video images from the source
        assert isinstance(source["video"], str), "video should be a string"
        video_file = os.path.join(source["data_path"], source["video"])
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
            raise FileNotFoundError
        
        def get_frame_indices(total_frames, fps=1):
            video_length = total_frames / fps
            interval = getattr(self.data_args, "base_interval", 2)
            num_frames_to_sample = round(video_length / interval)
            video_min_frames = getattr(self.data_args, "video_min_frames", 4)
            video_max_frames = getattr(self.data_args, "video_max_frames", 8)
            target_frames = min(
                max(num_frames_to_sample, video_min_frames), video_max_frames
            )
            frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
            frame_idx = np.unique(frame_idx)
            return frame_idx        

        # check whether video_file is a directory
        # if os.path.isdir(video_file):
        #     frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
        #     frame_files.sort()
        #     frame_idx = get_frame_indices(len(frame_files), 1)
        #     images = [frame_files[i] for i in frame_idx]
        #     images = [Image.open(frame).convert("RGB") for frame in images]
        # elif any([video_file.endswith(ext) for ext in [".mp4", ".avi", ".mov"]]):
        #     vr = VideoReader(video_file, num_threads=4)
        #     total_frames = len(vr)
        #     avg_fps = vr.get_avg_fps()
        #     frame_idx = get_frame_indices(total_frames, avg_fps)
        #     video = vr.get_batch(frame_idx).asnumpy()
            
        #     images = [Image.fromarray(frame).convert("RGB") for frame in video]
        
        if os.path.isdir(video_file):
            frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
            frame_files.sort()
            frame_idx = get_frame_indices(len(frame_files), 1)

            # images = [frame_files[i] for i in frame_idx]
            # images = [Image.open(frame).convert("RGB") for frame in images]

            images = []
            for idx in frame_idx:
                frame_path = frame_files[idx]
                with Image.open(frame_path) as img:
                    rgb_img = img.convert("RGB")
                    # 注意：这里需要复制数据，因为上下文管理器退出后会关闭文件
                    images.append(rgb_img.copy())  # 重要：复制图像数据
            
            # 及时清理临时列表
            del frame_files
            del frame_idx

            import gc
            gc.collect()

        elif any([video_file.endswith(ext) for ext in [".mp4", ".avi", ".mov"]]):
            vr = None
            try:
                vr = VideoReader(video_file, num_threads=4)
                total_frames = len(vr)
                avg_fps = vr.get_avg_fps()
                frame_idx = get_frame_indices(total_frames, avg_fps)
                
                # 分批处理帧，避免一次性加载所有帧
                batch_size = 10  # 根据需要调整
                images = []
                
                for i in range(0, len(frame_idx), batch_size):
                    batch_indices = frame_idx[i:i+batch_size]
                    video_batch = vr.get_batch(batch_indices).asnumpy()
                    
                    for frame in video_batch:
                        img = Image.fromarray(frame).convert("RGB")
                        images.append(img)
                        # 及时清理临时变量
                        del frame
                    
                    del video_batch
                    
            finally:
                # 确保 VideoReader 被正确释放
                if vr is not None:
                    del vr
            
            # 强制垃圾回收
            import gc
            gc.collect()

        return images

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        video = None
        
        if "video" in sources[0]:
            sources[0]["images"] = self.read_video_images(sources[0])
            num_image = len(sources[0]["images"])
            sources[0]["conversations"][0]["value"] = sources[0]["conversations"][0]["value"].replace(
                DEFAULT_VIDEO_TOKEN, "".join([DEFAULT_IMAGE_TOKEN] * num_image)
            )
            del sources[0]["video"]

        # # replace <image>\n with <image>
        sources[0]["conversations"][0]["value"] = sources[0]["conversations"][0]["value"].replace(
            f"{DEFAULT_IMAGE_TOKEN}\n", DEFAULT_IMAGE_TOKEN
        )

        # rename images tag
        if "images" in sources[0]:
            sources[0]["image"] = sources[0]["images"]
    
        # import pdb
        # pdb.set_trace()

        # notice that we use images as the tag
        if "image" in sources[0]:
            image_folder = self.list_data_dict[i]["data_path"]
            image_file = self.list_data_dict[i]["image"]
            if isinstance(image_file, List):

                if isinstance(image_file[0], str):
                    
                    if image_file[0].startswith("obs"):
                        image_file = [
                            io.BytesIO(mox.file.read(file, binary=True)) for file in image_file
                        ]
                    else:
                        image_file = [
                            os.path.join(image_folder, file) for file in image_file
                        ]
                    image_file = [Image.open(img).convert("RGB") for img in image_file]
                    
                    # new_image_file = []
                    # for file in image_file:
                    #     if file.startswith("obs"):
                    #         file_bytes = mox.file.read(file, binary=True)
                    #         file = io.BytesIO(file_bytes)
                    #     else:
                    #         file = os.path.join(image_folder, file)
        
                    #     new_image_file.append(Image.open(file).convert("RGB"))
                    # image_file = new_image_file
                elif isinstance(image_file[0], Image.Image):
                    pass
                else:
                    raise NotImplementedError
                
                def _resize_all_to_first(images, resample=Image.BILINEAR):
                    if len(images) <= 1:
                        return images
                    W, H = images[0].size  # 以第一张为准
                    out = [images[0]]
                    for img in images[1:]:
                        if img.size != (W, H):
                            img = img.resize((W, H), resample)
                        out.append(img)
                    return out
                image_file = _resize_all_to_first(image_file, resample=Image.BILINEAR)
                # draw visual markers
                self.draw_visual_marks(image_file, sources[0].get("spar_info", None))

                image, grid_thw, geometry_encoder_inputs = [], [], []
                for file in image_file:
                    ret = prepare_image_inputs(file, self.data_args.image_processor)
                    image.append(ret["pixel_values"])
                    geometry_encoder_inputs.append(ret["geometry_encoder_inputs"])
                    grid_thw.append(ret["image_grid_thw"])
            else:
                raise NotImplementedError

            sources = copy.deepcopy([e["conversations"] for e in sources])

            grid_thw_merged = copy.deepcopy(grid_thw)

            if self.depart_smi_token and len(sources[0][0]['value'].split("<image>")) > (self.smi_image_num+1):
                merge_size = self.data_args.image_processor.merge_size*self.smi_downsample_rate
            else: merge_size = self.data_args.image_processor.merge_size

            # import pdb
            # pdb.set_trace()

            grid_thw_merged = [
                merged_thw[0] * (merged_thw[1] // merge_size) * (merged_thw[2] // merge_size)
                for merged_thw in grid_thw_merged
            ]

            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="image"
            )

            position_ids, _ = self.get_rope_index(
                merge_size,
                data_dict["input_ids"],
                torch.stack(grid_thw, dim=0),
            )

            del grid_thw_merged
            del sources
        elif "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.list_data_dict[i]["data_path"]
            if isinstance(video_file, List):
                if len(video_file) > 1:
                    video_file = [
                        os.path.join(video_folder, file) for file in video_file
                    ]
                    results = [self.process_video(file) for file in video_file]
                    video, grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                video = [video]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="video"
            )
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                video_grid_thw=torch.stack(grid_thw, dim=0),
                second_per_grid_ts=second_per_grid_ts,
            )
            del grid_thw_merged
            del sources
        else:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged
            )
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )
            del sources

        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
                prompt=data_dict["prompts"][0],
                answer=data_dict["answers"][0],
                position_ids=position_ids,
            )

        if "image" in self.list_data_dict[i]:
            data_dict["pixel_values"] = image
            data_dict["image_grid_thw"] = grid_thw
            if getattr(self.data_args, "use_geometry_encoder", False):
                data_dict["geometry_encoder_inputs"] = geometry_encoder_inputs
        # video exist in the data
        elif "video" in self.list_data_dict[i]:
            data_dict["pixel_values_videos"] = video
            data_dict["video_grid_thw"] = grid_thw
        
        data_dict["tag"] = self.list_data_dict[i].get("tag", "2d")

        # import pdb
        # pdb.set_trace()

        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            itertools.chain(
                *(
                    instance["pixel_values"]
                    for instance in instances
                    if "pixel_values" in instance
                )
            )
        )
        videos = list(
            itertools.chain(
                *(
                    instance["pixel_values_videos"]
                    for instance in instances
                    if "pixel_values_videos" in instance
                )
            )
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(
                itertools.chain(
                    *(
                        instance["image_grid_thw"]
                        for instance in instances
                        if "image_grid_thw" in instance
                    )
                )
            )
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(
                    *(
                        instance["video_grid_thw"]
                        for instance in instances
                        if "video_grid_thw" in instance
                    )
                )
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids

        # import pdb
        # pdb.set_trace()
                
        # assume all data in a batch has geometry_encoder_inputs
        if "geometry_encoder_inputs" in instances[0]:
            geometry_encoder_inputs = [torch.stack(instance["geometry_encoder_inputs"]) for instance in instances]
            batch["geometry_encoder_inputs"] = geometry_encoder_inputs
            assert len(set([instance["tag"] for instance in instances])) == 1, "all data in a batch should have the same tag"
            batch["tag"] = instances[0]["tag"]
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )

        seq_lens = torch.tensor(
            [0] + [len(seq) for seq in input_ids], dtype=torch.int32
        )
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=0)
        labels = torch.cat(labels, dim=0)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids.unsqueeze(0),
            labels=labels.unsqueeze(0),
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            itertools.chain(
                *(
                    instance["pixel_values"]
                    for instance in instances
                    if "pixel_values" in instance
                )
            )
        )
        videos = list(
            itertools.chain(
                *(
                    instance["pixel_values_videos"]
                    for instance in instances
                    if "pixel_values_videos" in instance
                )
            )
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(
                itertools.chain(
                    *(
                        instance["image_grid_thw"]
                        for instance in instances
                        if "image_grid_thw" in instance
                    )
                )
            )
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(
                    *(
                        instance["video_grid_thw"]
                        for instance in instances
                        if "video_grid_thw" in instance
                    )
                )
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

                
        # assume all data in a batch has geometry_encoder_inputs
        if "geometry_encoder_inputs" in instances[0]:
            raise NotImplementedError("FlattenedDataCollatorForSupervisedDataset does not support geometry_encoder_inputs")

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
