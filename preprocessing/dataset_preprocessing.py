import xmltodict
import os
from glob import glob
import tifffile
from tqdm import tqdm
import imageio
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
import pandas as pd
import math
import argparse

def is_float(string):
    if string.replace(".", "").isnumeric():
        return True
    else:
        return False

def parse_string(input_string):
    input_string = input_string.replace(' ', '')
    output_string_list = []
    output_string_list.append(input_string[0] + "'")
    for i in range(1,len(input_string)):
        if input_string[i] == "{":
            output_string_list.append("{'")
        elif input_string[i] == "}" and input_string[i-1] != "}":
            output_string_list.append("'}")
        elif input_string[i] == ":":
            if input_string[i+1] != "{" and input_string != "[":
                output_string_list.append("':'")
            else:
                output_string_list.append("':")
        elif input_string[i] == ",":
            if input_string[i-1] == '}':
                output_string_list.append(",'")
            else:
                output_string_list.append("','")
        else:
            output_string_list.append(input_string[i])

    output_string = ''.join(output_string_list)

    final_string_list = []
    for sub_string in output_string.split("'"):
        if is_float(sub_string.replace("[", '').replace("]", '').replace("-", '').replace("+", '')):
            final_string_list.append(sub_string)
        elif any(s in ["{", "}", ":", ","] for s in sub_string):
            final_string_list.append(sub_string)
        else:
            final_string_list.append("'" + sub_string + "'")

    final_string = ''.join(final_string_list)

    return final_string

class OperaPhenixDataset():
    def __init__(self, root, plate_map, channels='all', transform=None):

        self.file_dir = root
        self.plate_map = plate_map
        self.transform = transform
        self.channel_name_dict = {'HOECHST33342': 'Hoechst',
                                  'Alexa488Restrictedemission': 'SytoxGreen',
                                  'Alexa568': 'RADA',
                                  'FM4-64': 'FM4-64',
                                  'FM4-64larger': 'FM4-64',
                                  'Brightfield': 'Brightfield'}

        self.channel_dict = dict()

        xml_path = os.path.join(self.file_dir, 'Index.xml')
        with open(xml_path, 'r', encoding='utf-8') as file:
            xml_file = file.read()

        self.xml_dict = xmltodict.parse(xml_file)

        self.make_channel_dict()
        self.make_image_path_dict()
        self.make_dataset_indices()

        if channels == 'all':
            self._channels = [ch for ch in ['Hoechst', 'SytoxGreen', 'FM4-64', 'Brightfield'] if ch in self.channel_dict.keys()]
        else:
            self._channels = channels

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes())}

    def make_channel_dict(self):
        for channel_num in range(self.num_channels()):
            # Dirty, but works for now
            if self.xml_dict['EvaluationInputData']['@Version'] == '2':
                parsed_dict = self.xml_dict['EvaluationInputData']['Maps']['Map'][1]['Entry']
            else:
                input_str = self.xml_dict['EvaluationInputData']['Maps']['Map'][0]['Entry'][channel_num]['FlatfieldProfile'].replace('Acapella:2013', 'Acapella_2013')
                parsed_string = parse_string(input_str)
                parsed_dict = eval(parsed_string)

            self.channel_dict[self.channel_name_dict[parsed_dict['ChannelName']]] = channel_num + 1

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, channel_list):
        self._channels = [ch for ch in channel_list if ch in self.channel_dict.keys()]

    def make_image_path_dict(self):
        xml_im_dict = self.xml_dict['EvaluationInputData']['Images']['Image']
        im_list = [(xml_im_dict[i]['Row'], xml_im_dict[i]['Col'], xml_im_dict[i]['FieldID'], xml_im_dict[i]['ChannelID'], xml_im_dict[i]['URL']) for i in range(len(self.xml_dict['EvaluationInputData']['Images']['Image']))]
        self.path_dict = dict(zip([(int(item[0]), int(item[1]), int(item[2]), int(item[3])) for item in im_list], [item[4] for item in im_list]))

    def num_channels(self):
        # Unclear how to extract this information if there's only one channel from version 2 metadata
        if self.xml_dict['EvaluationInputData']['@Version'] == '2':
            return 1
        else:
            return len(self.xml_dict['EvaluationInputData']['Maps']['Map'][0]['Entry'])

    def num_rows(self):
        xml_well_dict = self.xml_dict['EvaluationInputData']['Wells']['Well']
        rows = [int(xml_well_dict[i]['Row']) for i in range(len(xml_well_dict))]
        return max(rows)

    def num_columns(self):
        xml_well_dict =self.xml_dict['EvaluationInputData']['Wells']['Well']
        columns = [int(xml_well_dict[i]['Col']) for i in range(len(xml_well_dict))]
        return max(columns)

    def num_fields(self):
        fields = [int(self.xml_dict['EvaluationInputData']['Images']['Image'][i]['FieldID']) for i in range(self.__len__())]
        return max(fields)

    def classes(self):
        return sorted(list(set([item[0] for item in self.plate_map.values()])))

    def __len__(self):
        return math.floor(len(self.xml_dict['EvaluationInputData']['Images']['Image']) / self.num_channels())

    def read_image_stack(self, row, col, field, channels, dtype='uint8', clip=True, **kwargs):
        im_path_list = []
        for channel in channels:
            ch_num = self.channel_dict[channel]
            im_args.base_path = self.path_dict[(row, col, field, ch_num)]
            im_path = os.path.join(self.file_dir, im_args.base_path)
            im_path_list.append(im_path)

        im_list = []
        for im_path in  im_path_list:
            im = tifffile.imread(im_path)

            if clip:
                p_bot, p_top = np.percentile(im, kwargs.get('clip_lower', 0.1)), np.percentile(im, kwargs.get('clip_upper', 99.9))
                im = np.clip(im, p_bot, p_top)

            im = exposure.rescale_intensity(im, out_range=dtype)

            im_list.append(im)

        if kwargs.get('return_path', False):
            return np.stack(im_list), im_path_list
        else:
            return np.stack(im_list)

    def read_image_from_name(self, name, state, field, channels=None):
        inv_plate_map = {value: key for key, value in self.plate_map.items()}

        row, col = inv_plate_map[(name, state)]

        if channels == None:
            channels = self.channels

        return self.read_image_stack(row, col, field, channels)

    def make_dataset_indices(self):
        self.idx_list = []
        for row_idx, col_idx in list(self.plate_map.keys()):
                for field_idx in range(1, self.num_fields() + 1):
                    self.idx_list.append((row_idx, col_idx, field_idx))

    def __getitem__(self, idx):
        row, col, field = self.idx_list[idx]

        im = self.read_image_stack(row, col, field, self.channels)
        label = self.plate_map[(row, col)][0]

        if self.transform:
            im = self.transform(im)

        return im, self.class_to_idx[label]

def make_dir(dir):
    """Create directories including subdirectories"""
    dir_lst = dir.split('/')
    for idx in range(1, len(dir_lst) + 1):
        if not os.path.exists(os.path.join(*dir_lst[:idx])):
            os.mkdir(os.path.join(*dir_lst[:idx]))

def get_plate_map(path):

    row_dict = {"A": 1,
                "B": 2,
                "C": 3,
                "D": 4,
                "E": 5,
                "F": 6,
                "G": 7,
                "H": 8}

    plate_map_df = pd.read_csv(plate_map_path, delimiter=',', usecols=['cond', 'Destination well']).dropna()
    plate_map_df['Row'] = plate_map_df['Destination well'].map(lambda x: x[:1]).map(row_dict)
    plate_map_df['Column'] = plate_map_df['Destination well'].map(lambda x: int(x[1:]))

    plate_map_keys = [(row, col) for (row, col) in zip(list(plate_map_df.Row.values), list(plate_map_df.Column.values))]
    plate_map_values = plate_map_df.cond.values
    plate_map = dict(zip(plate_map_keys, plate_map_values))

    return plate_map

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--im_dir', required=True)
    parser.add_argument('--target_dir', required=True)
    parser.add_argument('--plate_map_path', required=True)
    parser.add_argument('--dtype', default='uint8', required=False)
    parser.add_argument('--channels', nargs='+', default=['Brightfield'])

    args = parser.parse_args()

    plate_map = get_plate_map(args.plate_map_path)

    data_path = os.path.join(args.im_dir, 'Images')

    op_data = OperaPhenixDataset(data_path, plate_map)

    make_dir(args.target_dir)
    for idx in tqdm(op_data.idx_list):
        try:
            condition_name = op_data.plate_map[(idx[0],idx[1])]

            cmpd_name, mic_name = condition_name.split('_')
            new_dir = os.path.join(args.target_dir, f'{cmpd_name}_{mic_name}')

            make_dir(new_dir)

            im_as_tiff_stack, im_paths = op_data.read_image_stack(idx[0], idx[1], idx[2], args.channels, clip=True, return_path=True, dtype=args.dtype)

            im_base_path = os.path.basename(im_paths[0])

            save_path = os.path.join(new_dir, im_base_path)
            tifffile.imwrite(save_path , im_as_tiff_stack, metadata=dict(axes='CYX', Labels=args.channels), imagej=True)

        except KeyError:
            print(f'KeyError: Could not find {idx}!')
        except ValueError:
            print(f'ValueError: {idx}')
