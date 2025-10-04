import lzma
import dahuffman
import pickle
import numpy as np
import torch

def huffman_encode(data):
    codec = dahuffman.HuffmanCodec.from_data(data)
    encoded_bytes = codec.encode(data)
    huffman_table = codec.get_code_table()
    return encoded_bytes, huffman_table

def huffman_decode(encoded_bytes, huffman_table):
    codec = dahuffman.HuffmanCodec(code_table=huffman_table)
    decoded_data = codec.decode(encoded_bytes)
    return np.array(decoded_data, dtype=np.uint16)

def save_comp(filename, save_dict):
    with lzma.open(filename, "wb") as f:
        pickle.dump(save_dict, f)

def load_comp(filename):
    with lzma.open(filename, "rb") as f:
        save_dict = pickle.load(f)
    return save_dict

def write_storage_original(save_dict, byte, numG):
    for name in save_dict:
        if name == 'xyz':
            byte['xyz'] = len(save_dict['xyz'])
        elif 'MLP' in name:
            byte['MLPs'] += save_dict[name].shape[0]*16/8
        else:
            attr, comp = name.split('_')
            if 'code' in comp:
                for i in range(len(save_dict[name])):
                    byte[attr] += save_dict[name][i].shape[0]*save_dict[name][i].shape[1]*16/8
            else:
                for i in range(len(save_dict[name])):
                    byte[attr] += len(save_dict[name][i])
    byte['total'] = byte['xyz'] + byte['scale'] + byte['rotation'] + byte['app'] + byte['MLPs']
    return "#G: " + str(numG) + "\nPosition: " + str(byte['xyz']) + "\nScale: " + str(byte['scale']) + "\nRotation: " + str(byte['rotation']) + "\nAppearance: " + str(byte['app']) + "\nMLPs: " + str(byte['MLPs'])+ "\nTotal: " + str(byte['total']) + "\n"

def write_storage(save_dict, numG):
    byte = {
        'xyz': 0.0,
        'scale': 0.0,
        'rotation': 0.0,
        'app': 0.0,
        'MLPs': 0.0,
        '4D': 0.0
    }

    xyz = save_dict['xyz']
    byte['xyz'] = xyz.numel() * xyz.element_size() / 1024 / 1024

    for name in ['scale_code', 'scale_index']:
        for arr in save_dict[name]:
            if hasattr(arr, 'nbytes'):
                byte['scale'] += arr.nbytes / 1024 / 1024
            else:
                byte['scale'] += len(arr) * 4 / 1024 / 1024

    for name in ['rotation_code', 'rotation_index']:
        for arr in save_dict[name]:
            if hasattr(arr, 'nbytes'):
                byte['rotation'] += arr.nbytes / 1024 / 1024
            else:
                byte['rotation'] += len(arr) * 4 / 1024 / 1024

    for name in ['app_code', 'app_index']:
        for arr in save_dict[name]:
            if hasattr(arr, 'nbytes'):
                byte['app'] += arr.nbytes / 1024 / 1024
            else:
                byte['app'] += len(arr) * 4 / 1024 / 1024

    for name in ['MLP_cont', 'MLP_dc', 'MLP_sh', 'MLP_opacity']:
        arr = save_dict[name]
        byte['MLPs'] += arr.nbytes / 1024 / 1024

    for name in ['t', 'scaling_t', 'rotation_r']:
        if name in save_dict:
            arr = save_dict[name]
            byte['4D'] += arr.nbytes / 1024 / 1024

    byte['total'] = byte['xyz'] + byte['scale'] + byte['rotation'] + byte['app'] + byte['MLPs'] + byte['4D']

    return (
        f"#G: {numG}\n"
        f"Position: {byte['xyz']:.3f} MB\n"
        f"Scale: {byte['scale']:.3f} MB\n"
        f"Rotation: {byte['rotation']:.3f} MB\n"
        f"Appearance: {byte['app']:.3f} MB\n"
        f"MLPs: {byte['MLPs']:.3f} MB\n"
        f"4D Extra: {byte['4D']:.3f} MB\n"
        f"Total: {byte['total']:.3f} MB\n"
    )



def splitBy3(a):
    x = a & 0x1FFFFF
    x = (x | x << 32) & 0x1F00000000FFFF
    x = (x | x << 16) & 0x1F0000FF0000FF
    x = (x | x << 8) & 0x100F00F00F00F00F
    x = (x | x << 4) & 0x10C30C30C30C30C3
    x = (x | x << 2) & 0x1249249249249249
    return x


def mortonEncode(pos: torch.Tensor) -> torch.Tensor:
    x, y, z = pos.unbind(-1)
    answer = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
    answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2
    return answer