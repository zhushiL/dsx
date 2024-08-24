import os
import hashlib 
import urllib 

import torch 
from PIL import Image 
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize 
from tqdm import tqdm

from .clip_model import build_model 
from .clip_tokenizer import Tokenizer 

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


__all__ = ["available_models", "load", "tokenize"]
_tokenizer = Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

def _download(url, root):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)
    
    # 如果存在就直接返回文件路径
    if os.path.exists(download_target) and expected_sha256:
        with open(download_target, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() == expected_sha256:
                return download_target

    #-------------------------------------------------------------------------------------------------------------------------------
    # 从指定的URL下载文件，同时在命令行中显示下载进度条
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))
    #-------------------------------------------------------------------------------------------------------------------------------

    return download_target

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    """ image transform """
    transform = Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    return transform

def available_models():
    """ 返回所有可用的模型 """
    return list(_MODELS.keys())

def load(name, device = "cuda" if torch.cuda.is_available() else "cpu", download_root=None):
    """ 给出模型名字，和路径，下载模型
    Args:
        name (str): The model name, option from available_models.
        device (str): "cuda" or "cpu".
        download_root (str): Download root.
    Return:
        model (nn.Modul): Model.
        _transform : Image transform.
    """

    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        print(f"请使用 available_models() 里面的模型：{available_models()}")

    with open(model_path, "rb") as f:
        try:
            model = torch.jit.load(f, map_location=device).eval()
            state_dict = None
        except:
            state_dict = torch.load(f, map_location="cpu")
    
    model = build_model(state_dict or model.state_dict()).to(device)
    if str(device) == "cpu":
        model.float()
        
    return model, _transform(model.visual.input_resolution)

def tokenize(texts, context_length=77, truncate=False):
    """ 在文本首尾加入特殊token: sot, eot 并在text填入0 时期达到统一输入长度
    Args:
        texts (str, list(str)): An input string or a list of strings to tokenize.
        context_length (int): Context length. Default: 77
        truncate (bool): If Ture, truncate the text to context_lenth when text is longer than it. 
    """
    
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                print(f"输入{texts[i]}比默认文本长度{context_length}还要长！")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result