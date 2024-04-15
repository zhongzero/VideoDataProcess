import torch
print("PyTorch Version: ",torch.__version__)
print("Is available: ", torch.cuda.is_available())
print("Current Device: ", torch.cuda.current_device())
print("Number of GPUs: ",torch.cuda.device_count())
print("arch support: ", torch.cuda.get_arch_list())

import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())


from share4v.model.builder import load_pretrained_model
from share4v.mm_utils import get_model_name_from_path
from share4v.eval.run_share4v import eval_model

model_path = "Lin-Chen/ShareGPT4V-7B"
prompt = "What is the most common catchphrase of the character on the right?"
image_file = "examples/breaking_bad.png"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)