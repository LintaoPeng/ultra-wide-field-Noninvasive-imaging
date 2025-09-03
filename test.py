import os
import torch
import numpy as np
import scipy.io as sio
import argparse
from torchvision.utils import save_image


from models import DenoisingDiffusion  # 确保 import 正确
from utils.utils import dict2namespace


dtype = 'float32'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.set_default_tensor_type(torch.FloatTensor)


def load_test_data(folder_path):
    files = sorted(os.listdir(folder_path), key=lambda x: int(x.split('.')[0]))
    training_x = []
    for file in files:
        mat = sio.loadmat(os.path.join(folder_path, file))["data"]
        imgx = (mat - mat.min()) / (mat.max() - mat.min())
        # 切割为28x28的块，并整合成32个通道
        blocks = []
        for i in range(0, 256, 32):
            for j in range(0, 256, 32):
                block = imgx[i:i + 32, j:j + 32]
                if block.shape == (32, 32):  # 确保是完整的28x28块
                    blocks.append(block)
                if len(blocks) == 64:  # 每张图片生成32个28x28块后停止
                    break # 跳出内层循环
            if len(blocks) == 64:
                training_x.append(blocks)
                break  # 跳出外层循环
    tensor_data = torch.from_numpy(np.array(training_x)).float()
    return tensor_data

def parse_args():
    parser = argparse.ArgumentParser(description="Test diffusion model")
    parser.add_argument('--config', type=str, default='speckle.yml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='./ckpt/checkpoint.pth.tar', help='Path to pretrained model checkpoint')
    parser.add_argument('--data_folder', type=str, default='./data/test/speckle/', help='Path to test data')
    parser.add_argument('--output_folder', type=str, default='./data/test/output/', help='Where to save results')
    parser.add_argument('--sampling_timesteps', type=int, default=10, help='Sampling steps')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load config
    import yaml
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    print("Loading test data from:", args.data_folder)
    test_data = load_test_data(args.data_folder)
    test_data = test_data.to(config.device)

    # Initialize model
    ddm = DenoisingDiffusion(args, config)
    ddm.load_ddm_ckpt(args.checkpoint, ema=True)
    model = ddm.model.to(config.device)
    model.eval()

    # Inference
    print("Running inference...")
    with torch.no_grad():
        for idx in range(test_data.size(0)):
            x = test_data[idx:idx+1]
            #x=x.unsqueeze(1)
            #print(x.shape)
            x = x.to(config.device)
            dummy_gt = torch.zeros((x.size(0), 1, x.size(2), x.size(3)), device=x.device)
            input_x = torch.cat([x, dummy_gt], dim=1)
            output = ddm.model(input_x)
            pred = torch.clamp(output["pred_x"], 0, 1)

            os.makedirs(args.output_folder, exist_ok=True)
            save_image(pred, os.path.join(args.output_folder, f"{idx+1}.png"))
            print(f"Saved prediction to {args.output_folder}/{idx+1}.png")

if __name__ == "__main__":
    main()
