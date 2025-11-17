import os
import torch
import torchvision
from diffusers import DDIMScheduler
import importlib.util


def load_module_from_path(path):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_checkpoint(save_path, device):
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Checkpoint not found: {save_path}")
    ckpt = torch.load(save_path, map_location=device)
    if isinstance(ckpt, dict):
        # Prefer EMA weights for sampling
        state = ckpt.get("ema_state_dict", ckpt.get("model_state_dict", ckpt))
    else:
        state = ckpt
    return state


def infer_num_classes_from_state(state):
    # Expect key like 'class_emb.weight'
    for k, v in state.items():
        if k.endswith('class_emb.weight') and v.ndim == 2:
            return v.shape[0]
    raise RuntimeError("Unable to infer num_classes from checkpoint state.")


def build_model(module, state, device):
    num_classes = infer_num_classes_from_state(state)
    # Instantiate with explicit num_classes to avoid relying on module-level dataset
    model = module.ClassConditionedUViT(num_classes=num_classes).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def generate_grid(model, categories, num_per_cat=4, img_sz=(256, 256), guidance_scale=2.0, steps=60, device='cpu'):
    # Create noise and labels
    total = len(categories) * num_per_cat
    x = torch.randn(total, 1, img_sz[0], img_sz[1], device=device)
    y_list = []
    for c in categories:
        y_list.extend([c] * num_per_cat)
    y = torch.tensor(y_list, device=device)

    # DDIM scheduler configured similarly to training
    ddim = DDIMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2', prediction_type='v_prediction')
    ddim.set_timesteps(steps)

    # Iterate timesteps
    for t in ddim.timesteps:
        # Conditional
        residual_cond = model(x, t.to(device), y)
        # Light unconditional via 10% dropout to class 0
        drop_mask = (torch.rand_like(y.float()) < 0.1)
        y_uncond = y.clone()
        y_uncond[drop_mask] = 0
        residual_uncond = model(x, t.to(device), y_uncond)
        residual = residual_uncond + guidance_scale * (residual_cond - residual_uncond)
        x = ddim.step(residual, t, x).prev_sample
    return x


def save_grid(x, out_path, nrow):
    torchvision.utils.save_image(
        x.detach().cpu().clip(-1, 1),
        out_path,
        nrow=nrow,
        normalize=True,
        value_range=(-1, 1)
    )


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

    # Paths to the two training scripts
    base_dir = os.path.dirname(__file__)
    file_12 = os.path.join(base_dir, '12_鼎-基于10_新尝试.py')
    file_10 = os.path.join(base_dir, '10_井_cp2-基于10_生成大图.py')

    # Resolve checkpoint paths based on script filenames
    base_12 = os.path.splitext(os.path.basename(file_12))[0]
    base_10 = os.path.splitext(os.path.basename(file_10))[0]
    ckpt_12 = os.path.join(base_dir, f"{base_12}_model.pth")
    ckpt_10 = os.path.join(base_dir, f"{base_10}_model.pth")

    # Load modules
    mod_12 = load_module_from_path(file_12)
    mod_10 = load_module_from_path(file_10)

    # Load checkpoints and build models
    state_12 = load_checkpoint(ckpt_12, device)
    state_10 = load_checkpoint(ckpt_10, device)
    model_12 = build_model(mod_12, state_12, device)
    model_10 = build_model(mod_10, state_10, device)

    # Specify the 4 categories to compare (edit as needed)
    categories = [2, 5, 8, 12]  # four class indices common to both
    num_per_cat = 4
    img_sz = (256, 256)

    # Generate grids
    x12 = generate_grid(model_12, categories, num_per_cat=num_per_cat, img_sz=img_sz, device=device)
    x10 = generate_grid(model_10, categories, num_per_cat=num_per_cat, img_sz=img_sz, device=device)

    # Save outputs side by side for comparison
    out_12 = os.path.join(base_dir, f"{base_12}_compare.png")
    out_10 = os.path.join(base_dir, f"{base_10}_compare.png")
    save_grid(x12, out_12, nrow=num_per_cat)
    save_grid(x10, out_10, nrow=num_per_cat)
    print(f"Saved MoE model grid: {out_12}")
    print(f"Saved baseline model grid: {out_10}")


if __name__ == '__main__':
    main()