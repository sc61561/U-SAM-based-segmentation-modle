"""
Convert U-SAM pth checkpoint to ONNX format.
Supports custom pth file path via command line argument.
Auto-detects whether the checkpoint has input_stem weights.
"""
import argparse
import os
import sys

import torch
import torch.nn as nn


def load_checkpoint_compat(path_or_file, map_location='cpu'):
    """Load checkpoint with compatibility handling."""
    try:
        return torch.load(path_or_file, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path_or_file, map_location=map_location)


def check_has_input_stem(checkpoint):
    """Check if checkpoint contains input_stem weights."""
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Check for any input_stem related keys
    for key in state_dict.keys():
        if key.startswith('input_stem.'):
            return True
    return False


class USAMExporterWithStem(nn.Module):
    """Wrapper for U-SAM model WITH input_stem layer."""
    
    def __init__(self, args):
        super().__init__()
        from segment_anything import sam_model_registry
        
        args.model_type = 'vit_b'
        args.sam_weight = 'weight/sam_vit_b_01ec64.pth'
        
        self.sam = sam_model_registry[args.model_type](
            num_classes=args.sam_num_classes,
            img_size=args.img_size,
            checkpoint=args.sam_weight
        )
        
        self.pixel_mean = None
        self.pixel_std = None
        self.img_size = args.img_size
        
        # Include input_stem layer
        self.input_stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 3, kernel_size=1),
        )
        
        from backbone import UNet as downsample
        self.backbone = downsample()
        
    def forward(self, x):
        device = x.device
        
        if self.pixel_mean is None:
            self.pixel_mean = (0.1364736, 0.1364736, 0.1364736)
            self.pixel_std = (0.23238614, 0.23238614, 0.23238614)
        
        pixel_mean = torch.tensor(self.pixel_mean).float().to(device)
        pixel_mean = pixel_mean.view(1, 3, 1, 1)
        pixel_std = torch.tensor(self.pixel_std).float().to(device)
        pixel_std = pixel_std.view(1, 3, 1, 1)
        x = (x - pixel_mean) / pixel_std
        
        x = self.input_stem(x)
        
        bt_feature, skip_feature = self.backbone(x)
        image_embedding = self.sam.image_encoder(bt_feature)
        
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        
        masks, low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            skip=skip_feature,
        )
        
        masks = self.sam.postprocess_masks(
            masks=masks,
            input_size=masks.shape[-2:],
            original_size=[self.img_size, self.img_size]
        )
        
        return masks


class USAMExporterNoStem(nn.Module):
    """Wrapper for U-SAM model WITHOUT input_stem layer (original USAM)."""
    
    def __init__(self, args):
        super().__init__()
        from segment_anything import sam_model_registry
        
        args.model_type = 'vit_b'
        args.sam_weight = 'weight/sam_vit_b_01ec64.pth'
        
        self.sam = sam_model_registry[args.model_type](
            num_classes=args.sam_num_classes,
            img_size=args.img_size,
            checkpoint=args.sam_weight
        )
        
        self.pixel_mean = None
        self.pixel_std = None
        self.img_size = args.img_size
        
        # NO input_stem layer - direct passthrough
        from backbone import UNet as downsample
        self.backbone = downsample()
        
    def forward(self, x):
        device = x.device
        
        if self.pixel_mean is None:
            self.pixel_mean = (0.1364736, 0.1364736, 0.1364736)
            self.pixel_std = (0.23238614, 0.23238614, 0.23238614)
        
        pixel_mean = torch.tensor(self.pixel_mean).float().to(device)
        pixel_mean = pixel_mean.view(1, 3, 1, 1)
        pixel_std = torch.tensor(self.pixel_std).float().to(device)
        pixel_std = pixel_std.view(1, 3, 1, 1)
        x = (x - pixel_mean) / pixel_std
        
        # No input_stem - directly use x
        bt_feature, skip_feature = self.backbone(x)
        image_embedding = self.sam.image_encoder(bt_feature)
        
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        
        masks, low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            skip=skip_feature,
        )
        
        masks = self.sam.postprocess_masks(
            masks=masks,
            input_size=masks.shape[-2:],
            original_size=[self.img_size, self.img_size]
        )
        
        return masks


def convert_pth_to_onnx(pth_path, output_path=None, img_size=224, sam_num_classes=3, opset_version=14):
    """
    Convert pth checkpoint to ONNX format.
    Auto-detects whether the checkpoint has input_stem weights.
    
    Args:
        pth_path: Path to the pth checkpoint file
        output_path: Path for the output ONNX file (optional)
        img_size: Input image size (default: 224)
        sam_num_classes: Number of segmentation classes (default: 3)
        opset_version: ONNX opset version (default: 14)
    """
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"pth file not found: {pth_path}")
    
    if output_path is None:
        output_path = pth_path.replace('.pth', '.onnx')
    
    print(f"Loading checkpoint from: {pth_path}")
    checkpoint = load_checkpoint_compat(pth_path, map_location='cpu')
    
    args = argparse.Namespace()
    args.img_size = img_size
    args.sam_num_classes = sam_num_classes
    
    # Auto-detect whether checkpoint has input_stem weights
    has_input_stem = check_has_input_stem(checkpoint)
    
    if has_input_stem:
        print("Detected: Model has input_stem layer (Stem+2.5D USAM)")
        model = USAMExporterWithStem(args)
    else:
        print("Detected: Model does NOT have input_stem layer (Original USAM)")
        model = USAMExporterNoStem(args)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
        print(f"Loaded model state dict from checkpoint")
    else:
        model.load_state_dict(checkpoint, strict=True)
        print(f"Loaded state dict directly")
    
    model.eval()
    
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    print(f"Exporting to ONNX: {output_path}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"SAM num classes: {sam_num_classes}")
    print(f"ONNX opset version: {opset_version}")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Successfully converted to ONNX: {output_path}")
    
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed!")
    except ImportError:
        print("Note: onnx package not installed, skipping validation")
    except Exception as e:
        print(f"Warning: ONNX validation failed: {e}")
    
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description='Convert U-SAM pth checkpoint to ONNX format')
    parser.add_argument('--pth_path', type=str, required=True,
                        help='Path to the input pth checkpoint file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path for the output ONNX file (default: same as input with .onnx extension)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--sam_num_classes', type=int, default=3,
                        help='Number of SAM segmentation classes (default: 3)')
    parser.add_argument('--opset_version', type=int, default=14,
                        help='ONNX opset version (default: 14)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert_pth_to_onnx(
        pth_path=args.pth_path,
        output_path=args.output,
        img_size=args.img_size,
        sam_num_classes=args.sam_num_classes,
        opset_version=args.opset_version
    )
