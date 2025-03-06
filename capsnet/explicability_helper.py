import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from capsnet.utils import margin_loss
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer_names):
        self.model = model
        self.target_layer_names = target_layer_names
        self.gradients = {layer: None for layer in target_layer_names}
        self.activations = {layer: None for layer in target_layer_names}
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(layer_name):
            def hook(module, input, output):
                self.activations[layer_name] = output.clone().detach()
            return hook

        def backward_hook(layer_name):
            def hook(module, grad_input, grad_output):
                self.gradients[layer_name] = grad_output[0].clone().detach()
            return hook

        for layer_name in self.target_layer_names:
            target_module = dict(self.model.named_modules())[layer_name]
            target_module.register_forward_hook(forward_hook(layer_name))
            target_module.register_full_backward_hook(backward_hook(layer_name))
    
    def __call__(self, input_image, class_index=None):
        self.model.zero_grad()
        output, _ = self.model(input_image)
        if class_index is None:
            class_index = output.argmax(dim=1).item()
        score = output[0, class_index]
        score.backward(retain_graph=True)
        
        cams = {}
        for layer_name in self.target_layer_names:
            weights = self.gradients[layer_name].mean(dim=[2, 3], keepdim=True)
            cam = (weights * self.activations[layer_name]).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            cams[layer_name] = cam.squeeze().cpu().numpy()
        
        return cams

def integrated_gradients(model, input_image, device, target_class, baseline=None, steps=50):
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(input_image).to(device)
    
    input_image = input_image.unsqueeze(0).to(device)
    
    scaled_images = [baseline + (float(i) / steps) * (input_image - baseline) for i in range(0, steps + 1)]
    scaled_images = torch.cat(scaled_images, dim=0)
    scaled_images.requires_grad = True

    outputs, _ = model(scaled_images)
    
    scores = outputs[:, target_class]

    grads = torch.autograd.grad(outputs=scores.sum(), inputs=scaled_images)[0]
    avg_grads = grads.mean(dim=0)
    integrated_grads = (input_image - baseline) * avg_grads.unsqueeze(0)

    return integrated_grads.squeeze(0).detach().cpu().numpy()

def visualize_high_contrast_explanations_for_all_classes(
    model, dataset, device, 
    target_layer_names, max_images_per_class=None
):
    model.eval()
    classes = dataset.classes
    grad_cam = GradCAM(model, target_layer_names=target_layer_names)
    
    num_layers = len(target_layer_names)
    num_cols = 2 + num_layers  

    fig, axes = plt.subplots(len(classes), num_cols, figsize=(4 * num_cols, 4 * len(classes)))

    for i, class_name in enumerate(classes):
        images_for_class = [img for img, label in dataset if classes[label] == class_name]

        if max_images_per_class:
            images_for_class = images_for_class[:max_images_per_class]

        if not images_for_class:
            continue  

        saliency_maps = []
        grad_cams_per_layer = {layer: [] for layer in target_layer_names}

        for img in images_for_class:
            img_tensor = img.unsqueeze(0).to(device)
            img_tensor.requires_grad = True

            outputs, _ = model(img_tensor)

            label = torch.tensor([dataset.classes.index(class_name)], device=device)
            loss = margin_loss(outputs, label)
            loss.backward()

            saliency = img_tensor.grad.abs().squeeze()
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            saliency_maps.append(saliency.cpu().numpy())

            cams = grad_cam(img_tensor)
            for layer in target_layer_names:
                cam_resized = F.interpolate(
                    torch.tensor(cams[layer], device=device).unsqueeze(0).unsqueeze(0), 
                    size=img.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze()
                grad_cams_per_layer[layer].append(cam_resized.cpu().numpy())

        saliency_p90 = np.percentile(np.stack(saliency_maps), 90, axis=0)
        grad_cams_p90 = {layer: np.percentile(np.stack(grad_cams_per_layer[layer]), 90, axis=0) for layer in target_layer_names}

        axes[i, 0].imshow(saliency_p90, cmap='gray')
        axes[i, 0].set_title(f"{class_name} - P90 Saliency", fontsize=12)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(saliency_p90, cmap='hot')
        axes[i, 1].set_title("P90 Saliency Map", fontsize=12)
        axes[i, 1].axis("off")

        for j, layer_name in enumerate(target_layer_names):
            axes[i, j + 2].imshow(grad_cams_p90[layer_name], cmap='jet')
            axes[i, j + 2].set_title(f"Grad-CAM P90 ({layer_name})", fontsize=12)
            axes[i, j + 2].axis("off")

    plt.suptitle("High-Contrast Explanations Across Classes", fontsize=16, y=1.02)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()


def visualize_explanations_for_specific_class(
    model, dataset, device, 
    target_layer_names, class_name, 
    image_amount=3, seed=42
):
    model.eval()
    random.seed(seed)

    classes = dataset.classes
    if class_name not in classes:
        raise ValueError(f"Invalid class: '{class_name}'")
    
    images_for_class = [(img, label) for img, label in dataset if classes[label] == class_name]
    selected_images = random.sample(images_for_class, min(image_amount, len(images_for_class)))

    grad_cam = GradCAM(model, target_layer_names=target_layer_names)

    num_layers = len(target_layer_names)
    num_cols = 3 + num_layers
    fig, axes = plt.subplots(image_amount, num_cols, figsize=(4 * num_cols, 4 * image_amount))

    for i, (img, label) in enumerate(selected_images):
        img_tensor = img.unsqueeze(0).to(device)
        img_tensor.requires_grad = True
        
        outputs, _ = model(img_tensor)
        label_tensor = torch.tensor([dataset.classes.index(class_name)], device=device)
        loss = margin_loss(outputs, label_tensor)
        loss.backward()

        saliency = img_tensor.grad.abs().squeeze()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        cams = grad_cam(img_tensor, class_index=label)
        
        target_class = dataset.classes.index(class_name)
        ig = integrated_gradients(model, img, device, target_class)
        ig_map = np.sum(ig, axis=0)
        ig_map_norm = (ig_map - ig_map.min()) / (ig_map.max() - ig_map.min() + 1e-8)

        axes[i, 0].imshow(img.squeeze().cpu().numpy(), cmap='gray')
        if i == 0:
            axes[i, 0].set_title(f"Original Image", fontsize=12)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(saliency.cpu().numpy(), cmap='hot')
        if i == 0:
            axes[i, 1].set_title("Saliency Map", fontsize=12)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(ig_map_norm, cmap='jet')
        if i == 0:
            axes[i, 2].set_title("Integrated Gradients", fontsize=12)
        axes[i, 2].axis("off")

        for j, layer_name in enumerate(target_layer_names):
            axes[i, j + 3].imshow(cams[layer_name], cmap='jet')
            if i == 0:
                axes[i, j + 3].set_title(f"Grad-CAM ({layer_name})", fontsize=12)
            axes[i, j + 2].axis("off")

    plt.suptitle(f"{class_name} class", fontsize=16, y=1.02)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()