import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(
        self,
        model
    ):
        self.model = model
        self.act = None
        self.grad = None
        self.handles = []
        
    def get_activation(self, module, input, output):
        self.act = output.detach()

    def get_gradient(self, module, grad_input, grad_output):
        grad = grad_output[0]
        self.grad = grad.detach()

    def hook_target_layer(
        self, 
        target_layer=-1
    ):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        handle = self.model.backbone.features[target_layer].register_forward_hook(self.get_activation)
        self.handles.append(handle)
        handle = self.model.backbone.features[target_layer].register_full_backward_hook(self.get_gradient)
        self.handles.append(handle)

    def __call__(
        self, 
        input_tensor, 
        target_class_idx=None
    ):            
        assert input_tensor.size(0) == 1, 'Batch size must be 1!'
        
        self.model.zero_grad()
        output, __ = self.model(input_tensor)
        
        if target_class_idx is None:
            target_class_idx = torch.argmax(output, dim=1).item()

        output[:, target_class_idx].backward()

        act_map = self.act
        grad_map = self.grad
            
        pooled_gradients = torch.mean(grad_map, dim=[0, 2, 3])

        weighted_act = act_map * pooled_gradients[None,:,None,None]
        gradcam_map = torch.sum(weighted_act, dim=1)
        gradcam_map = F.relu(gradcam_map)

        gradcam_map = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min() + 1e-8)

        return gradcam_map