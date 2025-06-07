import numpy as np
import torch
from ...api import BaseExplainer
from openbinn.experiment_utils import convert_to_numpy

# import lime
from .lime_package import lime_tabular
from .lime_package import lime_image

class LIME(BaseExplainer):
    """
    This class gets explanations using LIME.

    model : original model (torch.nn.Module)
    data : training data as torch.FloatTensor
    mode : str, "tabular" or "images"
    """

    def __init__(self, model, data: torch.FloatTensor, std: float = 0.1,
                 n_samples: int = 1000, kernel_width: float = 0.75,
                 sample_around_instance: bool = True, mode: str = "tabular",
                 discretize_continuous: bool = False, seed=None) -> None:
        
        self.output_dim = 2
        # 데이터를 numpy array로 변환 후, 2D가 아닐 경우 2D로 평탄화
        data_np = data.numpy()
        if data_np.ndim > 2:
            # 원래의 shape 저장 (예: (num_genes, num_features))
            self.original_shape = data_np.shape[1:]
            # 각 샘플을 평탄화하여 2D 형태로 변환
            data_np = data_np.reshape(data_np.shape[0], -1)
        else:
            self.original_shape = None
        self.data = data_np
        self.mode = mode
        # 모델과 predict 함수 설정
        self.model_module = model
        self.predict_function = model.predict
        self.n_samples = n_samples
        self.discretize_continuous = discretize_continuous
        self.sample_around_instance = sample_around_instance
        self.seed = seed

        # Tabular mode인 경우, numpy 배열을 입력받는 새로운 predict 함수 정의
        if self.mode == "tabular":
            def predict_np(x):
                x_tensor = torch.FloatTensor(x)
                with torch.no_grad():
                    out = self.predict_function(x_tensor)
                return out.detach().numpy()
            self.predict_function_np = predict_np
            self.explainer = lime_tabular.LimeTabularExplainer(
                self.data,
                mode="classification",
                sample_around_instance=self.sample_around_instance,
                discretize_continuous=self.discretize_continuous,
                # 평탄화된 데이터의 feature 수를 기반으로 kernel_width 설정
                kernel_width=kernel_width * np.sqrt(self.data.shape[1]),
                std=std
            )
        else:
            self.predict_function_np = self.predict_function  # for image mode
            self.explainer = lime_image.LimeImageExplainer()

        super(LIME, self).__init__(self.predict_function_np)

    def get_explanations(self, x: torch.FloatTensor, label=None) -> torch.FloatTensor:
        if label is None:
            label = self.predict_function_np(x.float().numpy()).argmax(axis=-1)
        else:
            label = convert_to_numpy(label)
        label = np.repeat(label, x.shape[0]) if label.shape == () else label

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        if self.mode == "tabular":
            x_np = x.numpy()
            if x_np.ndim > 2:
                x_np = x_np.reshape(x_np.shape[0], -1)
            num_features = x_np.shape[1]
            attribution_scores = np.zeros(x_np.shape)
            for i in range(x_np.shape[0]):
                exp = self.explainer.explain_instance(x_np[i, :], self.predict_function_np,
                                                      num_samples=self.n_samples,
                                                      num_features=num_features)
                for feature_idx, feature_attribution in exp.local_exp[1]:
                    attribution_scores[i, feature_idx] = feature_attribution * (2 * label[i] - 1)
            return torch.FloatTensor(attribution_scores)
        else:
            attribution_scores = []
            for i in range(x.shape[0]):
                exp = self.explainer.explain_instance(x,
                                                      self.predict_function_np,
                                                      top_labels=5,
                                                      hide_color=0,
                                                      num_samples=self.n_samples)
                attribution_scores.append(exp)
            return torch.FloatTensor(attribution_scores)

    def get_layer_explanations(self, inputs, label=None):
        """
        Returns explanations for each layer in the model (tabular mode only).
        """
        explanations = {}
        current_input = inputs.clone().detach()

        for name, layer in self.model_module.named_children():
            if self.mode != "tabular":
                raise NotImplementedError("get_layer_explanations is only implemented for tabular mode.")

            # Forward pass through the current layer.
            current_output = layer(current_input)
            
            # 보존: 원래의 shape 확인 (batch, d1, d2) 또는 2D인 경우 그대로 사용
            x_np = current_input.numpy()
            if x_np.ndim > 2:
                original_shape = x_np.shape[1:]  # 예: (num_genes, num_features)
                x_np_flat = x_np.reshape(x_np.shape[0], -1)
            else:
                original_shape = None
                x_np_flat = x_np

            flat_feature_dim = x_np_flat.shape[1]
            attribution_scores = np.zeros((x_np_flat.shape[0], flat_feature_dim))
            
            for i in range(x_np_flat.shape[0]):
                exp = self.explainer.explain_instance(
                    x_np_flat[i, :], 
                    self.predict_function_np,
                    num_samples=self.n_samples,
                    num_features=flat_feature_dim
                )
                # LIME returns local_exp as a dict; 여기서는 클래스 1의 결과를 사용 (예제와 동일)
                for feature_idx, feature_attribution in exp.local_exp[1]:
                    # feature_idx가 flat_feature_dim 범위를 벗어나지 않는지 확인
                    if feature_idx < flat_feature_dim:
                        # label이 주어졌으면 해당 값을 사용, 아니면 기본값 1
                        mult = (2 * (label[i] if label is not None else 1) - 1)
                        attribution_scores[i, feature_idx] = feature_attribution * mult

            # 원래의 shape으로 재구성
            if original_shape is not None:
                attribution_scores = attribution_scores.reshape(x_np.shape[0], *original_shape)
            
            explanations[name] = torch.FloatTensor(attribution_scores)
            # 다음 레이어를 위해 current_input 업데이트
            current_input = current_output

        return explanations
