# Explainers
from openbinn.explainers import Gradient, GradientShap, IntegratedGradients,\
    InputTimesGradient, SmoothGrad, LIME, DeepLiftShapExplainer, RandomBaseline, LRP, DeepLift, FeatureAblation
import inspect

explainers_dict = {
    'grad': Gradient,
    'gradshap': GradientShap,
    'sg': SmoothGrad,
    'itg': InputTimesGradient,
    'ig': IntegratedGradients,
    'shap': DeepLiftShapExplainer,
    'lime': LIME,
    "lrp": LRP,
    'control': RandomBaseline, 
    'deeplift': DeepLift, 
    'feature_ablation': FeatureAblation
}

def Explainer(method, model, param_dict=None):
    """
    Returns an explainer object for the given method
    :param method: str, name of the method
    :param model: PyTorch model or function
    :param param_dict: dict, __init__ parameters dictionary for the explainer
    :return: explainer object
    """
    if method not in explainers_dict.keys():
        raise NotImplementedError("This method has not been implemented, yet.")

    if param_dict is None:
        param_dict = {}

    if method in ['lime', 'ig'] and param_dict == {}:
        raise ValueError(
            f"Please provide training data for {method} using param_dict = "
            "openbinn.experiment_utils.fill_param_dict('{method}', {}, X_train)"
        )

    expl_cls = explainers_dict[method]

    # gradient based explainers shouldn't receive a baseline tensor
    if method in {'itg', 'sg', 'grad', 'gradshap'}:
        param_dict = {k: v for k, v in param_dict.items() if k != 'baseline'}
    sig = inspect.signature(expl_cls.__init__)
    valid_params = {k: v for k, v in param_dict.items() if k in sig.parameters}

    explainer = expl_cls(model, **valid_params)
    return explainer
