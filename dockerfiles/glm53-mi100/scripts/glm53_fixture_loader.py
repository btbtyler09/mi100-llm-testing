"""Test-only loader for four genuine checkpoint layers; never used in serving."""
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from probe_real_weights import selected_weights


@register_model_loader("glm53-four-layer-test")
class FourLayerLoader(BaseModelLoader):
    def download_model(self, model_config):
        return None

    def load_weights(self, model, model_config):
        loaded = model.load_weights(selected_weights())
        missing = set(dict(model.named_parameters())) - loaded
        # These are runtime-only act-order scratch arrays for a checkpoint
        # with actorder=None, rather than weights serialized in safetensors.
        scratch = {"w13_weight_g_idx", "w2_weight_g_idx",
                   "w13_g_idx_sort_indices", "w2_g_idx_sort_indices"}
        assert all(name.rsplit(".", 1)[-1] in scratch for name in missing), missing


def four_layer_config(config):
    if not hasattr(config, "text_config"):
        return config
    text = config.text_config
    text.num_hidden_layers = 4
    text.layer_types = text.layer_types[:4]
    text.mlp_layer_types = text.mlp_layer_types[:4]
    return config
