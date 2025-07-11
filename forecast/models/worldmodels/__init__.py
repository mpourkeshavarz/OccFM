from forecast.models.model_template import ModelTemplate

__all__ = {
    'ModelTemplate' : ModelTemplate,
}


def build_wm(model_cfg, dataset_cfg):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, dataset=dataset_cfg
    )
    return model
