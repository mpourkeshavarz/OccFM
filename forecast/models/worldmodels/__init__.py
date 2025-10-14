from forecast.models.model_template import ModelTemplate
from forecast.models.worldmodels.occfm import OccFM

__all__ = {
    'ModelTemplate' : ModelTemplate,
    'OccFM': OccFM,
}


def build_wm(model_cfg, loss_cfg, other_cfg):

    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, loss_cfg=loss_cfg, other_cfg=other_cfg
    )
    return model
