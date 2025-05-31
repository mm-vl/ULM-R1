from .r_base import (
    format_reward,
    format_tag_reward,
    common_cls_accuracy_reward,
    common_qa_accuracy_reward,
)
from .r_t2i import (
    t2i_clip_reward,
    t2i_cycle_consistency_reward,
    t2i_bidirectional_cycle_reward,
    t2i_match_reward,
    t2i_pixel_mse_reward,
    t2i_qa_reward,
    t2i_obj_det_reward,
    t2i_obj_cls_reward,
)

reward_funcs_registry = {
    # cot format
    "format": format_reward,
    "format_tag": format_tag_reward,
    # qa
    "qa_accuracy": common_qa_accuracy_reward,
    # object classification
    "oc_accuracy": common_cls_accuracy_reward,

    # text-to-image generation
    "t2i_clip_reward": t2i_clip_reward,
    "t2i_cycle_reward": t2i_cycle_consistency_reward,
    "t2i_bid_cycle_reward": t2i_bidirectional_cycle_reward,
    "t2i_ti_sim": t2i_match_reward,
    "t2i_pixel_mse": t2i_pixel_mse_reward,
    "t2i_qa": t2i_qa_reward,
    "t2i_oc": t2i_obj_cls_reward,
    "t2i_od": t2i_obj_det_reward,
}
