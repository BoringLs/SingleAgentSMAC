from collections import namedtuple

HyperParameter = namedtuple(
    "HyperParameter",
    [
        "action_num",
        "max_entity",
        "entity_num",
        "enemy_num",
        "entity_size",
        "map_channel",
        "scalar_size",
        "bias_value",
        "embedded_entity_size",
        "embedded_spatial_size",
        "total_embedded_size",
        "autoregressive_embedding_size",
        "embedded_state_size",
        "hidden_size",
        "max_map_channel",
        "original_16",
        "original_32",
        "original_64",
        "original_128",
        "original_256",
    ],
)


hyper_parameters = HyperParameter(
    action_num=14,
    max_entity=8,
    entity_num=8,
    enemy_num=8,
    # preprocess params
    entity_size=80,
    map_channel=4,
    scalar_size=13,
    bias_value=-1e9,
    # encoder out params
    embedded_entity_size=64,
    embedded_spatial_size=64,
    total_embedded_size=128,
    autoregressive_embedding_size=256,  # should equal to 16 x 16
    # core params
    embedded_state_size=128,
    hidden_size=128,
    max_map_channel=64,
    # hidden layer params
    original_16=16,
    original_32=32,
    original_64=64,
    original_128=128,
    original_256=256,
)
