"""
Enhancement tier modules.

Each module exposes:
    process(input_path, output_path, models, status_cb) -> dict

where models is a ClearVoiceModels instance and the dict contains metadata
about what was done (pipeline_used, separated_streams, ...).
"""
