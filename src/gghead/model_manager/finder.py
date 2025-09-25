import os
from elias.folder import ModelFolder
from elias.manager import ModelManager

from gghead.model_manager.gghead_model_manager import GGHeadModelFolder

_MODEL_FOLDERS_CLASSES = [
    GGHeadModelFolder,
]


def find_model_folder(run_name: str) -> ModelFolder:
    for model_folder_cls in _MODEL_FOLDERS_CLASSES:
        model_folder = model_folder_cls()
        if model_folder.resolve_run_name(run_name) is not None:
            return model_folder
        
        if run_name.startswith("gh") and len(run_name) >= 3:
            for d in model_folder._folder.ls():
                if d.startswith(run_name):
                    return model_folder
    
    raise ValueError(f"Could not locate model folder for run {run_name}. Is the run name correct?")


def full_run_name(run_name):
    for model_folder_cls in _MODEL_FOLDERS_CLASSES:
        model_folder = model_folder_cls()
        
        if model_folder.resolve_run_name(run_name) is not None:
            return run_name
        if run_name.startswith("gh") and len(run_name) >= 3:
            for d in model_folder._folder.ls():
                if d.startswith(run_name):
                    return d
    raise ValueError(f"Could not resolve full run name for {run_name}. Is the run name correct?")

def find_model_manager(run_name: str) -> ModelManager:
    run_name = full_run_name(run_name)
    model_folder = find_model_folder(run_name)
    return model_folder.open_run(run_name)
