from pydantic import validate_call
import json
from typing import Any
import torch
import logging

logger = logging.getLogger(__name__)

@validate_call
def create_images(
    *,
    zarr_dir: str,
    num_images: int = 2,
) -> dict[str, Any]:
    """
    Task description

    Args:
        zarr_dir: Description of `zarr_dir`
        num_images: Description of `num_images`
    """
    # Validate that torch is working
    x = torch.rand(5, 3)
    logger.info(f"[validate_torch] torch.rand(5, 3) = {x}") 
    logger.info(f"[validate_torch] torch.__version__ = {torch.__version__}")
    logger.info(f"[validate_torch] torch.cuda.is_available() = {torch.cuda.is_available()}")
    
    
    logger.info("[create_images] START")
    logger.info(f"[create_images] {zarr_dir}")
    logger.info(f"[create_images] {num_images=}")
    zarr_dir = zarr_dir.rstrip("/")
    output = dict(
        image_list_updates=[
            dict(zarr_url=f"{zarr_dir}/{ind}") for ind in range(num_images)
        ]
    )
    logger.info(f"[create_images] {json.dumps(output)}")
    logger.info("[create_images] END")
    return output
    


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=create_images)
