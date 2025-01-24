from typing import List, Optional
from dataclasses import dataclass, field

@dataclass
class WandbConfig:
    project: str = 'recipes' # wandb project name
    entity: Optional[str] = None # wandb entity name
    job_type: Optional[str] = None
    tags: Optional[List[str]] = None
    group: Optional[str] = None
    notes: Optional[str] = None
    mode: Optional[str] = None