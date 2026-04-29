"""
Viewpoint-balanced batch sampler for Phase 1 training.
Ensures each batch has both lateral and overhead viewpoints of each species.
"""

from torch.utils.data import Sampler
from typing import Iterator, List, Dict
from collections import defaultdict
import random


class ViewpointBalancedSampler(Sampler):
    """
    Sampler that ensures each batch contains both viewpoints (lateral/overhead)
    of the same pest species when available.
    
    Used only in Phase 1 (ArcFace training).
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        viewpoint_tags: Dict[int, str],
        seed: int = 42,
    ) -> None:
        """
        Initialize sampler.
        
        Args:
            dataset: PestDataset
            batch_size: Batch size
            viewpoint_tags: Dict mapping sample_idx -> "lateral" or "overhead"
            seed: Random seed
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.viewpoint_tags = viewpoint_tags
        self.seed = seed
        
        # Group by species and viewpoint
        self.species_viewpoint_groups = defaultdict(lambda: {"lateral": [], "overhead": []})
        
        for idx in range(len(dataset)):
            _, metadata = dataset[idx]
            species_id = metadata["species_id"]
            viewpoint = viewpoint_tags.get(idx, "unknown")
            
            if viewpoint in ("lateral", "overhead"):
                self.species_viewpoint_groups[species_id][viewpoint].append(idx)
    
    def __iter__(self) -> Iterator[int]:
        """Yield indices for balanced batches."""
        random.seed(self.seed)
        
        # Prepare pools per species
        pools = {}
        for species_id, vp_dict in self.species_viewpoint_groups.items():
            lateral = vp_dict["lateral"][:]
            overhead = vp_dict["overhead"][:]
            random.shuffle(lateral)
            random.shuffle(overhead)
            pools[species_id] = {"lateral": lateral, "overhead": overhead}
        
        batch = []
        species_list = list(pools.keys())
        
        while any(pools[s]["lateral"] or pools[s]["overhead"] for s in species_list):
            for species_id in species_list:
                # Try to add one lateral + one overhead from this species
                if pools[species_id]["lateral"]:
                    batch.append(pools[species_id]["lateral"].pop())
                
                if len(batch) < self.batch_size and pools[species_id]["overhead"]:
                    batch.append(pools[species_id]["overhead"].pop())
                
                if len(batch) >= self.batch_size:
                    for idx in batch:
                        yield idx
                    batch = []
        
        # Yield remaining
        if batch:
            for idx in batch:
                yield idx
    
    def __len__(self) -> int:
        return len(self.dataset)