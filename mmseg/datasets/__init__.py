'''
Author: Shuailin Chen
Created Date: 2021-07-12
Last Modified: 2021-08-04
	content: 
'''
from .ade import ADE20KDataset
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .chase_db1 import ChaseDB1Dataset
from .cityscapes import CityscapesDataset
from .custom import CustomDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .drive import DRIVEDataset
from .hrf import HRFDataset
from .pascal_context import PascalContextDataset, PascalContextDataset59
from .stare import STAREDataset
from .voc import PascalVOCDataset
from .sar_building import Sar_building
from .sar_building_semi import Sar_building_semi
from .npy_dataset import NpyDataset
from .sar_multich import Sar_multich
from .sar_rotate import Sar_Rotate
from .domain_adaptation import DomainAdaptationDataset
from .sar_building_domain_adaptation import SARBuildingDomainAdaptation
from .my_dataset_wrappers import MyConcatDataset




# __all__ = [
#     'CustomDataset', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
#     'DATASETS', 'build_dataset', 'PIPELINES', 'CityscapesDataset',
#     'PascalVOCDataset', 'ADE20KDataset', 'PascalContextDataset',
#     'PascalContextDataset59', 'ChaseDB1Dataset', 'DRIVEDataset', 'HRFDataset',
#     'STAREDataset'
# ]
