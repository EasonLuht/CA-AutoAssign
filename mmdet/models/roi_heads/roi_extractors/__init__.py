# Copyright (c) OpenMMLab. All rights reserved.
from .base_roi_extractor import BaseRoIExtractor
from .generic_roi_extractor import GenericRoIExtractor
from .single_level_roi_extractor import SingleRoIExtractor
from .soft_roi_selection import SoftRoIExtractor
from .all_level_auxiliary import AuxAllLevelRoIExtractor

__all__ = ['BaseRoIExtractor', 'SingleRoIExtractor', 'GenericRoIExtractor', 'SoftRoIExtractor', 'AuxAllLevelRoIExtractor']
