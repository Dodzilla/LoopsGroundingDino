from .GroundingDinoSAM2SegmentV2 import GroundingDinoSAM2SegmentV2, GroundingDinoSAM2SegmentV2Ex
from .node import SAM2ModelLoader, GroundingDinoModelLoader, GroundingDinoSAM2Segment, InvertMask, IsMaskEmptyNode

NODE_CLASS_MAPPINGS = {
    'SAM2ModelLoader (segment anything2)': SAM2ModelLoader,
    'GroundingDinoModelLoader (segment anything2)': GroundingDinoModelLoader,
    'GroundingDinoSAM2Segment (segment anything2)': GroundingDinoSAM2Segment,
    'InvertMask (segment anything)': InvertMask,
    "IsMaskEmpty": IsMaskEmptyNode,
    "GroundingDinoSAM2SegmentV2": GroundingDinoSAM2SegmentV2,
    "GroundingDinoSAM2SegmentV2Ex": GroundingDinoSAM2SegmentV2Ex,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'SAM2ModelLoader': 'SAM2 Model Loader',
    'GroundingDinoModelLoader': 'Grounding Dino Model Loader',
    'GroundingDinoSAM2Segment': 'Grounding Dino SAM2 Segment',
    'InvertMask': 'Invert Mask',
    "IsMaskEmpty": "Is Mask Empty",
    "GroundingDinoSAM2SegmentV2": "GroundingDINO + SAM2 Segment (V2: score+NMS+mid-quantile)",
    "GroundingDinoSAM2SegmentV2Ex": "GroundingDINO + SAM2 Segment (V2Ex: multi-run + multimask)",
}
