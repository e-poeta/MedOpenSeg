from monai.transforms import (
    LoadImaged, ScaleIntensityRanged, CropForegroundd, Orientationd,
    Spacingd, EnsureTyped, RandCropByPosNegLabeld, RandFlipd, RandRotate90d, RandShiftIntensityd, Compose,
    ConvertToMultiChannelBasedOnBratsClassesd, RandSpatialCropd,NormalizeIntensityd,RandScaleIntensityd,ToTensord, Resized, Lambdad
)
from preprocess.brats import CombineTumorLabels,AmosMapUnseenClasses,BtcvMapUnseenClasses,MSDMapUnseenClasses,DebugShape
import numpy as np


def get_btcv_transforms(device, num_samples=4):
    train_transforms= Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        BtcvMapUnseenClasses(keys="label"), #Renoving unseen classes for training
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image",allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(96, 96, 96),pos=1,neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
    ])
    val_transforms=Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        BtcvMapUnseenClasses(keys="label"), #Renoving unseen classes for training
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image",allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ])
    
    test_transforms=Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image",allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ])

    return train_transforms, val_transforms,test_transforms
    
    
def get_brats_transforms(device, num_samples=4):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"],ensure_channel_first=True),
            #ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            CombineTumorLabels(keys="label"),
            CropForegroundd(
                keys=["image", "label"], source_key="image", k_divisible=[96,96,96]
            ),
            RandSpatialCropd(
                keys=["image", "label"], roi_size=[96,96,96], random_size=False
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ToTensord(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"],ensure_channel_first=True),
            #ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            CombineTumorLabels(keys="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    
    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"],ensure_channel_first=True),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    
    #
    return train_transforms, val_transforms,test_transforms


def get_amos_transforms(device, num_samples=4):
    train_transforms= Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        AmosMapUnseenClasses(keys="label"), #Renoving unseen classes for training
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True), #check this for AMOS
        CropForegroundd(keys=["image", "label"], source_key="image",allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(64,64,64),pos=1,neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
    ])
    val_transforms=Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        AmosMapUnseenClasses(keys="label"),  #Removing unseen classes for validation
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image",allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ])
    eleonora_val_transforms=Compose([ LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        AmosMapUnseenClasses(keys="label"),  #Removing unseen classes for validation
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image",allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Resized(keys=["image", "label"], spatial_size=(64, 64, 64)),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ])

    test_transforms=Compose([
        LoadImaged(keys=["image"], ensure_channel_first=True, dtype=np.float32),
        LoadImaged(keys=["label"], ensure_channel_first=True, dtype=np.int16),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image",allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ])    
    return train_transforms, val_transforms,test_transforms

    
def get_msdpancreas_transforms(device,num_samples=4):
    train_transforms= Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        MSDMapUnseenClasses(keys="label"), #Renoving unseen classes for training
        ScaleIntensityRanged(keys=["image"], a_min=-87, a_max=199, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image",allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(64, 64, 64),pos=1,neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
    ])
    val_transforms=Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        MSDMapUnseenClasses(keys="label"), #Renoving unseen classes for training
        ScaleIntensityRanged(keys=["image"], a_min=-87, a_max=199, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image",allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ])
    
    test_transforms=Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ScaleIntensityRanged(keys=["image"], a_min=-87, a_max=199, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image",allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ])

    return train_transforms, val_transforms,test_transforms
    