from monai.transforms import MapTransform
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

##---------BRATS-------------------
class CombineTumorLabels(MapTransform):
    """
    A transform class to combine all tumor sub-regions into a single label.

    This transform modifies the `label` key in the input dictionary.
    All non-zero labels are converted to `1`, representing a unified tumor region.

    Args:
        data (dict): Input dictionary with a "label" key, containing the label array.

    Returns:
        dict: Updated dictionary with combined tumor labels.
    """
    def __call__(self, data):
        data["label"][data["label"] > 0] = 1  # Combine all tumor regions
        return data

class SplitSubRegions(MapTransform):
    """
    A transform class to split the tumor label into separate sub-regions.

    This transform creates new keys in the input dictionary for specific sub-regions:
    - `enhancing_tumor`: Regions with label value 4.
    - `edema`: Regions with label value 2.

    Args:
        data (dict): Input dictionary with a "label" key, containing the label array.

    Returns:
        dict: Updated dictionary with new keys for the tumor sub-regions.
    """
    def __call__(self, data):
        label = data["label"]
        data["enhancing_tumor"] = (label == 3).astype(int)  
        data["non-enhancing_tumor"] = (label == 2).astype(int)  
        data["edema"] = (label == 1).astype(int)  
        return data
    
##---------AMOS-------------------
class AmosMapUnseenClasses(MapTransform):
    """
    Map the unseen classes to background (0) for training.
    """
    def __call__(self, data):
        label = data["label"]
        # Map unseen classes (11-15) to background
        label[(label >= 11) & (label <= 15)] = 0
        data["label"] = label
        return data
    
class BtcvMapUnseenClasses(MapTransform):
    """
    Map the unseen classes to background (0) for training.
    """
    def __call__(self, data):
        label = data["label"]
        # Map unseen classes (11-13) to background
        label[(label >= 11) & (label <= 13)] = 0
        data["label"] = label
        return data
    
class MSDMapUnseenClasses(MapTransform):
    """
    Map the unseen classes to background (0) for training. Excluding tumor class.
    """
    def __call__(self, data):
        label = data["label"]
        # Map unseen classes (11-15) to background
        label[(label >= 2)] = 0
        data["label"] = label
        return data
    

##---------UTILS-------------------
    
def extract_one_shot_unknown_sample(data_loader):
    """
    Extracts one-shot labeled data for unseen classes (labels > 10).
    
    Args:
        data_loader: Dataloader containing validation samples.

    Returns:
        sample (dict): A single sample containing unknown labels.
    """
    for batch in data_loader:
        labels = batch["label"]
        if (labels > 1).any():  # TO DO Automatize per dataset
            print("[INFO] Found an example with unknown class.")
            return batch  # Return this batch as the one-shot example
    print("[WARNING] No unknown class found in validation set.")
    return None



def create_body_mask(ct_volume, min_val=-2000, connectivity=3):
    """
    Create a body mask from a 3D CT volume using Otsu thresholding and largest connected component.
    """
    flat_pixels = ct_volume.flatten()
    flat_pixels = flat_pixels[flat_pixels > min_val]

    otsu_thresh = threshold_otsu(flat_pixels)
    mask = (ct_volume > otsu_thresh)
    # Label connected components
    labeled_mask = label(mask, connectivity=connectivity)
    # Find the largest connected component by area
    max_area = 0
    largest_label = 0
    for prop in regionprops(labeled_mask):
        if prop.area > max_area:
            max_area = prop.area
            largest_label = prop.label
    # Keep only the largest component
    body_mask = (labeled_mask == largest_label)
    return body_mask
    
class DebugShape(MapTransform):
    def __call__(self, data):
        print(f"Image shape: {data['image'].shape}, Label shape: {data['label'].shape}")
        return data