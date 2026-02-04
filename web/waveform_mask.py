"""
ECG Waveform Masking for 3x4 Layout

Standard 12-lead ECG layout (3 rows x 4 columns):
    Column 0    Column 1    Column 2    Column 3
    (0-1250)   (1250-2500) (2500-3750) (3750-5000)
Row 0:  I         aVR         V1          V4
Row 1:  II        aVL         V2          V5
Row 2:  III       aVF         V3          V6

Lead II is the rhythm strip - it spans the full 5000 samples.
Other leads only have data in their respective quarter.

This module provides functions to apply the masking both with PyTorch and NumPy.
"""

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# Lead ordering for standard 12-lead ECG
LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Lead indices
LEAD_I = 0
LEAD_II = 1
LEAD_III = 2
LEAD_AVR = 3
LEAD_AVL = 4
LEAD_AVF = 5
LEAD_V1 = 6
LEAD_V2 = 7
LEAD_V3 = 8
LEAD_V4 = 9
LEAD_V5 = 10
LEAD_V6 = 11


def create_mask_numpy(total_samples: int = 5000, num_leads: int = 12) -> np.ndarray:
    """
    Create a mask for 12-lead ECG in 3x4 layout.

    Each lead only has data in its respective time quarter,
    except Lead II which is the rhythm strip (full duration).

    Args:
        total_samples: Total number of samples (default 5000)
        num_leads: Number of leads (default 12)

    Returns:
        mask: numpy array of shape (total_samples, num_leads)
              1.0 where data should be preserved, 0.0 where it should be zeroed
    """
    steps = total_samples // 4  # 1250 samples per quarter
    mask = np.zeros((total_samples, num_leads), dtype=np.float32)

    for i in range(num_leads):
        # Determine which quarter this lead belongs to
        # Leads 0-2 → quarter 0, leads 3-5 → quarter 1, etc.
        quarter = i // 3
        start = quarter * steps
        end = (quarter + 1) * steps
        mask[start:end, i] = 1.0

    # Lead II (index 1) is the rhythm strip - it spans the full recording
    mask[:, LEAD_II] = 1.0

    return mask


def create_mask_torch(total_samples: int = 5000, num_leads: int = 12, device=None, dtype=None):
    """
    Create a mask for 12-lead ECG in 3x4 layout using PyTorch.

    Args:
        total_samples: Total number of samples (default 5000)
        num_leads: Number of leads (default 12)
        device: torch device (default: cpu)
        dtype: torch dtype (default: float32)

    Returns:
        mask: torch tensor of shape (total_samples, num_leads)
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch not available. Use create_mask_numpy instead.")

    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32

    steps = total_samples // 4
    mask = torch.zeros(total_samples, num_leads, device=device, dtype=dtype)

    for i in range(num_leads):
        quarter = i // 3
        mask[quarter * steps:(quarter + 1) * steps, i] = 1.0

    # Lead II (index 1) is the rhythm strip
    mask[:, LEAD_II] = 1.0

    return mask


def apply_mask_numpy(waveform: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Apply the 3x4 layout mask to waveform data.

    Args:
        waveform: numpy array of shape (batch, samples, leads) or (samples, leads)
        mask: optional pre-computed mask

    Returns:
        masked waveform with same shape as input
    """
    if mask is None:
        if waveform.ndim == 3:
            samples, leads = waveform.shape[1], waveform.shape[2]
        else:
            samples, leads = waveform.shape
        mask = create_mask_numpy(samples, leads)

    if waveform.ndim == 3:
        # Batch dimension present: (batch, samples, leads)
        return waveform * mask[np.newaxis, :, :]
    else:
        # No batch dimension: (samples, leads)
        return waveform * mask


def apply_mask_torch(waveform, mask=None):
    """
    Apply the 3x4 layout mask to waveform data using PyTorch.

    Args:
        waveform: torch tensor of shape (batch, samples, leads) or (samples, leads)
        mask: optional pre-computed mask

    Returns:
        masked waveform with same shape as input
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch not available. Use apply_mask_numpy instead.")

    if mask is None:
        if waveform.ndim == 3:
            samples, leads = waveform.shape[1], waveform.shape[2]
        else:
            samples, leads = waveform.shape
        mask = create_mask_torch(samples, leads, device=waveform.device, dtype=waveform.dtype)

    if waveform.ndim == 3:
        # Batch dimension present: (batch, samples, leads)
        return waveform * mask.unsqueeze(0)
    else:
        # No batch dimension: (samples, leads)
        return waveform * mask


def pad_and_mask_leads(leads_dict: dict, target_samples: int = 5000) -> dict:
    """
    Pad leads to target length and apply 3x4 layout mask.

    This is the main function for preparing extracted ECG leads for the API.

    Args:
        leads_dict: Dictionary with lead names as keys and sample arrays as values
                   e.g., {'I': [samples], 'II': [samples], ...}
        target_samples: Target number of samples (default 5000)

    Returns:
        Dictionary with same keys but padded and masked sample arrays
    """
    # Create a full 12-lead waveform array
    waveform = np.zeros((target_samples, 12), dtype=np.float32)

    # Map lead names to indices
    lead_name_to_idx = {name: i for i, name in enumerate(LEAD_ORDER)}

    # Fill in the waveform
    for lead_name, samples in leads_dict.items():
        if lead_name in lead_name_to_idx:
            idx = lead_name_to_idx[lead_name]
            samples_arr = np.array(samples, dtype=np.float32)

            # Pad or truncate to target length
            if len(samples_arr) < target_samples:
                # Pad with zeros at the end
                padded = np.zeros(target_samples, dtype=np.float32)
                padded[:len(samples_arr)] = samples_arr
                samples_arr = padded
            elif len(samples_arr) > target_samples:
                samples_arr = samples_arr[:target_samples]

            waveform[:, idx] = samples_arr

    # Apply the mask
    mask = create_mask_numpy(target_samples, 12)
    masked_waveform = waveform * mask

    # Convert back to dictionary
    result = {}
    for lead_name in leads_dict.keys():
        if lead_name in lead_name_to_idx:
            idx = lead_name_to_idx[lead_name]
            result[lead_name] = masked_waveform[:, idx].tolist()

    return result


# Example usage and testing
if __name__ == "__main__":
    print("Testing waveform masking...")

    # Create sample data
    leads = {
        'I': np.random.randn(2500).tolist(),
        'II': np.random.randn(2500).tolist(),
        'III': np.random.randn(2500).tolist(),
        'aVR': np.random.randn(2500).tolist(),
        'aVL': np.random.randn(2500).tolist(),
        'aVF': np.random.randn(2500).tolist(),
        'V1': np.random.randn(2500).tolist(),
        'V2': np.random.randn(2500).tolist(),
        'V3': np.random.randn(2500).tolist(),
        'V4': np.random.randn(2500).tolist(),
        'V5': np.random.randn(2500).tolist(),
        'V6': np.random.randn(2500).tolist(),
    }

    # Apply masking
    masked = pad_and_mask_leads(leads, target_samples=5000)

    # Check results
    for lead_name in LEAD_ORDER:
        samples = np.array(masked[lead_name])
        nonzero_start = np.where(samples != 0)[0]
        if len(nonzero_start) > 0:
            first_nonzero = nonzero_start[0]
            last_nonzero = nonzero_start[-1]
            print(f"{lead_name:4s}: non-zero from {first_nonzero:4d} to {last_nonzero:4d}")
        else:
            print(f"{lead_name:4s}: all zeros")

    print("\nExpected pattern:")
    print("  I, II, III: quarter 0 (0-1250), but II spans full 0-5000")
    print("  aVR, aVL, aVF: quarter 1 (1250-2500)")
    print("  V1, V2, V3: quarter 2 (2500-3750)")
    print("  V4, V5, V6: quarter 3 (3750-5000)")
