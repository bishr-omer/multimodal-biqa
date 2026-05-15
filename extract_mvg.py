import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy
from scipy.ndimage import convolve
import warnings
warnings.filterwarnings('ignore')

# ── Log-Gabor Filter Bank ─────────────────────────────────────────
def precompute_log_gabor_filters(img_size, scales, orientations):
    rows, cols = img_size
    u, v = np.meshgrid(
        (np.arange(cols) - (cols // 2)) / (cols / 2),
        (np.arange(rows) - (rows // 2)) / (rows / 2)
    )
    radius = np.sqrt(u**2 + v**2)
    radius[radius == 0] = 1e-9
    theta = np.arctan2(v, u)

    sigma_f = 0.65
    sigma_theta = 0.3

    filter_bank = []
    for scale in scales:
        center_freq = scale
        for orientation in orientations:
            radial = np.exp(-(np.log(radius / center_freq))**2 / (2 * sigma_f**2))
            theta_diff = np.mod(theta - orientation, np.pi)
            theta_diff = np.minimum(theta_diff, np.pi - theta_diff)
            angular = np.exp(-theta_diff**2 / (2 * sigma_theta**2))
            f = radial * angular
            f[0, 0] = 0
            filter_bank.append(np.fft.ifftshift(f))
    return filter_bank

def extract_log_gabor_features(image, filter_bank):
    img_fft = np.fft.fft2(image.astype(np.float64))
    features = []
    for f in filter_bank:
        response = np.fft.ifft2(img_fft * f)
        mag = np.abs(response)
        stats = [np.mean(mag), np.std(mag), np.mean(mag**2)]
        features.extend(stats)
    return np.array(features)

# ── Edge Features ─────────────────────────────────────────────────
def extract_edge_features(img):
    img = img.astype(np.float64)
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Gx = convolve(img, Kx)
    Gy = convolve(img, Ky)
    mag = np.sqrt(Gx**2 + Gy**2)
    return np.array([np.mean(mag), np.std(mag), np.mean(mag**2), np.max(mag)])

# ── Precompute Filter Bank Once Globally ─────────────────────────
BLOCK_SIZE = 32
SCALES = [1.0, 0.25]
ORIENTATIONS = [0, np.pi/4, np.pi/2, 3*np.pi/4]
FILTER_BANK = precompute_log_gabor_filters((BLOCK_SIZE, BLOCK_SIZE), SCALES, ORIENTATIONS)

# ── Spatial Feature Extractor ─────────────────────────────────────
def spatfeature_fast(block):
    """Extract 48 spatial features using precomputed global filter bank."""
    img = block.astype(np.float64)
    I = img.flatten()

    # Basic statistics (5)
    m   = np.mean(I)
    s   = np.std(I)
    sk  = skew(I) if len(I) > 1 else 0
    ku  = kurtosis(I) if len(I) > 1 else 0
    ent = shannon_entropy(img)
    base_f = np.nan_to_num(np.array([m, s, sk, ku, ent]))

    # Edge features (4)
    edge_f = extract_edge_features(img)

    # LBP features (15)
    lbp = local_binary_pattern(img, P=4, R=2, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=15, range=(0, 15), density=True)
    lbp_f = lbp_hist.astype(np.float64)

    # Log-Gabor features (24)
    lg_f = extract_log_gabor_features(img, FILTER_BANK)

    features = np.concatenate([base_f, edge_f, lbp_f, lg_f])
    return features

# ── Spectral Feature Extractor ────────────────────────────────────
def extract_fdd_features(image):
    """First Digit Distribution features (9)"""
    first_digits = []
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    for b in range(image.shape[2]):
        band = np.abs(image[:, :, b]).flatten().astype(np.float64)
        band = band[(band > 0) & np.isfinite(band)]
        if len(band) > 0:
            digits = np.floor(band / 10**np.floor(np.log10(band)))
            first_digits.extend(digits.tolist())
    if len(first_digits) == 0:
        return np.zeros(9)
    counts, _ = np.histogram(first_digits, bins=np.arange(0.5, 10.5))
    total = counts.sum()
    return counts / total if total > 0 else np.zeros(9)

def extract_color_moments(image):
    """Color moments: mean, std, skewness per channel (12)"""
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)
    channels = min(image.shape[2], 4)
    features = []
    for i in range(channels):
        ch = image[:, :, i].flatten().astype(np.float64)
        features.extend([np.mean(ch), np.std(ch), skew(ch)])
    while len(features) < 12:
        features.append(0.0)
    return np.array(features[:12])

def specfeature(block):
    """Extract 21 spectral features from a single image block."""
    fdd_f    = extract_fdd_features(block)
    moment_f = extract_color_moments(block)
    features = np.concatenate([fdd_f, moment_f])
    return np.nan_to_num(features)

# ── Main Feature Extractor ────────────────────────────────────────
def extract_mvg_features(image_path, block_size=32):
    """Extract spatial + spectral features from a full image."""
    img  = np.array(Image.open(image_path).convert('RGB'))
    gray = np.array(Image.open(image_path).convert('L')).astype(np.float64)

    h, w   = gray.shape
    h_crop = (h // block_size) * block_size
    w_crop = (w // block_size) * block_size
    gray   = gray[:h_crop, :w_crop]
    img    = img[:h_crop, :w_crop, :]

    spat_features = []
    spec_features = []

    for i in range(0, h_crop, block_size):
        for j in range(0, w_crop, block_size):
            gray_block = gray[i:i+block_size, j:j+block_size]
            rgb_block  = img[i:i+block_size, j:j+block_size, :]
            spat_features.append(spatfeature_fast(gray_block))
            spec_features.append(specfeature(rgb_block))

    spat = np.array(spat_features)
    spec = np.array(spec_features)

    spat_agg = np.concatenate([np.mean(spat, axis=0), np.std(spat, axis=0)])
    spec_agg = np.concatenate([np.mean(spec, axis=0), np.std(spec, axis=0)])

    return np.concatenate([spat_agg, spec_agg])

# ── Run on Dataset ────────────────────────────────────────────────
def extract_dataset(image_folder, output_path):
    image_folder = Path(image_folder)
    image_paths  = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))
    print(f"Found {len(image_paths)} images")

    names    = []
    features = []

    for img_path in tqdm(image_paths, desc="Extracting MVG features"):
        try:
            feat = extract_mvg_features(img_path)
            names.append(img_path.name)
            features.append(feat)
        except Exception as e:
            print(f"Skipped {img_path.name}: {e}")

    features = np.array(features)
    np.save(output_path, {"names": names, "features": features})
    print(f"Saved {len(names)} feature vectors → shape {features.shape}")

if __name__ == "__main__":
    extract_dataset(
        image_folder="data/koniq10k/images",
        output_path="results/koniq_mvg_features.npy"
    )