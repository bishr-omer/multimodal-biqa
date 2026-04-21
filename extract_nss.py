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

BLOCK_SIZE = 32
SCALES     = [1.0, 0.25]
ORIENTATIONS = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# ── Log-Gabor Filter Bank ─────────────────────────────────────────
def precompute_log_gabor_filters(img_size, scales, orientations):
    rows, cols = img_size
    u, v = np.meshgrid(
        (np.arange(cols) - cols//2) / (cols/2),
        (np.arange(rows) - rows//2) / (rows/2)
    )
    radius = np.sqrt(u**2 + v**2)
    radius[radius == 0] = 1e-9
    theta = np.arctan2(v, u)
    sigma_f, sigma_theta = 0.65, 0.3
    filter_bank = []
    for scale in scales:
        for orientation in orientations:
            radial  = np.exp(-(np.log(radius/scale))**2 / (2*sigma_f**2))
            td      = np.mod(theta - orientation, np.pi)
            td      = np.minimum(td, np.pi - td)
            angular = np.exp(-td**2 / (2*sigma_theta**2))
            f       = radial * angular
            f[0,0]  = 0
            filter_bank.append(np.fft.ifftshift(f))
    return filter_bank

FILTER_BANK = precompute_log_gabor_filters(
    (BLOCK_SIZE, BLOCK_SIZE), SCALES, ORIENTATIONS
)

# ── Helpers ───────────────────────────────────────────────────────
def to_blocks_gray(arr, bs=32):
    h, w   = arr.shape
    h_c, w_c = (h//bs)*bs, (w//bs)*bs
    arr    = arr[:h_c, :w_c]
    b      = arr.reshape(h_c//bs, bs, w_c//bs, bs)
    return b.transpose(0,2,1,3).reshape(-1, bs, bs)

def to_blocks_rgb(arr, bs=32):
    h, w, c = arr.shape
    h_c, w_c = (h//bs)*bs, (w//bs)*bs
    arr    = arr[:h_c, :w_c]
    b      = arr.reshape(h_c//bs, bs, w_c//bs, bs, c)
    return b.transpose(0,2,1,3,4).reshape(-1, bs, bs, c)

# ── Fully Vectorized Spatial Features ────────────────────────────
def spatial_features_vectorized(gray):
    """
    Compute spatial features over the whole image — fast, no block loop.
    Returns 96-dim vector (mean+std of 48 features aggregated from blocks).
    """
    bs    = BLOCK_SIZE
    blocks = to_blocks_gray(gray.astype(np.float64), bs)  # (N,32,32)
    N     = len(blocks)
    flat  = blocks.reshape(N, -1)

    # ── Basic stats (5 per block) ─────────────────────────────────
    means  = flat.mean(axis=1)
    stds   = flat.std(axis=1)
    # skewness vectorized
    zm     = flat - means[:,None]
    skews  = (zm**3).mean(axis=1) / (stds**3 + 1e-8)
    kurts  = (zm**4).mean(axis=1) / (stds**4 + 1e-8) - 3
    entropies = np.array([shannon_entropy(blocks[i]) for i in range(N)])
    base_f = np.nan_to_num(
        np.stack([means, stds, skews, kurts, entropies], axis=1)
    )  # (N,5)

    # ── Edge features (4 per block) ───────────────────────────────
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float64)
    Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], np.float64)
    edge_f = np.zeros((N,4))
    for i in range(N):
        Gx  = convolve(blocks[i], Kx)
        Gy  = convolve(blocks[i], Ky)
        mag = np.sqrt(Gx**2 + Gy**2)
        edge_f[i] = [mag.mean(), mag.std(), (mag**2).mean(), mag.max()]

    # ── LBP features (15 per block) — vectorized histogram ───────
    lbp_f = np.zeros((N,15))
    for i in range(N):
        lbp       = local_binary_pattern(blocks[i], P=4, R=2, method='uniform')
        hist, _   = np.histogram(lbp, bins=15, range=(0,15), density=True)
        lbp_f[i]  = hist

    # ── Log-Gabor (24 per block) — batch FFT ─────────────────────
    blocks_fft = np.fft.fft2(blocks)        # (N,32,32) all at once
    lg_f       = np.zeros((N,24))
    for fi, filt in enumerate(FILTER_BANK):
        mag = np.abs(np.fft.ifft2(blocks_fft * filt))  # (N,32,32)
        fm  = mag.reshape(N,-1)
        lg_f[:, fi*3]   = fm.mean(axis=1)
        lg_f[:, fi*3+1] = fm.std(axis=1)
        lg_f[:, fi*3+2] = (fm**2).mean(axis=1)

    all_f = np.concatenate([base_f, edge_f, lbp_f, lg_f], axis=1)  # (N,48)
    return np.concatenate([all_f.mean(axis=0), all_f.std(axis=0)])  # 96

# ── Fully Vectorized Spectral Features ───────────────────────────
def spectral_features_vectorized(img):
    """
    Compute spectral features over whole image — fast, no block loop.
    Returns 42-dim vector.
    """
    bs      = BLOCK_SIZE
    blocks  = to_blocks_rgb(img.astype(np.float64), bs)  # (N,32,32,3)
    N       = len(blocks)

    # ── FDD features (9) — computed over whole image at once ─────
    fdd_f = np.zeros((N,9))
    for i in range(N):
        block = blocks[i]
        all_digits = []
        for b in range(block.shape[2]):
            band = np.abs(block[:,:,b]).flatten()
            band = band[(band>0) & np.isfinite(band)]
            if len(band):
                exp  = np.floor(np.log10(band))
                digs = np.floor(band / 10**exp)
                all_digits.append(digs)
        if all_digits:
            all_digits = np.concatenate(all_digits)
            counts, _  = np.histogram(all_digits, bins=np.arange(0.5,10.5))
            total      = counts.sum()
            fdd_f[i]   = counts/total if total>0 else 0

    # ── Color moments (12) — fully vectorized ────────────────────
    moment_f = np.zeros((N,12))
    ch_data  = blocks.reshape(N,-1,3)          # (N, 1024, 3)
    for ch in range(3):
        c = ch_data[:,:,ch]                    # (N,1024)
        moment_f[:,ch*3]   = c.mean(axis=1)
        moment_f[:,ch*3+1] = c.std(axis=1)
        zm  = c - c.mean(axis=1, keepdims=True)
        std = c.std(axis=1) + 1e-8
        moment_f[:,ch*3+2] = (zm**3).mean(axis=1) / std**3

    all_f = np.nan_to_num(
        np.concatenate([fdd_f, moment_f], axis=1)
    )  # (N,21)
    return np.concatenate([all_f.mean(axis=0), all_f.std(axis=0)])  # 42

# ── Main ──────────────────────────────────────────────────────────
def extract_mvg_features(image_path):
    img  = np.array(Image.open(image_path).convert('RGB'))
    gray = np.array(Image.open(image_path).convert('L')).astype(np.float64)
    spat = spatial_features_vectorized(gray)    # 96
    spec = spectral_features_vectorized(img)    # 42
    return np.concatenate([spat, spec])         # 138

def extract_dataset(image_folder, output_path):
    image_folder = Path(image_folder)
    image_paths  = sorted(
    list(image_folder.glob("*.jpg")) +
    list(image_folder.glob("*.JPG")) +
    list(image_folder.glob("*.png")) +
    list(image_folder.glob("*.bmp")) +
    list(image_folder.glob("*.BMP"))
    )
    print(f"Found {len(image_paths)} images")
    names, features = [], []
    for img_path in tqdm(image_paths, desc="Extracting MVG features"):
        try:
            feat = extract_mvg_features(img_path)
            names.append(img_path.name)
            features.append(feat)
        except Exception as e:
            print(f"Skipped {img_path.name}: {e}")
    features = np.array(features)
    np.save(output_path, {"names": names, "features": features})
    print(f"Saved {len(names)} vectors → shape {features.shape}")

if __name__ == "__main__":
    extract_dataset(
        image_folder="data/liveitw/images",
        output_path="results/liveitw_mvg_features.npy"
    )