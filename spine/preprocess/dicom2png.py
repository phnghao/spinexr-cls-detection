import numpy as np
import pydicom 
import cv2 as cv
import os
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pathlib import Path
import argparse
from joblib import Parallel, delayed
from tqdm import tqdm

# Function that convert to png by equalHist
def cvt_equalHist(dcm_file_pth): 
    """
    Convert dicom to png by equalHist
    """
    dcm_file = pydicom.dcmread(dcm_file_pth)
    if dcm_file.BitsStored in (10,12):
        dcm_file.BitsStored = 16
    try:
        dcm_img = dcm_file.pixel_array
        rescaled_img = cv.convertScaleAbs(dcm_img, alpha = 255.0 / dcm_img.max())
    except Exception as error:
        # print(error)
        return None
    
    if dcm_file.PhotometricInterpretation == 'MONOCHROME1':
       rescaled_img = cv.bitwise_not(rescaled_img)
    adjusted_img = cv.equalizeHist(rescaled_img)
    return adjusted_img

# Function converting dicom to png by voi_lut
def cvt_voi_lut(dcm_file_path, voi_lut = True): 
    """
    Convert dicom to png by built-in function apply_voi_lut from pydicom
    """
    dcm = pydicom.dcmread(dcm_file_path)

    if dcm.BitsStored in (10,12):
        dcm.BitsStored = 16
    try:
        dcm_img = apply_voi_lut(dcm.pixel_array, dcm) if voi_lut else dcm.pixel_array
    except Exception as error:
        # print(error)
        return None
    
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        dcm_img = np.amax(dcm_img ) - dcm_img

    dcm_img = (dcm_img - np.min(dcm_img))/ np.max(dcm_img)
    return (dcm_img*255).astype(np.uint8)

# Write results and status to file log
def log(message, log_file):
    # print(str(message))
    with open(log_file, 'a') as file:
       file.write(str(message) + '\n')

def process_dicom_image(dicom_file, outdir, log_file):
    img_ID = Path(dicom_file).stem  

    voi_lut = True
    png = cvt_voi_lut(str(dicom_file))

    if png is None:
        voi_lut = False
        png = cvt_equalHist(str(dicom_file))
        log(f'{str(dicom_file)} cannot apply voi lut (Applied equalHist completely)', log_file)

    has_png = True
    if png is not None:
        outfile = outdir/f'{img_ID}.png'
        cv.imwrite(outfile.as_posix(),png)
        log(f'{str(dicom_file)} applied voi lut completely', log_file)
    return {'image_ID': img_ID, 'has_png': has_png, 'voi_lut': voi_lut}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required = True, type = str)
    parser.add_argument('--output-dir', required = True, type = str)
    parser.add_argument('--cpus', type = int, default= 4)
    parser.add_argument('--log-file', required = True, type = str, help = 'Path to log file')
    parser.add_argument('--debug', action ='store_true')

    args = parser.parse_args()

    inp_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    log_file = args.log_file

    out_dir.mkdir(exist_ok= True, parents= True)
    dicom_files = list(inp_dir.glob('*dicom'))

    np.random.seed(42)
    np.random.shuffle(dicom_files)
    
    if args.debug:
        dicom_files = dicom_files[:20]
        args.cpus = 1

    imple = Parallel(
        n_jobs = args.cpus,
        backend='multiprocessing',
        prefer= 'processes',
        verbose = 0
    )
    do = delayed(process_dicom_image)

    tasks= (do(f, out_dir, log_file) for f in tqdm(dicom_files, desc='Converting', unit='img'))
    res = imple(tasks)

if __name__ == '__main__':
    main()