"""Experimental stripmap RDA focusing. Output is NOT coregistered to official SLC."""
from prepare_pairs import arguments,prepare_slc,load,save
import numpy as np
from scipy.fft import fft,ifft,fftfreq
import tempfile
from pathlib import Path

def main():
    a=arguments(__doc__).parse_args(); prepare_slc(a)
    src=a.out/a.pol/'03_range'; out=a.out/a.pol/'05_focus'; out.mkdir(parents=True,exist_ok=True)
    paths=list(src.glob('group_*.npy'))
    if not paths: raise FileNotFoundError('Run range_compress.py first')
    grid=load(a.out/a.pol/'04_slc/grid.json'); c=299792458.; wavelength=c/float(grid['radarFrequency']); speed=grid['satellite_speed_mps']
    from sentinel1decoder.constants import F_REF
    for path in paths:
        dest=out/path.name
        if dest.exists(): raise FileExistsError(dest)
        info=load(path.with_suffix('.json')); h=info['first_header']; fs=info['sampling_rate']; x=np.load(path,mmap_mode='r'); na,nr=x.shape
        ranges=(h['Rank']*h['PRI']+h['SWST']+320/(8*F_REF)+np.arange(nr)/fs)*c/2
        fa=fftfreq(na,h['PRI']); D=np.sqrt(1-(wavelength*fa/(2*speed))**2)
        with tempfile.TemporaryDirectory(dir=out) as scratch:
            spec=np.lib.format.open_memmap(Path(scratch)/'doppler.npy',mode='w+',dtype='complex64',shape=x.shape)
            for col in range(0,nr,128): spec[:,col:col+128]=fft(x[:,col:col+128],axis=0)
            spec.flush(); corrected=np.lib.format.open_memmap(Path(scratch)/'rcmc.npy',mode='w+',dtype='complex64',shape=x.shape); indices=np.arange(nr)
            for row in range(na):
                coord=indices+ranges*(1/D[row]-1)/(c/(2*fs))
                corrected[row]=np.interp(coord,indices,spec[row].real,left=0,right=0)+1j*np.interp(coord,indices,spec[row].imag,left=0,right=0)
            corrected.flush(); del spec
            y=np.lib.format.open_memmap(dest,mode='w+',dtype='complex64',shape=x.shape)
            for col in range(0,nr,128):
                filt=np.exp(4j*np.pi*ranges[None,col:col+128]*(D[:,None]-1)/wavelength)
                y[:,col:col+128]=ifft(corrected[:,col:col+128]*filt,axis=0)
            y.flush(); del y,corrected
        save(dest.with_suffix('.json'),{**info,'status':'experimental RDA; unregistered',
            'limitations':['constant satellite speed','zero Doppler centroid','linear RCMC','no SRC or azimuth overlap-save'],
            'diagnostic_only':not info['complete']})
        print('SAVED',dest)
if __name__=='__main__': main()
