"""Range matched filtering of decoded echoes. Run decode_raw.py first."""
from prepare_pairs import arguments,load,save
import numpy as np
from scipy.fft import fft,ifft,next_fast_len

def main():
    a=arguments(__doc__).parse_args()
    from sentinel1decoder.utilities import range_dec_to_sample_rate
    src=a.out/a.pol/'02_echo'; out=a.out/a.pol/'03_range'; out.mkdir(parents=True,exist_ok=True)
    paths=[p for p in src.glob('group_*.npy') if '.partial.' not in p.name]
    if not paths: raise FileNotFoundError('Run decode_raw.py first')
    for path in paths:
        dest=out/path.name
        if dest.exists(): raise FileExistsError(dest)
        info=load(path.with_suffix('.json')); h=info['first_header']; fs=range_dec_to_sample_rate(int(h['Range Decimation']))
        n=round(h['Tx Pulse Length']*fs); t=np.arange(n)/fs
        chirp=np.exp(2j*np.pi*(h['Tx Pulse Start Frequency']*t+.5*h['Tx Ramp Rate']*t*t))
        x=np.load(path,mmap_mode='r'); nf=next_fast_len(x.shape[1]+n-1)
        filt=fft(chirp[::-1].conj()/np.sqrt(n),nf)
        y=np.lib.format.open_memmap(dest,mode='w+',dtype='complex64',shape=x.shape)
        for row in range(0,len(x),32):
            conv=ifft(fft(x[row:row+32],nf,axis=1)*filt,axis=1)
            y[row:row+32]=conv[:,n-1:n-1+x.shape[1]]
        y.flush(); del y
        save(dest.with_suffix('.json'),{**info,'sampling_rate':fs,'matched_filter_samples':n,
            'range_coordinate':'chirp start; final chirp-length samples have truncated support'})
        print('SAVED',dest)
if __name__=='__main__': main()
