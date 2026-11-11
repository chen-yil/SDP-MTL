"""Export official SLC 512x512 windows. These are image samples, not raw echoes."""
from prepare_pairs import arguments,prepare_slc,load,save,starts,SLC
import numpy as np
import rasterio
from rasterio.windows import Window

def main():
    p=arguments(__doc__); p.add_argument('--limit',type=int,default=16,help='0=all'); a=p.parse_args()
    if a.limit<0: p.error('limit must be >=0')
    prepare_slc(a); grid=load(a.out/a.pol/'04_slc/grid.json'); out=a.out/a.pol/'slc_patches'; out.mkdir(parents=True,exist_ok=True); records=[]
    with rasterio.open(grid['uri']) as ds:
        sample=ds.read(1,out_shape=(max(1,ds.height//32),max(1,ds.width//32))); scale=float(np.percentile(np.abs(sample),99.5))
        if scale<=0: raise ValueError('Invalid normalization')
        for row in starts(ds.height,512,410):
            for col in starts(ds.width,512,410):
                if a.limit and len(records)>=a.limit: break
                dest=out/f'r{row:06d}_c{col:06d}.npz'
                if dest.exists(): raise FileExistsError(dest)
                x=ds.read(1,window=Window(col,row,512,512))
                np.savez_compressed(dest,slc_i=x.real.astype('float32'),slc_q=x.imag.astype('float32'),
                    intensity=np.abs(x).astype('float32')**2,slc_iq_normalized=np.stack([x.real/scale,x.imag/scale]).astype('float32'),
                    normalization_scale=scale,row=row,column=col)
                records.append({'file':dest.name,'row':row,'column':col})
            if a.limit and len(records)>=a.limit: break
    save(out/'manifest.json',{'source':SLC,'polarization':a.pol,'scale':scale,'patches':records})
    print('SAVED',len(records),'SLC patches')
if __name__=='__main__': main()
