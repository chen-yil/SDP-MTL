"""Shared paths only. No access to previous intermediate files."""
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.runtime'))
RAW='S1A_S3_RAW__0SDH_20240101T213605_20240101T213631_051918_0645D9_E9CA'
SLC='S1A_S3_SLC__1SDH_20240101T213606_20240101T213630_051918_0645D9_6D3B'
def arguments(description):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--scene',type=Path,default=ROOT/'Sence1_test_20240101')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/from_products')
    p.add_argument('--pol',choices=['hv','hh'],default='hv')
    return p

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,default=str),encoding='utf8')

def load(path): return json.loads(path.read_text(encoding='utf8'))

def product(scene,stem):
    p=scene/(stem+'.zip')
    if not p.exists(): raise FileNotFoundError(p)
    return p


"""01: Extract exact requested RAW product; parse packet headers with byte offsets."""

import zipfile,shutil

def prepare_raw(a):
    from sentinel1decoder.l0decoder import Level0Decoder,NoMorePacketsException
    out=a.out/a.pol/'01_packets'; out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(product(a.scene,RAW)) as z:
        names=[n for n in z.namelist() if f'-{a.pol}-' in n and n.endswith('.dat') and not n.endswith(('-annot.dat','-index.dat'))]
        if len(names)!=1: raise ValueError(names)
        dest=out/Path(names[0]).name
        if not dest.exists() or dest.stat().st_size!=z.getinfo(names[0]).file_size:
            with z.open(names[0]) as src,dest.open('wb') as dst: shutil.copyfileobj(src,dst,16*1024*1024)
    decoder=Level0Decoder(str(dest)); records=[]; groups=[]; previous=None; group=0
    with dest.open('rb') as f:
        while True:
            offset=f.tell()
            try: h,_=decoder._read_single_packet(f)
            except NoMorePacketsException: break
            key=(h['Swath Number'],h['Number of Quads'],h['Signal Type'])
            if key!=previous:
                group+=1; previous=key
                groups.append({'group':group,'signal_type':int(key[2]),'quads':int(key[1]),'packets':0,'first_header':h})
            groups[-1]['packets']+=1
            records.append({'offset':offset,'group':group,**h})
    save(out/'groups.json',groups)
    with (out/'packets.jsonl').open('w') as f:
        for r in records: f.write(json.dumps(r)+'\n')
    save(out/'source.json',{'raw':str(dest.resolve()),'archive':str(product(a.scene,RAW).resolve()),'polarization':a.pol})
    print([(g['group'],g['packets'],g['signal_type']) for g in groups],flush=True)


"""04: Read official SLC and its orbit/grid metadata directly from the requested ZIP."""

import zipfile,xml.etree.ElementTree as ET
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image

def prepare_slc(a):
    out=a.out/a.pol/'04_slc'; out.mkdir(parents=True,exist_ok=True)
    archive=product(a.scene,SLC)
    with zipfile.ZipFile(archive) as z:
        ann=next(n for n in z.namelist() if '/annotation/s1' in n and f'-{a.pol}-' in n)
        data=next(n for n in z.namelist() if '/measurement/' in n and f'-{a.pol}-' in n and n.endswith('.tiff'))
        xml=z.read(ann); (out/'annotation.xml').write_bytes(xml); r=ET.fromstring(xml)
    tags=['azimuthTimeInterval','slantRangeTime','radarFrequency','rangeSamplingRate','rangePixelSpacing','azimuthPixelSpacing','productFirstLineUtcTime']
    info={t:r.findtext('.//'+t) for t in tags}
    velocities=[]
    for o in r.findall('.//orbitList/orbit'):
        v=[float(o.findtext('velocity/'+axis)) for axis in ['x','y','z']]; velocities.append(np.linalg.norm(v))
    info['satellite_speed_mps']=float(np.mean(velocities))
    with rasterio.open('/vsizip/'+archive.resolve().as_posix()+'/'+data) as ds:
        info.update(width=ds.width,height=ds.height,uri=ds.name)
        x=ds.read(1,out_shape=(ds.height//16,ds.width//16)); db=20*np.log10(np.maximum(np.abs(x),1e-8))
        lo,hi=np.percentile(db,[2,99.5]); Image.fromarray((255*np.clip((db-lo)/(hi-lo),0,1)).astype('uint8')).save(out/'preview.png')
    save(out/'grid.json',info); print('SLC',info['height'],info['width'],flush=True)


"""10: Map SLC windows to raw packet/sample support using time, orbit and geolocation.
Output is a conservative geometric estimate, NOT a measured subpixel registration.
"""

from datetime import datetime,timezone
import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import RegularGridInterpolator,CubicHermiteSpline

C=299792458.
GPS_EPOCH=datetime(1980,1,6,tzinfo=timezone.utc).timestamp()

def utc(s): return datetime.fromisoformat(s.replace('Z','+00:00')).replace(tzinfo=timezone.utc).timestamp()

def ecef(lat,lon,h):
    lat,lon=np.deg2rad(lat),np.deg2rad(lon); e2=6.6943799901413165e-3
    n=6378137./np.sqrt(1-e2*np.sin(lat)**2)
    return np.array([(n+h)*np.cos(lat)*np.cos(lon),(n+h)*np.cos(lat)*np.sin(lon),(n*(1-e2)+h)*np.sin(lat)])

def starts(n,k,s): return sorted(set(range(0,n-k+1,s))|{n-k}) if n>=k else []

def map_windows(a):
    base=a.out/a.pol; grid=load(base/'04_slc/grid.json'); r=ET.parse(base/'04_slc/annotation.xml')
    # Hermite orbit interpolation uses annotated Earth-fixed positions AND velocities.
    orbits=r.findall('.//orbitList/orbit'); ot=np.array([utc(o.findtext('time')) for o in orbits]); origin=ot[0]
    pos=np.array([[float(o.findtext('position/'+s)) for s in ['x','y','z']] for o in orbits])
    vel=np.array([[float(o.findtext('velocity/'+s)) for s in ['x','y','z']] for o in orbits])
    orbit=CubicHermiteSpline(ot-origin,pos,vel,extrapolate=False)
    points=r.findall('.//geolocationGridPoint'); rr=sorted({int(q.findtext('line')) for q in points}); cc=sorted({int(q.findtext('pixel')) for q in points})
    xyz=np.full((len(rr),len(cc),3),np.nan)
    for q in points:
        xyz[rr.index(int(q.findtext('line'))),cc.index(int(q.findtext('pixel')))]=ecef(*[float(q.findtext(s)) for s in ['latitude','longitude','height']])
    if not np.isfinite(xyz).all(): raise ValueError('Nonrectangular geolocation grid')
    geo=RegularGridInterpolator((rr,cc),xyz,bounds_error=True)
    fm=r.findall('.//azimuthFmRate'); bandwidths=[float(q.text) for q in r.findall('.//azimuthProcessing/processingBandwidth')]
    if not bandwidths or not fm: raise ValueError('Missing aperture support annotations')
    bandwidth=max(bandwidths)
    packets=[]
    for line in (base/'01_packets/packets.jsonl').read_text().splitlines():
        h=json.loads(line)
        if h['Signal Type']==0:
            h['utc']=GPS_EPOCH+h['Coarse Time']+h['Fine Time']-a.gps_utc
            packets.append(h)
    times=np.array([h['utc'] for h in packets])
    if np.any(np.diff(times)<=0): raise ValueError('Nonmonotonic RAW pulse times')
    t0=utc(grid['productFirstLineUtcTime']); dt=float(grid['azimuthTimeInterval']); fs_slc=float(grid['rangeSamplingRate']); tau0=float(grid['slantRangeTime'])
    if not (times[0]-5<t0<times[-1]+5): raise ValueError('RAW/SLC timing mismatch: check GPS to UTC conversion')
    from sentinel1decoder.utilities import range_dec_to_sample_rate
    from sentinel1decoder.constants import F_REF
    records=[]; rejected=[]
    for row in starts(grid['height'],a.patch_size,a.stride):
        for col in starts(grid['width'],a.patch_size,a.stride):
            if a.limit and len(records)>=a.limit: break
            size=a.patch_size; ta=t0+row*dt; tb=t0+(row+size-1)*dt
            f=min(fm,key=lambda f:abs(utc(f.findtext('azimuthTime'))-(ta+tb)/2))
            coeff=np.array([float(v) for v in f.findtext('azimuthFmRatePolynomial').split()]); ft0=float(f.findtext('t0'))
            rates=np.polynomial.polynomial.polyval(tau0+np.array([col,col+size-1])/fs_slc-ft0,coeff)
            half=bandwidth/(2*min(abs(rates)))+a.azimuth_margin_seconds
            low,high=ta-half,tb+half
            i=int(np.searchsorted(times,low)); j=int(np.searchsorted(times,high,side='right'))
            reason=None
            if i==0 or j==len(times) or j<=i: reason='aperture extends beyond RAW acquisition'
            elif np.max(np.diff(times[i-1:j+1]))>1.5*np.median(np.diff(times)): reason='missing pulses in aperture'
            if reason:
                rejected.append({'row':row,'column':col,'reason':reason}); continue
            # Sample target patch footprint on a 3x3 grid; orbit determines migration per pulse.
            coords=np.array([(y,x) for y in np.linspace(row,row+size-1,3) for x in np.linspace(col,col+size-1,3)])
            targets=geo(coords); segments=[]
            for h in packets[i:j]:
                fs=range_dec_to_sample_rate(int(h['Range Decimation']))
                receive_start=h['Rank']*h['PRI']+h['SWST']+320/(8*F_REF)
                satellite=orbit(h['utc']+receive_start/2-origin)
                tau=2*np.linalg.norm(targets-satellite,axis=1)/C
                start=int(np.floor((tau.min()-receive_start)*fs))-a.range_margin_samples
                stop=int(np.ceil((tau.max()+h['Tx Pulse Length']-receive_start)*fs))+a.range_margin_samples+1
                n=h['Number of Quads']*2
                if not np.isfinite(tau).all() or start<0 or stop>n:
                    reason='footprint plus chirp support outside RAW receive window'; break
                segments.append({'offset':h['offset'],'group':h['group'],'utc':h['utc'],'sample_start':start,'sample_stop':stop})
            if reason:
                rejected.append({'row':row,'column':col,'reason':reason}); continue
            records.append({'id':f'r{row:06d}_c{col:06d}','slc_window':{'row':row,'column':col,'height':size,'width':size},
                'aperture_half_seconds':float(half),'raw_segments':segments,'mapping_status':'geometric_estimate_not_coregistration_verified'})
        if a.limit and len(records)>=a.limit: break
    report={'raw_product':RAW,'slc_product':SLC,'polarization':a.pol,'gps_utc_seconds':a.gps_utc,
        'mapping_status':'geometric_estimate_not_coregistration_verified',
        'assumptions':['GPS timestamp to UTC; packet time treated as pulse reference',
        'Earth-fixed annotated orbit; straight-line monostatic propagation at approximate midpoint',
        '3x3 interpolated geolocation footprint; nearest azimuth FM-rate annotation',
        'processed bandwidth defines aperture support plus explicit margin; no empirical registration correction',
        'no ionosphere/troposphere correction, bistatic refinement or verified moving-ship alignment'],
        'azimuth_margin_seconds':a.azimuth_margin_seconds,'range_margin_samples':a.range_margin_samples,
        'pairs':records,'rejected':rejected}
    save(base/'10_mapping/mapping.json',report)
    print('geometric mappings',len(records),'rejected',len(rejected),flush=True)
    if not records: raise RuntimeError('No supported SLC windows; inspect mapping.json')


"""11: Export linked SLC target and the supporting RAW aperture, directly from products.
RAW has a larger, variable shape. Padding is explicit and accompanied by a validity mask.
Geometric pairing is NOT a claim of verified pixel-level coregistration.
"""

import numpy as np
import rasterio
from rasterio.windows import Window

def export_pairs(a):
    base=a.out/a.pol; mapping=load(base/'10_mapping/mapping.json'); grid=load(base/'04_slc/grid.json')
    if mapping['polarization']!=a.pol: raise ValueError('Polarization mismatch')
    from sentinel1decoder.l0decoder import Level0Decoder
    from sentinel1decoder._user_data_decoder import UserDataDecoder
    decoder=Level0Decoder(load(base/'01_packets/source.json')['raw'])
    out=base/'11_linked_pairs'; out.mkdir(parents=True,exist_ok=True); manifest=[]
    pairs=mapping['pairs'][:a.limit] if a.limit else mapping['pairs']
    with open(decoder.filename,'rb') as f,rasterio.open(grid['uri']) as ds:
        for pair in pairs:
            segs=pair['raw_segments']; groups=sorted({s['group'] for s in segs})
            # Common native range origin within each packet-layout group; groups may change SWST.
            bounds={g:(min(s['sample_start'] for s in segs if s['group']==g),max(s['sample_stop'] for s in segs if s['group']==g)) for g in groups}
            width=max(hi-lo for lo,hi in bounds.values())
            raw=np.zeros((len(segs),width),dtype='complex64'); valid=np.zeros(raw.shape,dtype=bool)
            starts=[]; headers=[]
            for row,s in enumerate(segs):
                f.seek(s['offset']); h,payload=decoder._read_single_packet(f)
                x=np.asarray(UserDataDecoder(payload,int(h['BAQ Mode']),int(h['Number of Quads'])).decode(),dtype='complex64')
                lo,hi=bounds[s['group']]
                if lo<0 or hi>len(x) or not np.isfinite(x).all(): raise ValueError('Invalid packet support')
                raw[row,:hi-lo]=x[lo:hi]; valid[row,:hi-lo]=True; starts.append(lo); headers.append(h)
                if row%200==0: print(pair['id'],row,'/',len(segs),flush=True)
            w=pair['slc_window']; slc=ds.read(1,window=Window(w['column'],w['row'],w['width'],w['height']))
            scale=float(np.percentile(np.abs(raw[valid]),99.5))
            if scale<=0: raise ValueError('Invalid normalization')
            path=out/(pair['id']+'.npz')
            np.savez_compressed(path,raw_i=raw.real,raw_q=raw.imag,raw_valid=valid,
                raw_iq_normalized=np.stack([raw.real/scale,raw.imag/scale]),raw_normalization_scale=scale,
                raw_group=np.array([s['group'] for s in segs]),raw_packet_offsets=np.array([s['offset'] for s in segs],dtype='int64'),
                raw_sample_start=np.array(starts),raw_utc=np.array([s['utc'] for s in segs]),
                raw_swst=np.array([h['SWST'] for h in headers]),raw_pri=np.array([h['PRI'] for h in headers]),
                raw_rank=np.array([h['Rank'] for h in headers]),raw_range_decimation=np.array([h['Range Decimation'] for h in headers]),
                raw_swap_flag=np.array([h['Swap Flag'] for h in headers]),
                slc_i=slc.real.astype('float32'),slc_q=slc.imag.astype('float32'),slc_intensity=np.abs(slc).astype('float32')**2,
                slc_row=np.int64(w['row']),slc_column=np.int64(w['column']),
                mapping_status=np.array(pair['mapping_status']))
            save(out/(pair['id']+'.json'),{**pair,'raw_bounds_by_group':bounds,'raw_shape':list(raw.shape),
                'slc_shape':list(slc.shape),'normalization_scale':scale,'padding':'raw_valid=False is padding, never measured echo',
                'coordinates':'RAW columns begin at raw_sample_start per pulse; preserve raw_swst/rank/pri and group for timing',
                'mapping_file':str((base/'10_mapping/mapping.json').resolve()),'training_ready':False})
            manifest.append({'file':path.name,'raw_shape':list(raw.shape),'slc_shape':list(slc.shape),'mapping_status':pair['mapping_status']})
            save(out/'manifest.json',manifest)
            print('exported',path,flush=True)


"""12: Validate linked-pair provenance by independently re-decoding sampled original packets."""

import numpy as np
import rasterio
from rasterio.windows import Window

def validate_pairs(a):
    base=a.out/a.pol
    from sentinel1decoder.l0decoder import Level0Decoder
    from sentinel1decoder._user_data_decoder import UserDataDecoder
    src=load(base/'01_packets/source.json'); grid=load(base/'04_slc/grid.json'); out=base/'11_linked_pairs'
    files=sorted(out.glob('r*_c*.npz'))
    if not files: raise FileNotFoundError('No linked pairs')
    decoder=Level0Decoder(src['raw']); results=[]
    with open(src['raw'],'rb') as f,rasterio.open(grid['uri']) as ds:
        for path in files:
            with np.load(path,allow_pickle=False) as z:
                raw=z['raw_i']+1j*z['raw_q']; mask=z['raw_valid']; slc=z['slc_i']+1j*z['slc_q']
                if mask.shape!=raw.shape or not np.isfinite(raw).all(): raise ValueError('Invalid RAW arrays')
                if np.any(raw[~mask]!=0): raise ValueError('Invalid padding')
                if not np.all(np.diff(z['raw_utc'])>0): raise ValueError('Nonmonotonic pulse times')
                if not np.allclose(z['raw_iq_normalized'],np.stack([raw.real,raw.imag])/z['raw_normalization_scale']): raise ValueError('Normalization mismatch')
                for row in sorted({0,len(raw)//2,len(raw)-1}):
                    f.seek(int(z['raw_packet_offsets'][row])); h,payload=decoder._read_single_packet(f)
                    original=np.asarray(UserDataDecoder(payload,int(h['BAQ Mode']),int(h['Number of Quads'])).decode(),dtype='complex64')
                    start=int(z['raw_sample_start'][row]); n=int(mask[row].sum())
                    if not mask[row,:n].all() or mask[row,n:].any(): raise ValueError('Invalid support mask')
                    if not np.array_equal(raw[row,:n],original[start:start+n]): raise ValueError('RAW provenance mismatch')
                ref=ds.read(1,window=Window(int(z['slc_column']),int(z['slc_row']),slc.shape[1],slc.shape[0]))
                if not np.array_equal(slc,ref): raise ValueError('SLC provenance mismatch')
                if not np.allclose(z['slc_intensity'],np.abs(slc)**2): raise ValueError('Intensity mismatch')
                results.append({'file':path.name,'raw_shape':list(raw.shape),'slc_shape':list(slc.shape),
                    'sampled_raw_provenance':'passed (first/middle/last pulse)', 'slc_provenance':'passed',
                    'normalization_and_masks':'passed','mapping_status':str(z['mapping_status']),
                    'geometric_registration_accuracy':'not assessed by this check'})
    save(base/'12_validation/report.json',results); print(results)



def main():
    """Produce only linked RAW/SLC NPZ files; intermediate products are temporary."""
    import tempfile,zipfile,io,shutil,logging
    from types import SimpleNamespace
    p=arguments('Export corresponding RAW aperture and SLC patch into one NPZ per sample')
    p.set_defaults(out=ROOT/'outputs/paired_samples')
    p.add_argument('--limit',type=int,default=1,help='0=all supported windows')
    p.add_argument('--patch-size',type=int,default=512)
    p.add_argument('--stride',type=int,default=410)
    p.add_argument('--gps-utc',type=int,default=18)
    p.add_argument('--azimuth-margin-seconds',type=float,default=.1)
    p.add_argument('--range-margin-samples',type=int,default=64)
    a=p.parse_args()
    if a.limit<0 or a.patch_size<1 or a.stride<1 or a.azimuth_margin_seconds<0 or a.range_margin_samples<0:
        p.error('Invalid patch, limit or margin options')
    logging.basicConfig(level=logging.WARNING,handlers=[logging.StreamHandler()])
    final=a.out/a.pol
    final.mkdir(parents=True,exist_ok=True)
    if any(final.glob('r*_c*.npz')):
        p.error('Output already contains paired samples; choose a new --out directory')
    # Keep extraction and mapping scratch space inside the explicitly selected output root.
    with tempfile.TemporaryDirectory(prefix='_working_',dir=final) as scratch:
        work=SimpleNamespace(**vars(a)); work.out=Path(scratch)
        prepare_raw(work)
        prepare_slc(work)
        map_windows(work)
        export_pairs(work)
        validate_pairs(work)
        base=work.out/a.pol
        validation=load(base/'12_validation/report.json')
        mapping=load(base/'10_mapping/mapping.json')
        for record in validation:
            source=base/'11_linked_pairs'/record['file']
            metadata=load(source.with_suffix('.json'))
            metadata['mapping_assumptions']=mapping['assumptions']
            metadata['gps_utc_seconds']=mapping['gps_utc_seconds']
            metadata['raw_product']=RAW; metadata['slc_product']=SLC
            with zipfile.ZipFile(source,'a',compression=zipfile.ZIP_DEFLATED) as archive:
                for key,value in [('metadata_json',metadata),('validation_json',record)]:
                    stream=io.BytesIO()
                    np.lib.format.write_array(stream,np.array(json.dumps(value)),allow_pickle=False)
                    archive.writestr(key+'.npy',stream.getvalue())
            destination=final/source.name
            if destination.exists():
                raise FileExistsError(f'{destination} exists; use a new --out directory')
            shutil.move(str(source),str(destination))
            print('SAVED',destination,flush=True)

if __name__=='__main__': main()

