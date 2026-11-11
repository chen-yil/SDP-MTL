"""Decode the specified RAW ZIP into complex64 echo arrays, one per acquisition segment."""
from prepare_pairs import arguments,prepare_raw,load,save
import numpy as np

def main():
    p=arguments(__doc__); p.add_argument('--limit-lines',type=int,default=0,help='0=all; positive=diagnostic prefix per segment'); a=p.parse_args()
    if a.limit_lines<0: p.error('limit-lines must be >=0')
    from sentinel1decoder.l0decoder import Level0Decoder
    from sentinel1decoder._user_data_decoder import UserDataDecoder
    prepare_raw(a); base=a.out/a.pol; src=base/'01_packets'; out=base/'02_echo'; out.mkdir(parents=True,exist_ok=True)
    decoder=Level0Decoder(load(src/'source.json')['raw'])
    packets=[__import__('json').loads(s) for s in (src/'packets.jsonl').read_text().splitlines()]
    with open(decoder.filename,'rb') as f:
        for g in load(src/'groups.json'):
            if g['signal_type']!=0: continue
            headers=[h for h in packets if h['group']==g['group']]
            if a.limit_lines: headers=headers[:a.limit_lines]
            stem=f"group_{g['group']:02d}"; path=out/(stem+'.npy')
            if path.exists(): raise FileExistsError(path)
            partial=out/(stem+'.partial.npy'); x=np.lib.format.open_memmap(partial,mode='w+',dtype='complex64',shape=(len(headers),2*g['quads']))
            for row,h in enumerate(headers):
                f.seek(h['offset']); decoded,payload=decoder._read_single_packet(f)
                z=np.asarray(UserDataDecoder(payload,int(decoded['BAQ Mode']),int(decoded['Number of Quads'])).decode(),dtype='complex64')
                if z.shape!=(x.shape[1],) or not np.isfinite(z).all(): raise ValueError('Invalid decoded packet')
                x[row]=z
                if row%1000==0: print(stem,row,'/',len(headers),flush=True)
            x.flush(); del x; partial.replace(path)
            save(path.with_suffix('.json'),{**g,'decoded_lines':len(headers),'complete':len(headers)==g['packets'],'headers':headers})
            print('SAVED',path)
if __name__=='__main__': main()
