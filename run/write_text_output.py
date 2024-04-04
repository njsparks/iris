# %% turn iris numpy output into 1 timestep per line text output.
import os
import numpy as np
sep=' '
ext='.txt'
def write_tcs_ens(fid,tcs_ens,basin,tcID,year_num):
        for tcs_years in tcs_ens:
            for  tcs_year in tcs_years:
                for tc_num, tc in enumerate(tcs_year):
                    nt = len(tc['lon'])
                    for i in range(nt):
                        print(tcID,basin,year_num, tc_num, tc['month'],  i, tc['lon'][i], tc['lat'][i], tc['vmax'][i], tc['pmin'][i], tc['rmw'][i], tc['r18'][i], file=fid,sep=sep)
                    tcID+=1
                year_num+=1
        return tcID,year_num

def write(inpathbase,outpathbase,bstrs,nEns,nYperEns,nYperBlock):
    if not  os.path.exists(outpathbase):
        os.makedirs(outpathbase)

    # nYperEns, number of years per ensemble
    # nEns, number of simulation ensembles
    # nYperBlock, number of years per output block

    nY=nYperEns*nEns # total number of simulated years
    nBlocks=int(nY/nYperBlock) # number of output blocks
    nEnsPerBlock=int(nEns/nBlocks)
    
    outfnamebase = 'IRIS_'


    # sep='\t'
    # ext='.tab'

    headStr=['#tcid','basin', 'year', 'tc', 'month', 'timestep', 'lon', 'lat', 'vmax', 'pmin', 'rmw', 'r18']

    tcID = 0  # global tc index, spanning all basins and blocks 
    for bx in range(len(bstrs)):
        year_num=0 # reset year number to 0 for each basin
        print()
        fnamebase = 'results-'+bstrs[bx]+'-'
        print(fnamebase)
        for xBlock in range(nBlocks):
            print('Block #',xBlock)

            outfname = outfnamebase+bstrs[bx]+'_'+ str(nYperBlock) +'Y_n'+str(xBlock)+ext
            outpath = outpathbase+outfname

            # read iris out npy
            tcs_ens=[]
            for xEns in range(xBlock*nEnsPerBlock,(xBlock+1)*nEnsPerBlock):
                fnum = "{:04d}".format(xEns)
                fname = fnamebase+fnum+'.npy'
                path = inpathbase+fname
                res=np.load(path,allow_pickle=True).item()
                tcs_ens.append(res['tcs'])

            # write txt block
            with open(outpath, 'w') as fid:
                print(*headStr,file=fid,sep=sep)
                tcID,year_num=write_tcs_ens(fid,tcs_ens,bstrs[bx],tcID,year_num)

