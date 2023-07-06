import numpy as np
import sys, os

imodel = int(sys.argv[1])
PDBPATH = '/mnt/ceph/users/cmodi/PosteriorDB/'

nchains = 10
path = f'{PDBPATH}//PDB_{imodel:02d}/PDB_{imodel:02d}'
thin = 10
skipcols = 7

diagnostics, samples = [], []
for i in range(nchains):
    print(i)
    f = open(f'{path}.output{i}.csv')
    lines = f.readlines()
    ivals = []
    for l in lines:
        if l[0] != '#':
            vals = l.split(',')
            vals[-1] = vals[-1][:-1]
            try:
                ivals.append(np.array(vals).astype(float))
            except Exception as e:
                #should fail first time when we parse column-title line
                print(e)
                if i == 0 :
                    #make sure that we are splitting at the correct position
                    print('diagnostic parameters')
                    print(vals[:skipcols])
                    print('parameter names')
                    print(vals[skipcols:])
                    with open(f'{path}.samples.meta', "w") as f:
                        f.write("\n".join(vals[skipcols:]))
                    with open(f'{path}.diagnostics.meta', "w") as f:
                        f.write("\n".join(vals[:skipcols]))

    isamples = np.array(ivals)[::thin, skipcols:]
    idiagnostics = np.array(ivals)[::thin, :skipcols]
    samples.append(isamples)
    diagnostics.append(idiagnostics)

samples = np.stack(samples, axis=0)
diagnostics = np.stack(diagnostics, axis=0)
print(samples.shape)
print(diagnostics.shape)
np.save(f'{path}.samples', samples)
np.save(f'{path}.diagnostics', diagnostics)
